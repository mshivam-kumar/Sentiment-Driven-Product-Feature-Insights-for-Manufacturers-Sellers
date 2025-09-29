#!/bin/bash

# Setup script for Sentiment-Driven Product Feature Insights

set -e

echo "🚀 Setting up Sentiment-Driven Product Feature Insights..."

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ $(echo "$PYTHON_VERSION < 3.10" | bc -l) -eq 1 ]]; then
    echo "❌ Python 3.10+ is required. Current version: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION found"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is required but not installed."
    exit 1
fi

echo "✅ AWS CLI found"

# Check Terraform
if ! command -v terraform &> /dev/null; then
    echo "❌ Terraform is required but not installed."
    exit 1
fi

echo "✅ Terraform found"

# Install Python dependencies
echo "📦 Installing Python dependencies..."

# Install spaCy model
python3 -m spacy download en_core_web_sm

# Install requirements for each module
pip3 install -r data_ingest/requirements.txt
pip3 install -r models/aspect_extractor/requirements.txt
pip3 install -r models/sentiment/requirements.txt
pip3 install -r inference/lambda_inference/requirements.txt
pip3 install -r dashboard/requirements.txt

# Install development dependencies
pip3 install pytest pytest-cov flake8 black isort

echo "✅ Python dependencies installed"

# Set up environment variables
echo "🔧 Setting up environment variables..."

if [ ! -f .env ]; then
    cat > .env << EOF
# AWS Configuration
AWS_REGION=us-east-1
S3_BUCKET=your-bucket-name
DYNAMODB_TABLE=product_sentiment_insights

# API Configuration
API_BASE_URL=https://your-api-gateway-url.amazonaws.com/dev

# Model Configuration
MODEL_VERSION=v1
CONFIDENCE_THRESHOLD=0.5
SENTIMENT_THRESHOLD=0.3
EOF
    echo "📝 Created .env file. Please update with your values."
fi

# Run tests
echo "🧪 Running tests..."

# Run unit tests
python3 -m pytest tests/test_aspect_extraction.py -v
python3 -m pytest tests/test_sentiment_analysis.py -v
python3 -m pytest tests/test_integration.py -v

echo "✅ Tests passed"

# Deploy infrastructure
echo "🏗️ Deploying infrastructure..."

cd infra
terraform init
terraform plan
echo "📋 Review the Terraform plan above. Press Enter to continue or Ctrl+C to cancel."
read -r
terraform apply

echo "✅ Infrastructure deployed"

# Get outputs
API_URL=$(terraform output -raw api_gateway_url)
S3_BUCKET=$(terraform output -raw s3_raw_bucket)
DYNAMODB_TABLE=$(terraform output -raw dynamodb_table)

echo "📊 Infrastructure outputs:"
echo "  API URL: $API_URL"
echo "  S3 Bucket: $S3_BUCKET"
echo "  DynamoDB Table: $DYNAMODB_TABLE"

# Update .env with actual values
sed -i "s|your-bucket-name|$S3_BUCKET|g" ../.env
sed -i "s|https://your-api-gateway-url.amazonaws.com/dev|$API_URL|g" ../.env

cd ..

# Test data ingestion
echo "📥 Testing data ingestion..."

cd data_ingest
python3 download_and_upload.py \
  --dataset_id "McAuley-Lab/Amazon-Reviews-2023" \
  --subset "raw_review_All_Beauty" \
  --s3_bucket "$S3_BUCKET" \
  --s3_prefix "raw/All_Beauty" \
  --num_samples 100

echo "✅ Data ingestion completed"

cd ..

# Test aspect extraction
echo "🔍 Testing aspect extraction..."

cd models/aspect_extractor
python3 infer_aspect.py

echo "✅ Aspect extraction tested"

cd ../..

# Test sentiment analysis
echo "😊 Testing sentiment analysis..."

cd models/sentiment
python3 infer_sentiment.py

echo "✅ Sentiment analysis tested"

cd ../..

# Test API
echo "🌐 Testing API..."

# Wait for API to be ready
sleep 30

# Test health endpoint
curl -f "$API_URL/health" || echo "⚠️ API health check failed"

# Test product endpoint
curl -f "$API_URL/sentiment/product/B00YQ6X8EO" || echo "⚠️ Product endpoint test failed"

echo "✅ API tests completed"

# Start dashboard
echo "📊 Starting dashboard..."

cd dashboard
echo "🚀 Dashboard will be available at http://localhost:8501"
echo "Press Ctrl+C to stop the dashboard"

streamlit run streamlit_app.py

echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Update the .env file with your actual values"
echo "2. Run the data ingestion pipeline"
echo "3. Monitor the processing pipeline"
echo "4. Access the dashboard at http://localhost:8501"
echo "5. Use the API endpoints for integration"
echo ""
echo "📚 Documentation:"
echo "- README.md: Project overview and setup"
echo "- docs/model_card.md: Model details and performance"
echo "- docs/operating_runbook.md: Operations guide"
echo "- docs/athena_queries.sql: Sample queries for data analysis"
