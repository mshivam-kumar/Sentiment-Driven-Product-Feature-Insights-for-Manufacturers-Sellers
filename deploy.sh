#!/bin/bash

# Sentiment Analysis App Deployment Script
# Supports multiple deployment options: Docker, AWS ECS, Heroku, Streamlit Cloud

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command_exists aws; then
        print_warning "AWS CLI is not installed. AWS deployment will not be available."
    fi
    
    if ! command_exists git; then
        print_error "Git is not installed. Please install Git first."
        exit 1
    fi
    
    print_success "Prerequisites check completed."
}

# Function to build Docker image
build_docker_image() {
    print_status "Building Docker image..."
    
    # Build the image
    docker build -t sentiment-analysis-app:latest .
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully!"
    else
        print_error "Failed to build Docker image."
        exit 1
    fi
}

# Function to run locally with Docker
deploy_local_docker() {
    print_status "Deploying locally with Docker..."
    
    # Stop any existing container
    docker stop sentiment-analysis-app 2>/dev/null || true
    docker rm sentiment-analysis-app 2>/dev/null || true
    
    # Run the container
    docker run -d \
        --name sentiment-analysis-app \
        -p 8501:8501 \
        -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
        -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
        -e AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
        sentiment-analysis-app:latest
    
    if [ $? -eq 0 ]; then
        print_success "App deployed locally! Access it at http://localhost:8501"
    else
        print_error "Failed to deploy locally."
        exit 1
    fi
}

# Function to deploy to AWS ECS
deploy_aws_ecs() {
    print_status "Deploying to AWS ECS..."
    
    if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
        print_error "AWS credentials not set. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
        exit 1
    fi
    
    # Get AWS account ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    if [ $? -ne 0 ]; then
        print_error "Failed to get AWS account ID. Please check your AWS credentials."
        exit 1
    fi
    
    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories --repository-names sentiment-analysis-app 2>/dev/null || \
    aws ecr create-repository --repository-name sentiment-analysis-app
    
    # Login to ECR
    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
    
    # Tag and push image
    docker tag sentiment-analysis-app:latest $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/sentiment-analysis-app:latest
    docker push $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/sentiment-analysis-app:latest
    
    print_success "Image pushed to ECR successfully!"
    print_status "You can now create an ECS service using the ECS task definition in infra/ecs-task-definition.json"
}

# Function to deploy to Heroku
deploy_heroku() {
    print_status "Deploying to Heroku..."
    
    if ! command_exists heroku; then
        print_error "Heroku CLI is not installed. Please install it first: https://devcenter.heroku.com/articles/heroku-cli"
        exit 1
    fi
    
    # Login to Heroku
    heroku login
    
    # Create app if it doesn't exist
    if [ -z "$HEROKU_APP_NAME" ]; then
        HEROKU_APP_NAME="sentiment-analysis-$(date +%s)"
    fi
    
    heroku create $HEROKU_APP_NAME 2>/dev/null || print_warning "App $HEROKU_APP_NAME might already exist."
    
    # Set environment variables
    if [ ! -z "$AWS_ACCESS_KEY_ID" ]; then
        heroku config:set AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" --app $HEROKU_APP_NAME
    fi
    
    if [ ! -z "$AWS_SECRET_ACCESS_KEY" ]; then
        heroku config:set AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" --app $HEROKU_APP_NAME
    fi
    
    if [ ! -z "$AWS_DEFAULT_REGION" ]; then
        heroku config:set AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" --app $HEROKU_APP_NAME
    fi
    
    # Deploy
    git push heroku main
    
    print_success "App deployed to Heroku! Access it at https://$HEROKU_APP_NAME.herokuapp.com"
}

# Function to deploy to Streamlit Cloud
deploy_streamlit_cloud() {
    print_status "Deploying to Streamlit Cloud..."
    
    print_status "To deploy to Streamlit Cloud:"
    echo "1. Go to https://share.streamlit.io/"
    echo "2. Connect your GitHub repository"
    echo "3. Set the main file path to: dashboard/streamlit_app.py"
    echo "4. Add environment variables if needed:"
    echo "   - AWS_ACCESS_KEY_ID"
    echo "   - AWS_SECRET_ACCESS_KEY"
    echo "   - AWS_DEFAULT_REGION"
    echo "5. Deploy!"
    
    print_success "Follow the instructions above to deploy to Streamlit Cloud."
}

# Main deployment function
main() {
    echo "🚀 Sentiment Analysis App Deployment Script"
    echo "=============================================="
    
    # Check prerequisites
    check_prerequisites
    
    # Build Docker image
    build_docker_image
    
    # Show deployment options
    echo ""
    echo "Deployment Options:"
    echo "1. Run locally with Docker"
    echo "2. Deploy to AWS ECS"
    echo "3. Deploy to Heroku"
    echo "4. Deploy to Streamlit Cloud (manual)"
    echo "5. All of the above"
    
    read -p "Choose deployment option (1-5): " choice
    
    case $choice in
        1)
            deploy_local_docker
            ;;
        2)
            deploy_aws_ecs
            ;;
        3)
            deploy_heroku
            ;;
        4)
            deploy_streamlit_cloud
            ;;
        5)
            deploy_local_docker
            deploy_aws_ecs
            deploy_heroku
            deploy_streamlit_cloud
            ;;
        *)
            print_error "Invalid option. Please choose 1-5."
            exit 1
            ;;
    esac
    
    print_success "Deployment completed!"
}

# Run main function
main "$@"