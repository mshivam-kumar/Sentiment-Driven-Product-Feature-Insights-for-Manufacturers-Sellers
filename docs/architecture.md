# System Architecture

## Overview

The Sentiment-Driven Product Feature Insights system is designed as a scalable, cloud-native ML pipeline that processes Amazon reviews to extract product features and analyze sentiment.

## Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │    │   Data Ingestion│    │   Raw Storage   │
│                 │    │                 │    │                 │
│ Hugging Face    │───▶│ download_and_   │───▶│   S3 Bucket     │
│ Amazon Reviews  │    │ upload.py       │    │   (JSONL)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Processing    │    │   Aspect         │    │   Sentiment     │
│   Pipeline      │    │   Extraction     │    │   Analysis      │
│                 │    │                 │    │                 │
│ S3 Event ──────▶│    │ spaCy + YAKE    │    │ DistilBERT      │
│ SQS Queue       │    │ Noun Phrases    │    │ Pre-trained     │
│ Lambda          │    │ Keywords        │    │ Sentiment       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Aggregation   │    │   Storage        │    │   API Layer     │
│                 │    │                 │    │                 │
│ DynamoDB        │◀───│ DynamoDB        │    │ API Gateway     │
│ Atomic Updates  │    │ Aggregated      │    │ Lambda Handler  │
│ Snippets        │    │ Insights        │    │ REST Endpoints  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Visualization │    │   Monitoring    │    │   CI/CD         │
│                 │    │                 │    │                 │
│ Streamlit       │    │ CloudWatch      │    │ GitHub Actions  │
│ Dashboard       │    │ Alarms          │    │ Terraform       │
│ Charts & Tables │    │ Logs            │    │ Automated Deploy │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Component Details

### 1. Data Ingestion Layer
- **Component**: `data_ingest/download_and_upload.py`
- **Purpose**: Download Amazon reviews from Hugging Face and upload to S3
- **Input**: McAuley-Lab/Amazon-Reviews-2023 dataset
- **Output**: JSONL files in S3
- **Format**: `s3://bucket/raw/category/YYYYMMDD/file.jsonl`

### 2. Processing Pipeline
- **Trigger**: S3 event notifications → SQS
- **Processor**: Lambda function with ML models
- **Models**: 
  - Aspect Extraction: spaCy + YAKE
  - Sentiment Analysis: DistilBERT
- **Output**: Aspect-sentiment pairs

### 3. Storage Layer
- **Raw Data**: S3 (JSONL format)
- **Processed Data**: DynamoDB (aggregated insights)
- **Schema**: 
  - PK: parent_asin
  - SK: feature
  - Attributes: sentiment scores, snippets, timestamps

### 4. API Layer
- **Gateway**: AWS API Gateway
- **Handler**: Lambda function
- **Endpoints**:
  - `GET /sentiment/product/{asin}`: Product insights
  - `GET /sentiment/product/{asin}/top-features`: Top features
  - `GET /sentiment/search`: Feature search
  - `GET /health`: Health check

### 5. Visualization Layer
- **Dashboard**: Streamlit application
- **Features**: Interactive charts, product analysis, trend visualization
- **Data Source**: API endpoints

### 6. Infrastructure
- **IaC**: Terraform
- **Compute**: AWS Lambda (serverless)
- **Storage**: S3 + DynamoDB
- **Monitoring**: CloudWatch
- **CI/CD**: GitHub Actions

## Data Flow

### 1. Ingestion Flow
```
Hugging Face Dataset → Python Script → S3 Bucket
```

### 2. Processing Flow
```
S3 Event → SQS → Lambda → ML Models → DynamoDB
```

### 3. Query Flow
```
User Request → API Gateway → Lambda → DynamoDB → Response
```

### 4. Visualization Flow
```
Dashboard → API Gateway → Lambda → DynamoDB → Charts
```

## Scalability Considerations

### Horizontal Scaling
- **Lambda**: Auto-scaling based on SQS queue depth
- **DynamoDB**: On-demand billing mode
- **API Gateway**: Built-in scaling

### Performance Optimization
- **Batch Processing**: Process multiple reviews per Lambda invocation
- **Caching**: API Gateway caching for frequently accessed data
- **Partitioning**: S3 partitioning by date and category

### Cost Optimization
- **Serverless**: Pay-per-use model
- **Storage Classes**: S3 Intelligent Tiering
- **Reserved Capacity**: DynamoDB reserved capacity for predictable workloads

## Security

### Data Protection
- **Encryption**: S3 server-side encryption, DynamoDB encryption at rest
- **Access Control**: IAM roles and policies
- **Network**: VPC endpoints for private communication

### Compliance
- **Data Privacy**: No PII storage
- **Audit**: CloudTrail logging
- **Monitoring**: CloudWatch alarms

## Monitoring and Observability

### Metrics
- **Processing**: Reviews per hour, latency, error rate
- **Quality**: Aspect precision, sentiment accuracy
- **Infrastructure**: Lambda duration, DynamoDB capacity

### Alerts
- **Errors**: Lambda error rate > 5%
- **Performance**: Queue depth > 100 messages
- **Quality**: Model confidence < 0.5

### Logging
- **Application**: CloudWatch Logs
- **Access**: CloudTrail
- **Performance**: X-Ray tracing

## Deployment

### Environments
- **Development**: Local development with mocked services
- **Staging**: Full AWS deployment with test data
- **Production**: Production AWS deployment

### Deployment Process
1. **Code**: GitHub Actions CI/CD
2. **Infrastructure**: Terraform apply
3. **Models**: S3 model artifacts
4. **API**: Lambda function updates
5. **Dashboard**: Streamlit deployment

### Rollback Strategy
- **Infrastructure**: Terraform state rollback
- **Models**: Blue-green deployment
- **API**: Lambda version rollback
- **Data**: S3 versioning

## Performance Characteristics

### Throughput
- **Ingestion**: 1000+ reviews per hour
- **Processing**: 30+ reviews per minute
- **API**: 100+ requests per second

### Latency
- **Processing**: < 2 seconds per review
- **API**: < 500ms per request
- **Dashboard**: < 1 second page load

### Accuracy
- **Aspect Extraction**: 65% F1 score (baseline)
- **Sentiment Analysis**: 82% accuracy
- **Confidence**: > 0.5 for high-quality insights

## Future Enhancements

### Model Improvements
- **ABSA Training**: Custom aspect-based sentiment analysis
- **Multi-language**: Support for multiple languages
- **Real-time**: Stream processing capabilities

### Feature Additions
- **Trend Analysis**: Time-series sentiment trends
- **Competitive Analysis**: Cross-product comparisons
- **Recommendations**: Product improvement suggestions

### Infrastructure Upgrades
- **SageMaker**: Managed model serving
- **Elasticsearch**: Advanced search capabilities
- **Kubernetes**: Container orchestration
