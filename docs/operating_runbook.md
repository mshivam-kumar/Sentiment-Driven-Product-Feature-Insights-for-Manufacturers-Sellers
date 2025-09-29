# Operating Runbook for Sentiment-Driven Product Feature Insights

## Overview

This runbook provides operational guidance for managing the Sentiment-Driven Product Feature Insights system in production environments.

## System Architecture

### Components
1. **Data Ingestion**: S3-based data storage and processing
2. **Aspect Extraction**: spaCy + YAKE-based feature extraction
3. **Sentiment Analysis**: DistilBERT-based sentiment scoring
4. **Aggregation**: DynamoDB-based sentiment aggregation
5. **API**: RESTful API for data access
6. **Dashboard**: Streamlit-based visualization

### Infrastructure
- **AWS S3**: Raw data storage
- **AWS DynamoDB**: Aggregated insights storage
- **AWS Lambda**: Serverless compute
- **AWS API Gateway**: API management
- **AWS SQS**: Message queuing
- **Terraform**: Infrastructure as Code

## Monitoring and Alerting

### Key Metrics
1. **Processing Metrics**:
   - Reviews processed per hour
   - Processing latency (P50, P95, P99)
   - Error rate
   - Queue depth

2. **Quality Metrics**:
   - Aspect extraction precision
   - Sentiment accuracy
   - Model confidence scores
   - Data quality scores

3. **Infrastructure Metrics**:
   - Lambda execution duration
   - DynamoDB read/write capacity
   - S3 storage usage
   - API Gateway latency

### CloudWatch Alarms
```bash
# Lambda errors alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "sentiment-insights-lambda-errors" \
  --alarm-description "Lambda function errors" \
  --metric-name Errors \
  --namespace AWS/Lambda \
  --statistic Sum \
  --period 300 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2

# SQS queue depth alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "sentiment-insights-queue-depth" \
  --alarm-description "SQS queue depth" \
  --metric-name ApproximateNumberOfVisibleMessages \
  --namespace AWS/SQS \
  --statistic Average \
  --period 300 \
  --threshold 100 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2
```

### Log Analysis
```bash
# View Lambda logs
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/sentiment-insights"

# View recent errors
aws logs filter-log-events \
  --log-group-name "/aws/lambda/sentiment-insights-inference" \
  --filter-pattern "ERROR"
```

## Deployment Procedures

### Initial Deployment
1. **Infrastructure Setup**:
   ```bash
   cd infra
   terraform init
   terraform plan
   terraform apply
   ```

2. **Model Deployment**:
   ```bash
   # Package Lambda functions
   cd inference/lambda_inference
   zip -r inference_function.zip .
   
   cd ../../api
   zip -r api_function.zip .
   
   # Deploy to AWS
   aws lambda update-function-code \
     --function-name sentiment-insights-inference \
     --zip-file fileb://inference_function.zip
   ```

3. **Data Ingestion**:
   ```bash
   cd data_ingest
   python download_and_upload.py \
     --dataset_id "McAuley-Lab/Amazon-Reviews-2023" \
     --subset "raw_review_All_Beauty" \
     --s3_bucket "your-bucket-name" \
     --s3_prefix "raw/All_Beauty" \
     --num_samples 1000
   ```

### Model Updates

#### Blue-Green Deployment
1. **Prepare New Model**:
   ```bash
   # Train new model
   cd models/sentiment
   python train_sentiment.py \
     --train_file data/train.jsonl \
     --output_dir s3://your-bucket/models/sentiment/v2
   ```

2. **Deploy to Staging**:
   ```bash
   # Update Lambda function with new model
   aws lambda update-function-code \
     --function-name sentiment-insights-inference-staging \
     --zip-file fileb://inference_function_v2.zip
   ```

3. **Validation**:
   ```bash
   # Run validation tests
   python tests/test_model_validation.py
   ```

4. **Production Deployment**:
   ```bash
   # Switch traffic to new model
   aws lambda update-function-code \
     --function-name sentiment-insights-inference \
     --zip-file fileb://inference_function_v2.zip
   ```

#### Rollback Procedure
1. **Identify Issue**:
   ```bash
   # Check error rates
   aws cloudwatch get-metric-statistics \
     --namespace AWS/Lambda \
     --metric-name Errors \
     --dimensions Name=FunctionName,Value=sentiment-insights-inference \
     --start-time 2024-01-01T00:00:00Z \
     --end-time 2024-01-01T23:59:59Z \
     --period 300 \
     --statistics Sum
   ```

2. **Rollback to Previous Version**:
   ```bash
   # Revert to previous model version
   aws lambda update-function-code \
     --function-name sentiment-insights-inference \
     --zip-file fileb://inference_function_v1.zip
   ```

3. **Verify Rollback**:
   ```bash
   # Monitor error rates
   aws cloudwatch get-metric-statistics \
     --namespace AWS/Lambda \
     --metric-name Errors \
     --dimensions Name=FunctionName,Value=sentiment-insights-inference \
     --start-time 2024-01-01T00:00:00Z \
     --end-time 2024-01-01T23:59:59Z \
     --period 300 \
     --statistics Sum
   ```

## Troubleshooting

### Common Issues

#### 1. High Error Rates
**Symptoms**: CloudWatch alarms triggering, high error rates in Lambda functions

**Diagnosis**:
```bash
# Check Lambda function logs
aws logs filter-log-events \
  --log-group-name "/aws/lambda/sentiment-insights-inference" \
  --filter-pattern "ERROR" \
  --start-time $(date -d '1 hour ago' +%s)000
```

**Resolution**:
- Check model dependencies and versions
- Verify input data format
- Review memory and timeout settings
- Check DynamoDB permissions

#### 2. Processing Delays
**Symptoms**: High queue depth, slow processing times

**Diagnosis**:
```bash
# Check SQS queue metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/SQS \
  --metric-name ApproximateNumberOfVisibleMessages \
  --dimensions Name=QueueName,Value=sentiment-insights-processing \
  --start-time $(date -d '1 hour ago' +%s)000 \
  --end-time $(date +%s)000 \
  --period 300 \
  --statistics Average
```

**Resolution**:
- Scale Lambda concurrency
- Optimize model inference
- Check DynamoDB throttling
- Review batch processing settings

#### 3. Data Quality Issues
**Symptoms**: Low confidence scores, inconsistent results

**Diagnosis**:
```bash
# Check data quality metrics
aws dynamodb scan \
  --table-name product_sentiment_insights \
  --filter-expression "confidence < :threshold" \
  --expression-attribute-values '{":threshold":{"N":"0.5"}}'
```

**Resolution**:
- Review input data quality
- Update aspect extraction rules
- Retrain sentiment model
- Adjust confidence thresholds

### Performance Optimization

#### 1. Lambda Optimization
```bash
# Increase memory allocation
aws lambda update-function-configuration \
  --function-name sentiment-insights-inference \
  --memory-size 2048

# Increase timeout
aws lambda update-function-configuration \
  --function-name sentiment-insights-inference \
  --timeout 300
```

#### 2. DynamoDB Optimization
```bash
# Enable auto-scaling
aws application-autoscaling register-scalable-target \
  --service-namespace dynamodb \
  --resource-id table/product_sentiment_insights \
  --scalable-dimension dynamodb:table:WriteCapacityUnits \
  --min-capacity 5 \
  --max-capacity 100
```

#### 3. S3 Optimization
```bash
# Enable S3 Transfer Acceleration
aws s3api put-bucket-accelerate-configuration \
  --bucket your-bucket-name \
  --accelerate-configuration Status=Enabled
```

## Maintenance Procedures

### Daily Tasks
1. **Monitor System Health**:
   - Check CloudWatch dashboards
   - Review error logs
   - Verify processing metrics

2. **Data Quality Checks**:
   - Validate input data format
   - Check processing success rates
   - Monitor confidence scores

### Weekly Tasks
1. **Performance Review**:
   - Analyze processing times
   - Review error patterns
   - Check resource utilization

2. **Model Quality Assessment**:
   - Sample and validate results
   - Check for model drift
   - Review aspect extraction quality

### Monthly Tasks
1. **Model Updates**:
   - Retrain models with new data
   - Update aspect extraction rules
   - Improve sentiment analysis

2. **Infrastructure Review**:
   - Review costs and optimization
   - Update security configurations
   - Plan capacity changes

## Security Procedures

### Access Management
```bash
# Create IAM user for operations
aws iam create-user --user-name sentiment-insights-ops

# Attach necessary policies
aws iam attach-user-policy \
  --user-name sentiment-insights-ops \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### Data Encryption
```bash
# Enable S3 encryption
aws s3api put-bucket-encryption \
  --bucket your-bucket-name \
  --server-side-encryption-configuration '{
    "Rules": [
      {
        "ApplyServerSideEncryptionByDefault": {
          "SSEAlgorithm": "AES256"
        }
      }
    ]
  }'
```

### Monitoring and Auditing
```bash
# Enable CloudTrail
aws cloudtrail create-trail \
  --name sentiment-insights-trail \
  --s3-bucket-name your-cloudtrail-bucket
```

## Backup and Recovery

### Data Backup
```bash
# Backup DynamoDB table
aws dynamodb create-backup \
  --table-name product_sentiment_insights \
  --backup-name sentiment-insights-backup-$(date +%Y%m%d)
```

### Disaster Recovery
1. **Infrastructure Recovery**:
   ```bash
   cd infra
   terraform apply
   ```

2. **Data Recovery**:
   ```bash
   # Restore from S3 backup
   aws s3 sync s3://backup-bucket/sentiment-insights/ s3://your-bucket/
   ```

3. **Model Recovery**:
   ```bash
   # Download and deploy models
   aws s3 sync s3://your-bucket/models/ ./models/
   ```

## Contact Information

### On-Call Rotation
- **Primary**: [Primary On-Call Contact]
- **Secondary**: [Secondary On-Call Contact]
- **Escalation**: [Escalation Contact]

### Emergency Procedures
1. **Critical Issues**: Contact on-call engineer immediately
2. **Service Outages**: Follow incident response procedures
3. **Data Breaches**: Follow security incident procedures

### Documentation Updates
- Update this runbook monthly
- Document new procedures and issues
- Review and update contact information
- Validate all procedures through testing
