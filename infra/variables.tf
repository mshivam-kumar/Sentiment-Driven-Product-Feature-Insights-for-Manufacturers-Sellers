variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "dynamodb_table_name" {
  description = "DynamoDB table name"
  type        = string
  default     = "product_sentiment_insights"
}

# Terraform variables for Sentiment-Driven Product Feature Insights

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "sentiment-insights"
}

variable "raw_bucket_name" {
  description = "Name of the S3 bucket for raw data"
  type        = string
  default     = "sentiment-insights-raw-data"
}

variable "models_bucket_name" {
  description = "Name of the S3 bucket for model artifacts"
  type        = string
  default     = "sentiment-insights-models"
}

variable "dynamodb_table_name" {
  description = "Name of the DynamoDB table for sentiment insights"
  type        = string
  default     = "product_sentiment_insights"
}

variable "sqs_queue_name" {
  description = "Name of the SQS queue for processing"
  type        = string
  default     = "sentiment-insights-processing"
}

variable "lambda_memory_size" {
  description = "Memory size for Lambda functions"
  type        = number
  default     = 1024
}

variable "lambda_timeout" {
  description = "Timeout for Lambda functions in seconds"
  type        = number
  default     = 300
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 14
}

variable "tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project     = "SentimentInsights"
    Environment = "dev"
    ManagedBy   = "Terraform"
  }
}
