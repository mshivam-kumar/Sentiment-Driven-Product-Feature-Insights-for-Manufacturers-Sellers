# Terraform configuration for Sentiment-Driven Product Feature Insights

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# S3 Bucket for raw data storage
resource "aws_s3_bucket" "raw_bucket" {
  bucket = var.raw_bucket_name
  
  tags = {
    Name        = "Raw Data Bucket"
    Environment = var.environment
    Project     = "SentimentInsights"
  }
}

resource "aws_s3_bucket_versioning" "raw_bucket_versioning" {
  bucket = aws_s3_bucket.raw_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "raw_bucket_encryption" {
  bucket = aws_s3_bucket.raw_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# S3 Bucket for model artifacts
resource "aws_s3_bucket" "models_bucket" {
  bucket = var.models_bucket_name
  
  tags = {
    Name        = "Models Bucket"
    Environment = var.environment
    Project     = "SentimentInsights"
  }
}

resource "aws_s3_bucket_versioning" "models_bucket_versioning" {
  bucket = aws_s3_bucket.models_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# DynamoDB table for sentiment insights
resource "aws_dynamodb_table" "sentiment_insights" {
  name           = var.dynamodb_table_name
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "parent_asin"
  range_key      = "feature"

  attribute {
    name = "parent_asin"
    type = "S"
  }

  attribute {
    name = "feature"
    type = "S"
  }

  tags = {
    Name        = "Sentiment Insights"
    Environment = var.environment
    Project     = "SentimentInsights"
  }
}

# SQS Queue for processing pipeline
resource "aws_sqs_queue" "processing_queue" {
  name                      = var.sqs_queue_name
  delay_seconds             = 0
  max_message_size          = 262144
  message_retention_seconds = 1209600
  receive_wait_time_seconds = 0
  visibility_timeout_seconds = 300

  tags = {
    Name        = "Processing Queue"
    Environment = var.environment
    Project     = "SentimentInsights"
  }
}

# Dead letter queue
resource "aws_sqs_queue" "processing_dlq" {
  name = "${var.sqs_queue_name}-dlq"

  tags = {
    Name        = "Processing DLQ"
    Environment = var.environment
    Project     = "SentimentInsights"
  }
}

resource "aws_sqs_queue_redrive_policy" "processing_queue_redrive" {
  queue_url = aws_sqs_queue.processing_queue.id
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.processing_dlq.arn
    maxReceiveCount     = 3
  })
}

# IAM role for Lambda functions
resource "aws_iam_role" "lambda_execution_role" {
  name = "${var.project_name}-lambda-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "Lambda Execution Role"
    Environment = var.environment
    Project     = "SentimentInsights"
  }
}

# IAM policy for Lambda functions
resource "aws_iam_role_policy" "lambda_policy" {
  name = "${var.project_name}-lambda-policy"
  role = aws_iam_role.lambda_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.raw_bucket.arn}/*",
          "${aws_s3_bucket.models_bucket.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:DeleteItem",
          "dynamodb:Query",
          "dynamodb:Scan"
        ]
        Resource = aws_dynamodb_table.sentiment_insights.arn
      },
      {
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes"
        ]
        Resource = aws_sqs_queue.processing_queue.arn
      }
    ]
  })
}

# Lambda function for inference
resource "aws_lambda_function" "inference_function" {
  filename         = "inference_function.zip"
  function_name    = "${var.project_name}-inference"
  role            = aws_iam_role.lambda_execution_role.arn
  handler         = "handler.lambda_handler"
  runtime         = "python3.10"
  timeout         = 300
  memory_size     = 1024

  environment {
    variables = {
      DYNAMODB_TABLE = aws_dynamodb_table.sentiment_insights.name
      S3_BUCKET      = aws_s3_bucket.raw_bucket.bucket
    }
  }

  depends_on = [
    aws_iam_role_policy.lambda_policy,
    aws_cloudwatch_log_group.inference_logs
  ]

  tags = {
    Name        = "Inference Function"
    Environment = var.environment
    Project     = "SentimentInsights"
  }
}

# Lambda function for API
resource "aws_lambda_function" "api_function" {
  filename         = "api_function.zip"
  function_name    = "${var.project_name}-api"
  role            = aws_iam_role.lambda_execution_role.arn
  handler         = "lambda_api_handler.lambda_handler"
  runtime         = "python3.10"
  timeout         = 30
  memory_size     = 512

  environment {
    variables = {
      DYNAMODB_TABLE = aws_dynamodb_table.sentiment_insights.name
    }
  }

  depends_on = [
    aws_iam_role_policy.lambda_policy,
    aws_cloudwatch_log_group.api_logs
  ]

  tags = {
    Name        = "API Function"
    Environment = var.environment
    Project     = "SentimentInsights"
  }
}

# CloudWatch log groups
resource "aws_cloudwatch_log_group" "inference_logs" {
  name              = "/aws/lambda/${var.project_name}-inference"
  retention_in_days = 14

  tags = {
    Name        = "Inference Logs"
    Environment = var.environment
    Project     = "SentimentInsights"
  }
}

resource "aws_cloudwatch_log_group" "api_logs" {
  name              = "/aws/lambda/${var.project_name}-api"
  retention_in_days = 14

  tags = {
    Name        = "API Logs"
    Environment = var.environment
    Project     = "SentimentInsights"
  }
}

# API Gateway
resource "aws_api_gateway_rest_api" "sentiment_api" {
  name        = "${var.project_name}-api"
  description = "API for Sentiment-Driven Product Feature Insights"

  endpoint_configuration {
    types = ["REGIONAL"]
  }

  tags = {
    Name        = "Sentiment API"
    Environment = var.environment
    Project     = "SentimentInsights"
  }
}

# API Gateway resources
resource "aws_api_gateway_resource" "sentiment_resource" {
  rest_api_id = aws_api_gateway_rest_api.sentiment_api.id
  parent_id   = aws_api_gateway_rest_api.sentiment_api.root_resource_id
  path_part   = "sentiment"
}

resource "aws_api_gateway_resource" "product_resource" {
  rest_api_id = aws_api_gateway_rest_api.sentiment_api.id
  parent_id   = aws_api_gateway_resource.sentiment_resource.id
  path_part   = "product"
}

resource "aws_api_gateway_resource" "asin_resource" {
  rest_api_id = aws_api_gateway_rest_api.sentiment_api.id
  parent_id   = aws_api_gateway_resource.product_resource.id
  path_part   = "{asin}"
}

resource "aws_api_gateway_resource" "top_features_resource" {
  rest_api_id = aws_api_gateway_rest_api.sentiment_api.id
  parent_id   = aws_api_gateway_resource.asin_resource.id
  path_part   = "top-features"
}

resource "aws_api_gateway_resource" "search_resource" {
  rest_api_id = aws_api_gateway_rest_api.sentiment_api.id
  parent_id   = aws_api_gateway_resource.sentiment_resource.id
  path_part   = "search"
}

resource "aws_api_gateway_resource" "health_resource" {
  rest_api_id = aws_api_gateway_rest_api.sentiment_api.id
  parent_id   = aws_api_gateway_rest_api.sentiment_api.root_resource_id
  path_part   = "health"
}

# API Gateway methods
resource "aws_api_gateway_method" "product_get" {
  rest_api_id   = aws_api_gateway_rest_api.sentiment_api.id
  resource_id   = aws_api_gateway_resource.asin_resource.id
  http_method   = "GET"
  authorization = "NONE"
}

resource "aws_api_gateway_method" "top_features_get" {
  rest_api_id   = aws_api_gateway_rest_api.sentiment_api.id
  resource_id   = aws_api_gateway_resource.top_features_resource.id
  http_method   = "GET"
  authorization = "NONE"
}

resource "aws_api_gateway_method" "search_get" {
  rest_api_id   = aws_api_gateway_rest_api.sentiment_api.id
  resource_id   = aws_api_gateway_resource.search_resource.id
  http_method   = "GET"
  authorization = "NONE"
}

resource "aws_api_gateway_method" "health_get" {
  rest_api_id   = aws_api_gateway_rest_api.sentiment_api.id
  resource_id   = aws_api_gateway_resource.health_resource.id
  http_method   = "GET"
  authorization = "NONE"
}

# Lambda permissions
resource "aws_lambda_permission" "api_gateway_invocation" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api_function.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.sentiment_api.execution_arn}/*/*"
}

# API Gateway integrations
resource "aws_api_gateway_integration" "product_integration" {
  rest_api_id = aws_api_gateway_rest_api.sentiment_api.id
  resource_id = aws_api_gateway_resource.asin_resource.id
  http_method = aws_api_gateway_method.product_get.http_method

  integration_http_method = "POST"
  type                   = "AWS_PROXY"
  uri                    = aws_lambda_function.api_function.invoke_arn
}

resource "aws_api_gateway_integration" "top_features_integration" {
  rest_api_id = aws_api_gateway_rest_api.sentiment_api.id
  resource_id = aws_api_gateway_resource.top_features_resource.id
  http_method = aws_api_gateway_method.top_features_get.http_method

  integration_http_method = "POST"
  type                   = "AWS_PROXY"
  uri                    = aws_lambda_function.api_function.invoke_arn
}

resource "aws_api_gateway_integration" "search_integration" {
  rest_api_id = aws_api_gateway_rest_api.sentiment_api.id
  resource_id = aws_api_gateway_resource.search_resource.id
  http_method = aws_api_gateway_method.search_get.http_method

  integration_http_method = "POST"
  type                   = "AWS_PROXY"
  uri                    = aws_lambda_function.api_function.invoke_arn
}

resource "aws_api_gateway_integration" "health_integration" {
  rest_api_id = aws_api_gateway_rest_api.sentiment_api.id
  resource_id = aws_api_gateway_resource.health_resource.id
  http_method = aws_api_gateway_method.health_get.http_method

  integration_http_method = "POST"
  type                   = "AWS_PROXY"
  uri                    = aws_lambda_function.api_function.invoke_arn
}

# API Gateway deployment
resource "aws_api_gateway_deployment" "sentiment_api_deployment" {
  depends_on = [
    aws_api_gateway_integration.product_integration,
    aws_api_gateway_integration.top_features_integration,
    aws_api_gateway_integration.search_integration,
    aws_api_gateway_integration.health_integration
  ]

  rest_api_id = aws_api_gateway_rest_api.sentiment_api.id
  stage_name  = var.environment

  lifecycle {
    create_before_destroy = true
  }
}

# S3 event notification for processing pipeline
resource "aws_s3_bucket_notification" "raw_bucket_notification" {
  bucket = aws_s3_bucket.raw_bucket.id

  queue {
    queue_arn = aws_sqs_queue.processing_queue.arn
    events    = ["s3:ObjectCreated:*"]
    filter_prefix = "raw/"
  }

  depends_on = [aws_sqs_queue_policy.processing_queue_policy]
}

# SQS queue policy for S3 notifications
resource "aws_sqs_queue_policy" "processing_queue_policy" {
  queue_url = aws_sqs_queue.processing_queue.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
        Action = "sqs:SendMessage"
        Resource = aws_sqs_queue.processing_queue.arn
        Condition = {
          ArnEquals = {
            "aws:SourceArn" = aws_s3_bucket.raw_bucket.arn
          }
        }
      }
    ]
  })
}

# CloudWatch alarms
resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "${var.project_name}-lambda-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = "300"
  statistic           = "Sum"
  threshold           = "5"
  alarm_description   = "This metric monitors lambda errors"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    FunctionName = aws_lambda_function.inference_function.function_name
  }
}

resource "aws_cloudwatch_metric_alarm" "sqs_queue_depth" {
  alarm_name          = "${var.project_name}-sqs-queue-depth"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "ApproximateNumberOfVisibleMessages"
  namespace           = "AWS/SQS"
  period              = "300"
  statistic           = "Average"
  threshold           = "100"
  alarm_description   = "This metric monitors SQS queue depth"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    QueueName = aws_sqs_queue.processing_queue.name
  }
}

# SNS topic for alerts
resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-alerts"

  tags = {
    Name        = "Alerts Topic"
    Environment = var.environment
    Project     = "SentimentInsights"
  }
}

# Outputs
output "api_gateway_url" {
  description = "API Gateway URL"
  value       = "https://${aws_api_gateway_rest_api.sentiment_api.id}.execute-api.${var.aws_region}.amazonaws.com/${var.environment}"
}

output "s3_raw_bucket" {
  description = "S3 bucket for raw data"
  value       = aws_s3_bucket.raw_bucket.bucket
}

output "dynamodb_table" {
  description = "DynamoDB table name"
  value       = aws_dynamodb_table.sentiment_insights.name
}

output "sqs_queue_url" {
  description = "SQS queue URL"
  value       = aws_sqs_queue.processing_queue.url
}
