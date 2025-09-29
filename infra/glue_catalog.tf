# Glue Catalog configuration for S3 data querying

# Glue Database
resource "aws_glue_catalog_database" "sentiment_insights_db" {
  name = "sentiment_insights"
  
  description = "Database for sentiment-driven product feature insights"
  
  parameters = {
    "description" = "Database containing Amazon review data and sentiment insights"
  }
}

# Glue Table for raw reviews
resource "aws_glue_catalog_table" "amazon_reviews_all_beauty" {
  name          = "amazon_reviews_all_beauty"
  database_name = aws_glue_catalog_database.sentiment_insights_db.name
  description   = "Amazon reviews for All Beauty category"

  table_type = "EXTERNAL_TABLE"

  parameters = {
    "classification" = "json"
    "typeOfData"     = "file"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.raw_bucket.bucket}/raw/All_Beauty/"
    input_format  = "org.apache.hadoop.mapred.TextInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"

    ser_de_info {
      name                  = "json-serde"
      serialization_library = "org.openx.data.jsonserde.JsonSerDe"
      
      parameters = {
        "serialization.format" = "1"
        "case.insensitive"     = "true"
        "mapping.battery"      = "battery_life"
        "mapping.camera"       = "camera_quality"
        "mapping.screen"       = "display_quality"
      }
    }

    columns {
      name = "rating"
      type = "double"
      comment = "Product rating (1-5 stars)"
    }

    columns {
      name = "title"
      type = "string"
      comment = "Review title"
    }

    columns {
      name = "text"
      type = "string"
      comment = "Review text content"
    }

    columns {
      name = "asin"
      type = "string"
      comment = "Amazon Standard Identification Number"
    }

    columns {
      name = "parent_asin"
      type = "string"
      comment = "Parent product ASIN"
    }

    columns {
      name = "timestamp"
      type = "bigint"
      comment = "Review timestamp (Unix epoch in milliseconds)"
    }

    columns {
      name = "user_id"
      type = "string"
      comment = "User identifier (hashed)"
    }

    columns {
      name = "verified_purchase"
      type = "boolean"
      comment = "Whether the purchase was verified"
    }
  }

  partition_keys {
    name = "year"
    type = "string"
    comment = "Year partition (YYYY)"
  }

  partition_keys {
    name = "month"
    type = "string"
    comment = "Month partition (MM)"
  }

  partition_keys {
    name = "day"
    type = "string"
    comment = "Day partition (DD)"
  }
}

# Glue Table for sentiment insights
resource "aws_glue_catalog_table" "product_sentiment_insights" {
  name          = "product_sentiment_insights"
  database_name = aws_glue_catalog_database.sentiment_insights_db.name
  description   = "Aggregated product sentiment insights"

  table_type = "EXTERNAL_TABLE"

  parameters = {
    "classification" = "json"
    "typeOfData"     = "file"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.raw_bucket.bucket}/processed/sentiment/"
    input_format  = "org.apache.hadoop.mapred.TextInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"

    ser_de_info {
      name                  = "json-serde"
      serialization_library = "org.openx.data.jsonserde.JsonSerDe"
      
      parameters = {
        "serialization.format" = "1"
        "case.insensitive"     = "true"
      }
    }

    columns {
      name = "parent_asin"
      type = "string"
      comment = "Parent product ASIN"
    }

    columns {
      name = "feature"
      type = "string"
      comment = "Product feature name"
    }

    columns {
      name = "agg_score_sum"
      type = "double"
      comment = "Sum of sentiment scores"
    }

    columns {
      name = "agg_score_count"
      type = "int"
      comment = "Count of sentiment scores"
    }

    columns {
      name = "positive_snippets"
      type = "array<string>"
      comment = "Positive review snippets"
    }

    columns {
      name = "negative_snippets"
      type = "array<string>"
      comment = "Negative review snippets"
    }

    columns {
      name = "last_updated"
      type = "bigint"
      comment = "Last update timestamp"
    }

    columns {
      name = "category"
      type = "string"
      comment = "Product category"
    }
  }

  partition_keys {
    name = "year"
    type = "string"
    comment = "Year partition (YYYY)"
  }

  partition_keys {
    name = "month"
    type = "string"
    comment = "Month partition (MM)"
  }

  partition_keys {
    name = "day"
    type = "string"
    comment = "Day partition (DD)"
  }
}

# Glue Crawler for raw data
resource "aws_glue_crawler" "raw_data_crawler" {
  database_name = aws_glue_catalog_database.sentiment_insights_db.name
  name          = "sentiment-insights-raw-crawler"
  role          = aws_iam_role.glue_role.arn

  s3_target {
    path = "s3://${aws_s3_bucket.raw_bucket.bucket}/raw/"
  }

  schedule = "cron(0 0 * * ? *)"  # Daily at midnight

  tags = {
    Name        = "Raw Data Crawler"
    Environment = var.environment
    Project     = "SentimentInsights"
  }
}

# Glue Crawler for processed data
resource "aws_glue_crawler" "processed_data_crawler" {
  database_name = aws_glue_catalog_database.sentiment_insights_db.name
  name          = "sentiment-insights-processed-crawler"
  role          = aws_iam_role.glue_role.arn

  s3_target {
    path = "s3://${aws_s3_bucket.raw_bucket.bucket}/processed/"
  }

  schedule = "cron(0 1 * * ? *)"  # Daily at 1 AM

  tags = {
    Name        = "Processed Data Crawler"
    Environment = var.environment
    Project     = "SentimentInsights"
  }
}

# IAM role for Glue
resource "aws_iam_role" "glue_role" {
  name = "${var.project_name}-glue-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "glue.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "Glue Role"
    Environment = var.environment
    Project     = "SentimentInsights"
  }
}

# IAM policy for Glue
resource "aws_iam_role_policy" "glue_policy" {
  name = "${var.project_name}-glue-policy"
  role = aws_iam_role.glue_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "${aws_s3_bucket.raw_bucket.arn}/*",
          "${aws_s3_bucket.raw_bucket.arn}"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "glue:GetTable",
          "glue:GetTables",
          "glue:GetDatabase",
          "glue:GetDatabases",
          "glue:CreateTable",
          "glue:UpdateTable",
          "glue:DeleteTable"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# Attach AWS managed policy for Glue
resource "aws_iam_role_policy_attachment" "glue_service_role" {
  role       = aws_iam_role.glue_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}

# Outputs
output "glue_database_name" {
  description = "Glue database name"
  value       = aws_glue_catalog_database.sentiment_insights_db.name
}

output "glue_table_name" {
  description = "Glue table name for raw reviews"
  value       = aws_glue_catalog_table.amazon_reviews_all_beauty.name
}
