# Minimal learning scaffold for core resources

resource "aws_s3_bucket" "raw" {
  bucket = "skchatbotbucket"
}

resource "aws_sqs_queue" "reviews" {
  name = "sentiment-insights-processing"
}

resource "aws_dynamodb_table" "sentiment_insights" {
  name         = var.dynamodb_table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "parent_asin"
  attribute { name = "parent_asin" type = "S" }
}

data "aws_iam_policy_document" "lambda_assume" {
  statement { actions = ["sts:AssumeRole"], principals { type = "Service" identifiers = ["lambda.amazonaws.com"] } }
}

resource "aws_iam_role" "lambda_role" {
  name               = "sentiment-insights-lambda-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "sentiment-insights-lambda-inline"
  role = aws_iam_role.lambda_role.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      { Effect = "Allow", Action = ["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"], Resource = "*" },
      { Effect = "Allow", Action = ["dynamodb:PutItem","dynamodb:UpdateItem","dynamodb:GetItem","dynamodb:Query"], Resource = aws_dynamodb_table.sentiment_insights.arn },
      { Effect = "Allow", Action = ["sqs:ReceiveMessage","sqs:DeleteMessage","sqs:GetQueueAttributes"], Resource = aws_sqs_queue.reviews.arn },
      { Effect = "Allow", Action = ["s3:GetObject"], Resource = "${aws_s3_bucket.raw.arn}/*" }
    ]
  })
}

output "dynamodb_table" { value = aws_dynamodb_table.sentiment_insights.name }

