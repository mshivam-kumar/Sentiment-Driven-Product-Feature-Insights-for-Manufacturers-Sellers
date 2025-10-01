## Sentiment-Driven Product Feature Insights — Project Learning & Implementation Notes

### Overview
This document captures the end-to-end implementation, design choices, problems faced, resolutions, and operational guidance for the Sentiment-Driven Product Feature Insights platform (SellerIQ). It consolidates learnings from the implementation prompt, code, and deployment work.

### High-level Goals
- Extract product features from Amazon reviews and compute aspect-level sentiment.
- Aggregate results per product (ASIN) and serve via a public API and Streamlit dashboard.
- Make the system cloud-native, reproducible (IaC), observable, and easy to demo.

### Architecture
- Ingestion: Hugging Face dataset → S3 (JSONL)
- Processing: S3 → SQS → Lambda (spaCy aspects + HF SST-2 sentiment) → DynamoDB aggregates
- Serving: API Gateway → Lambda
- Visualization: Streamlit dashboard
- Infrastructure: Terraform (+ GitHub Actions for CI/CD)

### Key Components and How We Built Them
1) Data Ingestion to S3
- Implemented `data_ingest/download_and_upload.py` to pull from `McAuley-Lab/Amazon-Reviews-2023` and write JSONL to S3 with date-based prefixes. Validated local sample and S3 listing.

2) Aspect Extraction (Baseline)
- Used spaCy noun-chunk extraction as a fast, transparent baseline. Example:
```python
import spacy
nlp = spacy.load("en_core_web_sm")
def extract_aspects(text: str):
    doc = nlp(text)
    return list({chunk.text.strip().lower() for chunk in doc.noun_chunks})
```

3) Sentiment Mapping
- Used Hugging Face `distilbert-base-uncased-finetuned-sst-2-english` to classify sentences and map to [-1,+1]. Sentiment per sentence was assigned to aspects present in that sentence.

4) Aggregation in DynamoDB
- Wrote per-`parent_asin` per-`feature` aggregates using atomic counters: `agg_score_sum`, `agg_score_count`, plus `last_updated` and snippets.
- Idempotency used `(user_id, timestamp)` to avoid double counting.

5) Event-driven Processing
- Orchestrated ingestion → inference → aggregation via S3 events → SQS → Lambda. Added CloudWatch alarms for Lambda error rates and SQS queue depth.

6) Public API
- API Gateway + Lambda endpoints:
  - `GET /sentiment/product/{asin}?feature=...&window=30d|90d|1y|10000d`
  - `GET /sentiment/product/{asin}/top-features?limit=10`
  - `GET /sentiment/search?query=...`
  - Returns aggregated sentiment, counts, trends, and snippets.

7) Streamlit Dashboard
- Product Analysis (ASIN, feature filter, time window) and Feature Search.
- RAG Chat (optional): semantic search over review texts with Sentence-Transformers and a chat UI, grounded by retrieved snippets.

8) RAG Enhancements
- Embedding model: `sentence-transformers/all-mpnet-base-v2` with normalized embeddings.
- Configurable context source via env/Secrets: `RAG_REVIEWS_SOURCE` (S3/HTTPS/JSONL) and `RAG_REVIEWS_MAX`.
- Local cached embeddings for fast reloads: `.rag_cache/<hash>/embeddings.npy` and `reviews_meta.json`.

9) Seeding Aggregates
- Script `scripts/seed_from_expanded.py` streams expanded JSONL and invokes inference Lambda to create aggregates for many ASINs. Useful to warm up the API responses and demo.

### AWS Integration & Access
- Auth: AWS CLI configured locally; on Streamlit Cloud we used Secrets and public/presigned S3 URLs since IAM roles aren’t attached by default.
- Data: Raw reviews were written to `s3://<bucket>/raw/<category>/...`.
- Compute: Lambda for inference and API; SQS for buffering.
- Storage: DynamoDB table `product_sentiment_insights`.
- API Gateway: REST API (v1). Base: `https://f3157r5ca4.execute-api.us-east-1.amazonaws.com/dev`.
- Observability: CloudWatch logs, metrics, alarms.

#### Do it yourself next time (step-by-step with code)

1) Minimal Terraform (S3, SQS, DynamoDB, IAM, Lambda, API)
```hcl
# variables.tf
variable "aws_region" { default = "us-east-1" }
variable "dynamodb_table_name" { default = "product_sentiment_insights" }

# provider.tf
provider "aws" { region = var.aws_region }

# s3.tf (raw bucket)
resource "aws_s3_bucket" "raw" { bucket = "skchatbotbucket" }

# sqs.tf
resource "aws_sqs_queue" "reviews" { name = "sentiment-insights-processing" }

# dynamodb.tf
resource "aws_dynamodb_table" "sentiment_insights" {
  name         = var.dynamodb_table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "parent_asin"
  attribute { name = "parent_asin" type = "S" }
}

# iam.tf (lambda role)
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

# lambda_api.tf (zip already uploaded or use filename)
resource "aws_lambda_function" "api" {
  function_name = "sentiment-insights-api"
  role          = aws_iam_role.lambda_role.arn
  handler       = "lambda_api_handler.lambda_handler"
  runtime       = "python3.10"
  filename      = "../api_function.zip"
  timeout       = 15
  environment { variables = { DYNAMODB_TABLE = aws_dynamodb_table.sentiment_insights.name } }
}

# apigw.tf (REST v1)
resource "aws_api_gateway_rest_api" "api" { name = "sentiment-insights-api" }
resource "aws_api_gateway_resource" "sentiment" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  parent_id   = aws_api_gateway_rest_api.api.root_resource_id
  path_part   = "sentiment"
}
resource "aws_api_gateway_resource" "product" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  parent_id   = aws_api_gateway_resource.sentiment.id
  path_part   = "product"
}
resource "aws_api_gateway_resource" "asin" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  parent_id   = aws_api_gateway_resource.product.id
  path_part   = "{asin}"
}
resource "aws_api_gateway_method" "get_product" {
  rest_api_id   = aws_api_gateway_rest_api.api.id
  resource_id   = aws_api_gateway_resource.asin.id
  http_method   = "GET"
  authorization = "NONE"
}
resource "aws_api_gateway_integration" "lambda_proxy" {
  rest_api_id             = aws_api_gateway_rest_api.api.id
  resource_id             = aws_api_gateway_resource.asin.id
  http_method             = aws_api_gateway_method.get_product.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.api.invoke_arn
}
resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.api.execution_arn}/*/*"
}
resource "aws_api_gateway_deployment" "deploy" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  triggers    = { redeploy = timestamp() }
  depends_on  = [aws_api_gateway_integration.lambda_proxy]
}
resource "aws_api_gateway_stage" "dev" {
  rest_api_id   = aws_api_gateway_rest_api.api.id
  deployment_id = aws_api_gateway_deployment.deploy.id
  stage_name    = "dev"
}

output "api_base_url" { value = aws_api_gateway_stage.dev.invoke_url }
output "dynamodb_table" { value = aws_dynamodb_table.sentiment_insights.name }
```

2) Lambda (API) skeleton with DynamoDB query
```python
import os, json, boto3
from datetime import datetime, timedelta

TABLE = os.environ.get("DYNAMODB_TABLE", "product_sentiment_insights")
ddb = boto3.resource("dynamodb").Table(TABLE)

def lambda_handler(event, context):
    asin = event.get("pathParameters", {}).get("asin")
    params = event.get("queryStringParameters") or {}
    window = params.get("window", "30d")
    # normalize window
    if window == "All Time":
        window = "10000d"

    # example: fetch feature aggregates by partition key
    # (full impl may use GSI/time filtering if designed)
    resp = ddb.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key("parent_asin").eq(asin),
        Limit=200
    )
    items = resp.get("Items", [])
    # rollup into response shape
    result = {
        "asin": asin,
        "features": {},
        "overall_sentiment": 0.0,
        "total_reviews": 0,
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "category": "All_Beauty"
    }
    for it in items:
        feature = it.get("feature", "unknown")
        ssum = float(it.get("agg_score_sum", 0))
        cnt = int(it.get("agg_score_count", 0))
        result["features"][feature] = {
            "score": (ssum / cnt) if cnt else 0.0,
            "count": cnt,
            "positive_snippets": it.get("positive_snippets", []),
            "negative_snippets": it.get("negative_snippets", []),
            "trend": it.get("trend", "stable"),
        }
        result["total_reviews"] += cnt
    body = json.dumps(result, default=str)
    return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": body}
```

3) CloudWatch alarms (CLI)
```bash
# Lambda errors alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "sentiment-insights-lambda-errors" \
  --namespace AWS/Lambda --metric-name Errors \
  --dimensions Name=FunctionName,Value=sentiment-insights-api \
  --statistic Sum --period 300 --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold --evaluation-periods 1

# SQS queue depth alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "sentiment-insights-queue-depth" \
  --namespace AWS/SQS --metric-name ApproximateNumberOfVisibleMessages \
  --dimensions Name=QueueName,Value=sentiment-insights-processing \
  --statistic Average --period 300 --threshold 100 \
  --comparison-operator GreaterThanThreshold --evaluation-periods 2
```

4) Find your API base URL
```bash
# REST (v1) API id and stage
aws apigateway get-rest-apis --query "items[].{Name:name,Id:id}" --output table
aws apigateway get-stages --rest-api-id <rest_api_id> --query "item[].stageName" --output text
# Construct: https://<rest_api_id>.execute-api.<region>.amazonaws.com/<stage>
```

5) Presign S3 for Streamlit Cloud RAG
```bash
aws s3 presign s3://skchatbotbucket/raw/All_Beauty/raw_review_All_Beauty_expanded.jsonl --expires-in 86400
```

6) Streamlit Secrets (TOML)
```toml
API_BASE_URL = "https://f3157r5ca4.execute-api.us-east-1.amazonaws.com/dev"
RAG_REVIEWS_SOURCE = "https://<presigned-url>"
RAG_REVIEWS_MAX = "500"
```

7) Test the API quickly
```bash
curl "$API_BASE_URL/sentiment/product/B00YQ6X8EO?window=10000d"
```

### Common Problems & Resolutions
1) 404 on Product Analysis for valid ASIN
- Cause: No aggregates for that ASIN/window in DynamoDB.
- Fix: Seed the ASIN by invoking the inference Lambda for a set of reviews. Then query with `window=10000d` or “All Time”.

2) RAG shows only 10 reviews on Streamlit Cloud
- Cause: Streamlit Cloud cannot read private S3; loader falls back to a tiny sample.
- Fixes:
  - Set `RAG_REVIEWS_SOURCE` to a presigned HTTPS URL (Secrets) and rerun.
  - Or make the S3 object public.
  - Added cache keying by source and forced rebuild when cache is tiny with a real source present.

3) API base URL confusion
- Cause: Mixing API Gateway v1/v2. Our API is REST (v1).
- Fix: Resolve REST id `f3157r5ca4`, get stage `dev`, construct: `https://f3157r5ca4.execute-api.us-east-1.amazonaws.com/dev`.

4) Decimal serialization in Lambda and DynamoDB
- Ensured proper numeric serialization and consistent ISO timestamps to avoid runtime errors.

5) Time-window defaults filtering out data
- Mapped “All Time” to `10000d` in the dashboard to ensure records spanning years are included.

### Representative Code Snippets
1) Dashboard product fetch (window mapping and error handling)
```python
def fetch_product_sentiment(self, asin: str, feature: str = None, window: str = "30d"):
    url = f"{self.api_base_url}/sentiment/product/{asin}"
    params = {}
    if feature:
        params['feature'] = feature
    if window == "All Time" or window is None:
        params['window'] = '10000d'
    else:
        params['window'] = window
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()
```

2) RAG embeddings and retrieval
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
emb = model.encode(texts, normalize_embeddings=True)
q = model.encode([query], normalize_embeddings=True)
sims = cosine_similarity(q, emb)[0]
top_idx = np.argsort(sims)[::-1][:10]
```

3) Seed script (invoke Lambda per review)
```python
resp = client.invoke(
  FunctionName=function_name,
  InvocationType="RequestResponse",
  Payload=json.dumps(payload).encode("utf-8"),
)
```

### How to Run (Local)
1) Ingestion
```bash
cd data_ingest && pip install -r requirements.txt
python download_and_upload.py --dataset_id McAuley-Lab/Amazon-Reviews-2023 \
  --subset raw_review_All_Beauty --s3_bucket skchatbotbucket --s3_prefix raw/All_Beauty --num_samples 1000
```

2) Dashboard (with RAG locally)
```bash
cd dashboard && pip install -r requirements.txt
export API_BASE_URL="https://f3157r5ca4.execute-api.us-east-1.amazonaws.com/dev"
export RAG_REVIEWS_SOURCE="s3://skchatbotbucket/raw/All_Beauty/raw_review_All_Beauty_expanded.jsonl"
export RAG_REVIEWS_MAX=500
streamlit run streamlit_app.py
```

3) Seeding (warm up aggregates)
```bash
python scripts/seed_from_expanded.py \
  --source s3://skchatbotbucket/raw/All_Beauty/raw_review_All_Beauty_expanded.jsonl \
  --function-name sentiment-insights-inference \
  --region us-east-1 \
  --per-asin-limit 50
```

### How to Run (Streamlit Cloud)
- Set Secrets (TOML):
```
API_BASE_URL = "https://f3157r5ca4.execute-api.us-east-1.amazonaws.com/dev"
RAG_REVIEWS_SOURCE = "https://<presigned-s3-jsonl-url>"
RAG_REVIEWS_MAX = "500"
```
- Rerun app. Use the RAG Debug expander to confirm `branch=expanded_source` and `loaded_reviews≈500`.

### Testing, CI/CD, and Observability
- Unit tests for extractors and sentiment mapping; integration tests for end-to-end flow.
- GitHub Actions workflow: install, lint, test; optional deploy steps.
- CloudWatch: Lambda error alarms, SQS queue depth alarms; logs reviewed for failures.

### Lessons Learned
- Start with a transparent baseline (spaCy + SST-2) to validate the pipeline quickly.
- Make idempotency and time-window semantics explicit to avoid data confusion.
- For client-side RAG on hosted platforms, prefer presigned/public data sources or prebuilt caches.
- Cache embeddings to keep the UX snappy on restarts.

### Next Improvements
- Add MMR/diversity in retrieval and inline citations in chat answers.
- Train a lightweight ABSA model with LoRA and distillation for higher precision.
- Add authentication to API and per-tenant isolation if multi-user.
- Expand search to OpenSearch for feature/phrase queries.

### Quick Troubleshooting
- Product Analysis 404: seed ASIN and use `window=10000d`.
- RAG shows 10 reviews on Cloud: set `RAG_REVIEWS_SOURCE` to presigned URL; rerun.
- API base URL: for REST API v1 → `https://{id}.execute-api.{region}.amazonaws.com/{stage}`.

---
These notes are intended as your “study sheet” for interviews and maintenance. Pair them with `README.md`, `docs/architecture.md`, and the code for complete context.


