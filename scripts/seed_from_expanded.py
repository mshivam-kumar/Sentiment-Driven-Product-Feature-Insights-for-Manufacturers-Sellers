#!/usr/bin/env python3
"""
Seed DynamoDB aggregates by invoking the inference Lambda over an expanded JSONL corpus.

Usage examples:
  python scripts/seed_from_expanded.py \
    --source s3://skchatbotbucket/raw/All_Beauty/raw_review_All_Beauty_expanded.jsonl \
    --function-name sentiment-insights-inference \
    --region us-east-1 \
    --per-asin-limit 50

  python scripts/seed_from_expanded.py \
    --source data_ingest/data_ingest/raw_review_All_Beauty_expanded.jsonl \
    --function-name sentiment-insights-inference

Notes:
 - Streams JSONL and invokes Lambda once per review (small payloads)
 - Use --per-asin-limit to cap reviews per ASIN to control cost/time
 - Retries transient invoke errors; prints progress every N items
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Any, Iterable

import boto3

try:
    import s3fs  # optional for s3 streaming
except Exception:
    s3fs = None


def iter_jsonl(path: str, max_items: int = None) -> Iterable[Dict[str, Any]]:
    count = 0
    if path.startswith("s3://"):
        if s3fs is None:
            raise RuntimeError("s3fs not installed. pip install s3fs")
        fs = s3fs.S3FileSystem()
        with fs.open(path, "r") as f:
            for line in f:
                if max_items is not None and count >= max_items:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    yield obj
                    count += 1
                except Exception:
                    continue
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if max_items is not None and count >= max_items:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    yield obj
                    count += 1
                except Exception:
                    continue


def invoke_lambda(client, function_name: str, payload: Dict[str, Any]) -> bool:
    for attempt in range(3):
        try:
            resp = client.invoke(
                FunctionName=function_name,
                InvocationType="RequestResponse",
                Payload=json.dumps(payload).encode("utf-8"),
            )
            status = resp.get("StatusCode", 0)
            if status >= 200 and status < 300:
                return True
        except Exception as e:
            if attempt == 2:
                print(f"Invoke failed after retries: {e}")
                return False
            time.sleep(1.0 * (attempt + 1))
    return False


def main():
    parser = argparse.ArgumentParser(description="Seed DynamoDB aggregates by invoking inference Lambda over expanded reviews")
    parser.add_argument("--source", required=True, help="Path to expanded JSONL (s3://... or local path)")
    parser.add_argument("--function-name", required=True, help="Lambda function name for inference")
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "us-east-1"), help="AWS region")
    parser.add_argument("--max-items", type=int, default=None, help="Max total reviews to process")
    parser.add_argument("--per-asin-limit", type=int, default=50, help="Max reviews per ASIN")
    parser.add_argument("--sleep-ms", type=int, default=50, help="Sleep milliseconds between invokes")
    parser.add_argument("--asin-only", nargs='*', help="Seed only these ASINs (space-separated)")
    args = parser.parse_args()

    lambda_client = boto3.client("lambda", region_name=args.region)

    per_asin_counts: Dict[str, int] = defaultdict(int)
    total = 0
    succeeded = 0
    failed = 0

    for review in iter_jsonl(args.source, max_items=args.max_items):
        asin = review.get("asin") or review.get("parent_asin")
        if not asin:
            continue
        if args.asin_only:
            parent_asin = review.get("parent_asin")
            target_set = set(args.asin_only)
            if asin not in target_set and (parent_asin not in target_set if parent_asin else True):
                continue
        if per_asin_counts[asin] >= args.per_asin_limit:
            continue

        # Minimal payload expected by inference Lambda
        payload = {
            "text": review.get("text") or review.get("review_text") or "",
            "asin": review.get("asin", asin),
            "parent_asin": review.get("parent_asin", asin),
            "user_id": review.get("user_id", "seed_user"),
            "timestamp": int(review.get("timestamp", 0) or 0),
            "rating": review.get("rating", None),
        }

        ok = invoke_lambda(lambda_client, args.function_name, payload)
        total += 1
        per_asin_counts[asin] += 1
        if ok:
            succeeded += 1
        else:
            failed += 1

        if total % 25 == 0:
            print(f"Progress: total={total} ok={succeeded} fail={failed} unique_asins={len(per_asin_counts)}")

        # Throttle a bit to be polite
        time.sleep(args.sleep_ms / 1000.0)

    print("Done.")
    print(f"Processed={total} Succeeded={succeeded} Failed={failed} UniqueASINs={len(per_asin_counts)}")


if __name__ == "__main__":
    sys.exit(main())


