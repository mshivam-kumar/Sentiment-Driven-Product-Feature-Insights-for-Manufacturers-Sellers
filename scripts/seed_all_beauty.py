import json
import time
from datasets import load_dataset
import boto3

REGION = us-east-1
LAMBDA_FN = sentiment-insights-inference

client = boto3.client(lambda, region_name=REGION)

ds = load_dataset(McAuley-Lab/Amazon-Reviews-2023, raw_review_All_Beauty, split=full)
invoked = 0
attempted = 0

for ex in ds.select(range(1500)):
    text = (ex.get(text) or ).strip()
    asin = ex.get(asin) or ex.get(parent_asin) or 
    parent_asin = ex.get(parent_asin) or ex.get(asin) or 
    if not text or not asin:
        continue
    payload = {
        text: text,
        asin: asin,
        parent_asin: parent_asin,
        user_id: ex.get(user_id) or seed_user,
        timestamp: int(ex.get(timestamp) or 0),
    }
    try:
        resp = client.invoke(
            FunctionName=LAMBDA_FN,
            InvocationType=RequestResponse,
            Payload=json.dumps(payload).encode(utf-8),
        )
        if resp.get(StatusCode) == 200:
            invoked += 1
    except Exception:
        pass
    attempted += 1
    if invoked >= 150:
        break
    time.sleep(0.03)

print(json.dumps({invoked_ok: invoked, attempted: attempted}))
