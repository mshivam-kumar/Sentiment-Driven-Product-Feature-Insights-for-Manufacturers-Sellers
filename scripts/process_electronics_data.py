#!/usr/bin/env python3
"""
Process Electronics data and add to the system
"""

import os
import sys
import json
import boto3
from decimal import Decimal

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def process_electronics_data():
    """Process Electronics data and add to DynamoDB"""
    try:
        print("📊 Processing Electronics data...")
        
        # Read electronics data
        electronics_file = "data_ingest/data_ingest/raw_review_Electronics.jsonl"
        if not os.path.exists(electronics_file):
            print("❌ Electronics data file not found. Please run download_electronics.py first.")
            return False
        
        reviews = []
        with open(electronics_file, 'r') as f:
            for line in f:
                if line.strip():
                    reviews.append(json.loads(line.strip()))
        
        print(f"📖 Loaded {len(reviews)} Electronics reviews")
        
        # Initialize AWS clients
        lambda_client = boto3.client('lambda', region_name='us-east-1')
        
        # Process reviews through inference Lambda
        processed_count = 0
        for i, review in enumerate(reviews[:100]):  # Process first 100 reviews
            try:
                # Prepare event for Lambda
                event = {
                    'review_text': review.get('review_text', ''),
                    'parent_asin': review.get('parent_asin', ''),
                    'rating': review.get('rating', 0),
                    'timestamp': '2024-01-01T00:00:00Z'  # Default timestamp
                }
                
                # Invoke Lambda
                response = lambda_client.invoke(
                    FunctionName='sentiment-insights-inference',
                    InvocationType='RequestResponse',
                    Payload=json.dumps(event)
                )
                
                result = json.loads(response['Payload'].read())
                if result.get('statusCode') == 200:
                    processed_count += 1
                
                if (i + 1) % 10 == 0:
                    print(f"📈 Processed {i + 1}/{min(100, len(reviews))} reviews...")
                    
            except Exception as e:
                print(f"⚠️ Error processing review {i}: {e}")
                continue
        
        print(f"✅ Successfully processed {processed_count} Electronics reviews")
        return True
        
    except Exception as e:
        print(f"❌ Error processing Electronics data: {e}")
        return False

if __name__ == "__main__":
    success = process_electronics_data()
    if success:
        print("🎉 Electronics data processing completed!")
    else:
        print("💥 Electronics data processing failed!")
