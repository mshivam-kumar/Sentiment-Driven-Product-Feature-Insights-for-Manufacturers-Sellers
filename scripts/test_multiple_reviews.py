#!/usr/bin/env python3
"""
Test script to simulate multiple reviews for the same product to show
how the system aggregates features across multiple reviews.
"""

import json
import boto3
import time

def test_multiple_reviews():
    """Test with multiple reviews for the same ASIN."""
    
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    
    # Test ASIN
    test_asin = "TEST_MULTI_123"
    
    # Multiple reviews for the same product
    reviews = [
        {
            'text': 'This product has excellent battery life that lasts all day. The camera quality is amazing with great photos.',
            'user_id': 'user1'
        },
        {
            'text': 'The design is beautiful and modern. However, the price is quite expensive and the customer service is terrible.',
            'user_id': 'user2'
        },
        {
            'text': 'The build quality feels solid and durable. Overall, great value for money despite the high cost.',
            'user_id': 'user3'
        },
        {
            'text': 'The sound quality is fantastic and the display is crystal clear. Very happy with this purchase.',
            'user_id': 'user4'
        },
        {
            'text': 'The shipping was fast and the packaging was excellent. The product arrived in perfect condition.',
            'user_id': 'user5'
        }
    ]
    
    print(f"Processing {len(reviews)} reviews for ASIN: {test_asin}")
    
    # Process each review
    for i, review in enumerate(reviews):
        payload = {
            'text': review['text'],
            'asin': test_asin,
            'parent_asin': test_asin,
            'user_id': review['user_id'],
            'timestamp': int(time.time() * 1000) + i
        }
        
        response = lambda_client.invoke(
            FunctionName='sentiment-insights-inference',
            InvocationType='RequestResponse',
            Payload=json.dumps(payload).encode('utf-8')
        )
        
        result = json.loads(response['Payload'].read())
        body = json.loads(result['body'])
        
        print(f"Review {i+1}: {len(body.get('aspects', {}))} aspects extracted")
        for aspect, data in body.get('aspects', {}).items():
            print(f"  {aspect}: {data['score']:.2f}")
        print()
    
    # Now test the API to see aggregated results
    print("Testing API with aggregated data...")
    import requests
    
    api_url = "https://f3157r5ca4.execute-api.us-east-1.amazonaws.com/dev/sentiment/product/" + test_asin
    response = requests.get(api_url)
    
    if response.status_code == 200:
        data = response.json()
        print(f"API Response - Features: {len(data.get('features', {}))}")
        for feature, info in data.get('features', {}).items():
            print(f"  {feature}: {info['score']:.2f} ({info['count']} reviews)")
    else:
        print(f"API Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_multiple_reviews()
