#!/usr/bin/env python3
"""
Clear DynamoDB table and re-seed with improved aspect extraction.
"""

import boto3
import json
import time
from typing import Dict, List, Any
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TableManager:
    """Manage DynamoDB table operations."""
    
    def __init__(self):
        """Initialize the table manager."""
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        self.table = self.dynamodb.Table('product_sentiment_insights')
        self.lambda_client = boto3.client('lambda', region_name='us-east-1')
        self.function_name = 'sentiment-insights-inference'
        
    def clear_table(self):
        """Clear all items from the table."""
        logger.info("Clearing DynamoDB table...")
        
        # Get all items
        response = self.table.scan()
        items = response['Items']
        
        # Continue scanning if there are more items
        while 'LastEvaluatedKey' in response:
            response = self.table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            items.extend(response['Items'])
        
        logger.info(f"Found {len(items)} items to delete")
        
        # Delete items in batches
        with self.table.batch_writer() as batch:
            for item in items:
                batch.delete_item(
                    Key={
                        'parent_asin': item['parent_asin'],
                        'feature': item['feature']
                    }
                )
        
        logger.info("Table cleared successfully")
    
    def load_reviews(self, file_path: str) -> List[Dict[str, Any]]:
        """Load reviews from JSONL file."""
        reviews = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        review = json.loads(line.strip())
                        reviews.append(review)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                        continue
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return []
        
        logger.info(f"Loaded {len(reviews)} reviews from {file_path}")
        return reviews
    
    def invoke_lambda(self, review_data: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke Lambda function for a single review."""
        try:
            # Ensure we have all required fields
            payload = {
                'text': review_data.get('text', ''),
                'asin': review_data.get('asin', ''),
                'parent_asin': review_data.get('parent_asin', review_data.get('asin', '')),
                'user_id': review_data.get('user_id', 'unknown'),
                'timestamp': review_data.get('timestamp', int(time.time() * 1000))
            }
            
            response = self.lambda_client.invoke(
                FunctionName=self.function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload).encode('utf-8')
            )
            
            status_code = response.get('StatusCode', 0)
            if status_code == 200:
                return {'success': True, 'asin': payload['asin'], 'status_code': status_code}
            else:
                return {'success': False, 'asin': payload['asin'], 'status_code': status_code}
                
        except Exception as e:
            logger.error(f"Error invoking Lambda for ASIN {review_data.get('asin', 'unknown')}: {e}")
            return {'success': False, 'asin': review_data.get('asin', 'unknown'), 'error': str(e)}
    
    def reseed_data(self, data_file: str) -> Dict[str, Any]:
        """Re-seed the table with improved data."""
        logger.info("Starting re-seeding with improved aspect extraction...")
        
        # Load reviews
        reviews = self.load_reviews(data_file)
        if not reviews:
            return {'error': 'No reviews loaded'}
        
        # Filter out reviews with empty or very short text
        valid_reviews = [
            review for review in reviews 
            if review.get('text', '').strip() and len(review.get('text', '').strip()) > 10
        ]
        
        logger.info(f"Processing {len(valid_reviews)} valid reviews")
        
        results = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'unique_asins': set(),
            'errors': []
        }
        
        # Process reviews
        for i, review in enumerate(valid_reviews):
            if i % 10 == 0:
                logger.info(f"Processing review {i+1}/{len(valid_reviews)}")
            
            result = self.invoke_lambda(review)
            results['total_processed'] += 1
            results['unique_asins'].add(review.get('asin', ''))
            
            if result['success']:
                results['successful'] += 1
            else:
                results['failed'] += 1
                if 'error' in result:
                    results['errors'].append(result['error'])
            
            # Small delay to avoid overwhelming the Lambda
            time.sleep(0.1)
        
        results['unique_asins'] = len(results['unique_asins'])
        return results

def main():
    """Main function."""
    # Path to the data file
    data_file = "data_ingest/data_ingest/raw_review_All_Beauty.jsonl"
    
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return
    
    manager = TableManager()
    
    # Clear the table
    manager.clear_table()
    
    # Re-seed with improved data
    results = manager.reseed_data(data_file)
    
    print("\n" + "="*50)
    print("RE-SEEDING RESULTS")
    print("="*50)
    print(f"Total reviews processed: {results.get('total_processed', 0)}")
    print(f"Successful: {results.get('successful', 0)}")
    print(f"Failed: {results.get('failed', 0)}")
    print(f"Unique ASINs: {results.get('unique_asins', 0)}")
    
    if results.get('errors'):
        print(f"\nErrors encountered: {len(results['errors'])}")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")

if __name__ == "__main__":
    main()
