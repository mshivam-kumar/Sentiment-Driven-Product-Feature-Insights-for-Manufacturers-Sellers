#!/usr/bin/env python3
"""
Process the expanded All_Beauty dataset to populate DynamoDB with more real data.
"""

import json
import boto3
import time
import logging
from typing import Dict, List, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpandedDataProcessor:
    """Process expanded All_Beauty dataset."""
    
    def __init__(self):
        """Initialize the processor."""
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        self.table = self.dynamodb.Table('product_sentiment_insights')
        self.lambda_client = boto3.client('lambda', region_name='us-east-1')
        self.function_name = 'sentiment-insights-inference'
        
    def clear_existing_data(self):
        """Clear existing data to start fresh."""
        logger.info("Clearing existing data...")
        
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
        
        logger.info("Existing data cleared successfully")
    
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
    
    def process_reviews(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process reviews and populate DynamoDB."""
        logger.info(f"Processing {len(reviews)} reviews...")
        
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
        
        # Process reviews in batches
        batch_size = 20
        for i in range(0, len(valid_reviews), batch_size):
            batch = valid_reviews[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(valid_reviews) + batch_size - 1)//batch_size}")
            
            for review in batch:
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
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database."""
        response = self.table.scan(Select='COUNT')
        total_items = response['Count']
        
        # Continue scanning if there are more items
        while 'LastEvaluatedKey' in response:
            response = self.table.scan(Select='COUNT', ExclusiveStartKey=response['LastEvaluatedKey'])
            total_items += response['Count']
        
        # Get unique ASINs
        asins = set()
        response = self.table.scan(ProjectionExpression='parent_asin')
        for item in response['Items']:
            asins.add(item['parent_asin'])
        
        while 'LastEvaluatedKey' in response:
            response = self.table.scan(ProjectionExpression='parent_asin', ExclusiveStartKey=response['LastEvaluatedKey'])
            for item in response['Items']:
                asins.add(item['parent_asin'])
        
        return {
            'total_items': total_items,
            'unique_asins': len(asins)
        }

def main():
    """Main function."""
    data_file = "data_ingest/data_ingest/raw_review_All_Beauty_expanded.jsonl"
    
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return
    
    processor = ExpandedDataProcessor()
    
    # Clear existing data
    processor.clear_existing_data()
    
    # Load expanded reviews
    reviews = processor.load_reviews(data_file)
    
    # Process reviews
    results = processor.process_reviews(reviews)
    
    # Get final database stats
    stats = processor.get_database_stats()
    
    print("\n" + "="*60)
    print("EXPANDED ALL_BEAUTY DATA PROCESSING RESULTS")
    print("="*60)
    print(f"Reviews processed: {results.get('total_processed', 0)}")
    print(f"Successful: {results.get('successful', 0)}")
    print(f"Failed: {results.get('failed', 0)}")
    print(f"Unique ASINs processed: {results.get('unique_asins', 0)}")
    print(f"Total items in database: {stats['total_items']}")
    print(f"Unique ASINs in database: {stats['unique_asins']}")
    
    if results.get('errors'):
        print(f"\nErrors encountered: {len(results['errors'])}")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")

if __name__ == "__main__":
    main()
