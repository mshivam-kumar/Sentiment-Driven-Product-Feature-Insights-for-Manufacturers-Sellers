"""
Lambda inference handler for aspect extraction and sentiment analysis.

This module handles the inference pipeline for processing reviews and extracting
aspect-sentiment insights.
"""

import json
import boto3
import os
from typing import Dict, List, Any
import sys
import traceback
from decimal import Decimal
from time import time as _time
# Import the modules directly since they're in the same directory
try:
    from infer_aspect import AspectExtractor
except Exception as e:
    print(f"Import error for AspectExtractor: {e}")
    raise e

# Provide a lightweight sentiment fallback if transformers are unavailable
class _LightSentiment:
    def split_into_sentences(self, text: str):
        import re as _re
        parts = _re.findall(r"[^.!?]+[.!?]?", text or "")
        out = []
        for p in parts:
            s = p.strip()
            if s.endswith('.'):
                s = s[:-1]
            if s:
                out.append(s)
        return out

    def analyze_sentence_sentiment(self, sentence: str) -> float:
        s = (sentence or "").lower()
        pos = ["great", "good", "excellent", "amazing", "love", "nice", "worth"]
        neg = ["blurry", "bad", "terrible", "poor", "hate", "broken", "waste"]
        if any(w in s for w in pos):
            return 0.7
        if any(w in s for w in neg):
            return -0.7
        if any(w in s for w in ["ok", "okay", "fine", "average"]):
            return 0.0
        return 0.0

    def map_sentiment_to_aspects(self, text, aspects):
        sentences = self.split_into_sentences(text)
        result = {}
        for a, conf in (aspects or {}).items():
            relevant = [s for s in sentences if any(tok in s.lower() for tok in a.split('_'))] or (sentences[:1] if sentences else [])
            if relevant:
                scores = [self.analyze_sentence_sentiment(s) for s in relevant]
                score = sum(scores) / max(len(scores), 1)
                best = relevant[0]
            else:
                score = 0.0
                best = text[:100] + "..." if text and len(text) > 100 else (text or "")
                relevant = [best] if best else []
            result[a] = {"score": float(score), "sentence": best, "confidence": float(conf), "relevant_sentences": relevant}
        return result

    def process_review_with_aspects(self, review_text, aspects, review_id=None, asin=None):
        mapping = self.map_sentiment_to_aspects(review_text or "", aspects or {})
        overall = 0.0
        return {"review_id": review_id, "asin": asin, "text": review_text or "", "aspects": mapping, "overall_sentiment": overall}

try:
    from infer_sentiment import SentimentAnalyzer as _FullSentiment
    SentimentAnalyzer = _FullSentiment
except Exception as e:
    print(f"Import error for SentimentAnalyzer: {e}")
    SentimentAnalyzer = _LightSentiment


class ReviewProcessor:
    """Main processor for review analysis."""
    
    def __init__(self):
        """Initialize the processor with models."""
        try:
            self.aspect_extractor = AspectExtractor()
            self.sentiment_analyzer = SentimentAnalyzer()
            self.dynamodb = boto3.resource('dynamodb')
            self.table_name = os.environ.get('DYNAMODB_TABLE', 'product_sentiment_insights')
            self.table = self.dynamodb.Table(self.table_name)
        except Exception as e:
            print(f"Error initializing ReviewProcessor: {e}")
            raise
    
    def process_review(self, review_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single review and extract aspect-sentiment insights.
        
        Args:
            review_data: Dictionary containing review information
            
        Returns:
            Dictionary with aspect-sentiment analysis results
        """
        try:
            review_text = review_data.get('text', '')
            asin = review_data.get('asin', '')
            parent_asin = review_data.get('parent_asin', asin)
            review_ts = int(review_data.get('timestamp') or 0)
            review_id = review_data.get('user_id', '') + '_' + str(review_ts)
            
            # Extract aspects
            aspect_result = self.aspect_extractor.process_review(review_text, review_id)
            aspects = aspect_result['aspects']
            
            if not aspects:
                return {
                    'review_id': review_id,
                    'asin': asin,
                    'parent_asin': parent_asin,
                    'aspects': {},
                    'status': 'no_aspects_found'
                }
            
            # Analyze sentiment for each aspect
            sentiment_result = self.sentiment_analyzer.process_review_with_aspects(
                review_text, aspects, review_id, asin
            )
            
            return {
                'review_id': review_id,
                'asin': asin,
                'parent_asin': parent_asin,
                'aspects': sentiment_result['aspects'],
                'overall_sentiment': sentiment_result['overall_sentiment'],
                'timestamp': review_ts,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Error processing review: {e}")
            return {
                'review_id': review_data.get('user_id', '') + '_' + str(review_data.get('timestamp', '')),
                'asin': review_data.get('asin', ''),
                'parent_asin': review_data.get('parent_asin', review_data.get('asin', '')),
                'aspects': {},
                'status': 'error',
                'error': str(e)
            }
    
    def update_dynamodb(self, result: Dict[str, Any]) -> bool:
        try:
            parent_asin = result['parent_asin']
            aspects = result['aspects']
            ts_value = int(result.get('timestamp') or int(_time() * 1000))
            ts_dec = Decimal(str(ts_value))
            one = Decimal('1')
            zero = Decimal('0')

            for aspect, sentiment_info in aspects.items():
                score = float(sentiment_info['score'])
                score_dec = Decimal(str(score))

                self.table.update_item(
                    Key={'parent_asin': parent_asin, 'feature': aspect},
                    UpdateExpression="""
                        SET agg_score_sum = if_not_exists(agg_score_sum, :zero) + :score,
                            agg_score_count = if_not_exists(agg_score_count, :zero) + :one,
                            last_updated = :ts,
                            category = :cat
                    """,
                    ExpressionAttributeValues={
                        ':score': score_dec,
                        ':one': one,
                        ':zero': zero,
                        ':ts': ts_dec,
                        ':cat': 'All_Beauty'
                    }
                )

                if score > 0.5:
                    self.table.update_item(
                        Key={'parent_asin': parent_asin, 'feature': aspect},
                        UpdateExpression="""
                            SET positive_snippets = list_append(if_not_exists(positive_snippets, :empty), :snippet)
                        """,
                        ExpressionAttributeValues={
                            ':empty': [],
                            ':snippet': [sentiment_info['sentence']]
                        }
                    )
                elif score < -0.5:
                    self.table.update_item(
                        Key={'parent_asin': parent_asin, 'feature': aspect},
                        UpdateExpression="""
                            SET negative_snippets = list_append(if_not_exists(negative_snippets, :empty), :snippet)
                        """,
                        ExpressionAttributeValues={
                            ':empty': [],
                            ':snippet': [sentiment_info['sentence']]
                        }
                    )
            return True
        except Exception as e:
            print(f"Error updating DynamoDB: {e}")
            return False


    # def update_dynamodb(self, result: Dict[str, Any]) -> bool:
    #     """
    #     Update DynamoDB with aggregated sentiment data.
        
    #     Args:
    #         result: Processed review result
            
    #     Returns:
    #         Boolean indicating success
    #     """
    #     try:
    #         parent_asin = result['parent_asin']
    #         aspects = result['aspects']
            
    #         for aspect, sentiment_info in aspects.items():
    #             score = sentiment_info['score']
    #             confidence = sentiment_info['confidence']
    #             # Use event timestamp if present, else current time
    #             from time import time as _time
    #             ts_value = int(result.get('timestamp') or int(_time() * 1000))
                
    #             # Update DynamoDB with atomic operations
    #             self.table.update_item(
    #                 Key={
    #                     'parent_asin': parent_asin,
    #                     'feature': aspect
    #                 },
    #                 UpdateExpression="""
    #                     SET agg_score_sum = if_not_exists(agg_score_sum, :zero) + :score,
    #                         agg_score_count = if_not_exists(agg_score_count, :zero) + :one,
    #                         last_updated = :timestamp,
    #                         category = :category
    #                 """,
    #                 ExpressionAttributeValues={
    #                     ':score': score,
    #                     ':one': 1,
    #                     ':zero': 0,
    #                     ':timestamp': ts_value,
    #                     ':category': 'All_Beauty'  # This should be dynamic
    #                 }
    #             )
                
    #             # Store positive/negative snippets
    #             if score > 0.5:
    #                 self.table.update_item(
    #                     Key={
    #                         'parent_asin': parent_asin,
    #                         'feature': aspect
    #                     },
    #                     UpdateExpression="""
    #                         SET positive_snippets = list_append(if_not_exists(positive_snippets, :empty_list), :snippet)
    #                     """,
    #                     ExpressionAttributeValues={
    #                         ':empty_list': [],
    #                         ':snippet': [sentiment_info['sentence']]
    #                     }
    #                 )
    #             elif score < -0.5:
    #                 self.table.update_item(
    #                     Key={
    #                         'parent_asin': parent_asin,
    #                         'feature': aspect
    #                     },
    #                     UpdateExpression="""
    #                         SET negative_snippets = list_append(if_not_exists(negative_snippets, :empty_list), :snippet)
    #                     """,
    #                     ExpressionAttributeValues={
    #                         ':empty_list': [],
    #                         ':snippet': [sentiment_info['sentence']]
    #                     }
    #                 )
            
    #         return True
            
    #     except Exception as e:
    #         print(f"Error updating DynamoDB: {e}")
    #         return False


def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Args:
        event: Lambda event data
        context: Lambda context
        
    Returns:
        Response dictionary
    """
    try:
        processor = ReviewProcessor()
        
        # Handle different event types
        if 'Records' in event:
            # SQS event
            results = []
            for record in event['Records']:
                try:
                    # Parse SQS message
                    if 'body' in record:
                        review_data = json.loads(record['body'])
                    else:
                        review_data = record
                    
                    # Process review
                    result = processor.process_review(review_data)
                    
                    # Update DynamoDB
                    if result['status'] == 'success':
                        processor.update_dynamodb(result)
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error processing record: {e}")
                    results.append({
                        'status': 'error',
                        'error': str(e)
                    })
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Processing completed',
                    'results': results
                })
            }
        
        elif 'review_text' in event or 'text' in event:
            # Direct API call (support both 'review_text' and 'text')
            payload = event
            if 'review_text' in event and 'text' not in event:
                payload = dict(event)
                payload['text'] = event.get('review_text', '')
            result = processor.process_review(payload)
            
            if result['status'] == 'success':
                processor.update_dynamodb(result)
            
            return {
                'statusCode': 200,
                'body': json.dumps(result)
            }
        
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Invalid event format'
                })
            }
    
    except Exception as e:
        print(f"Lambda handler error: {e}")
        print(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }


def test_local():
    """Test function for local development."""
    sample_review = {
        'text': 'Battery life is great but the camera is blurry.',
        'asin': 'B00YQ6X8EO',
        'parent_asin': 'B00YQ6X8EO',
        'user_id': 'test_user',
        'timestamp': 1588687728923
    }
    
    # Mock event
    event = sample_review
    
    result = lambda_handler(event, None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    test_local()
