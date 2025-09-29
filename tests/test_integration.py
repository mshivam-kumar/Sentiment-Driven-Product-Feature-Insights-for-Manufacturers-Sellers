"""
Integration tests for the complete sentiment analysis pipeline.
"""

import pytest
import sys
import os
import json
import tempfile
from unittest.mock import Mock, patch

# Add the project directories to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models', 'aspect_extractor'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models', 'sentiment'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'inference', 'lambda_inference'))

from infer_aspect import AspectExtractor
from infer_sentiment import SentimentAnalyzer
from handler import ReviewProcessor


class TestIntegrationPipeline:
    """Integration tests for the complete pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.aspect_extractor = AspectExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end processing of a review."""
        review_data = {
            'text': 'Battery life is great but the camera is blurry.',
            'asin': 'B00YQ6X8EO',
            'parent_asin': 'B00YQ6X8EO',
            'user_id': 'test_user',
            'timestamp': 1588687728923
        }
        
        # Step 1: Extract aspects
        aspect_result = self.aspect_extractor.process_review(
            review_data['text'], 
            f"{review_data['user_id']}_{review_data['timestamp']}"
        )
        
        assert aspect_result['aspects'] != {}, "No aspects extracted"
        assert aspect_result['aspect_count'] > 0, "No aspects counted"
        
        # Step 2: Analyze sentiment for each aspect
        sentiment_result = self.sentiment_analyzer.process_review_with_aspects(
            review_data['text'],
            aspect_result['aspects'],
            f"{review_data['user_id']}_{review_data['timestamp']}",
            review_data['asin']
        )
        
        assert sentiment_result['aspects'] != {}, "No aspect sentiments found"
        assert 'overall_sentiment' in sentiment_result
        
        # Step 3: Verify the structure of the final result
        for aspect, sentiment_info in sentiment_result['aspects'].items():
            assert 'score' in sentiment_info
            assert 'sentence' in sentiment_info
            assert 'confidence' in sentiment_info
            assert 'relevant_sentences' in sentiment_info
            
            # Verify score is in valid range
            assert -1.0 <= sentiment_info['score'] <= 1.0, \
                f"Invalid sentiment score: {sentiment_info['score']}"
    
    def test_multiple_reviews_processing(self):
        """Test processing multiple reviews."""
        reviews = [
            {
                'text': 'Battery life is great but the camera is blurry.',
                'asin': 'B00YQ6X8EO',
                'parent_asin': 'B00YQ6X8EO',
                'user_id': 'user1',
                'timestamp': 1588687728923
            },
            {
                'text': 'The screen quality is excellent and the design is beautiful.',
                'asin': 'B00YQ6X8EO',
                'parent_asin': 'B00YQ6X8EO',
                'user_id': 'user2',
                'timestamp': 1588687728924
            },
            {
                'text': 'Fast shipping and good packaging, but the price is too high.',
                'asin': 'B00YQ6X8EO',
                'parent_asin': 'B00YQ6X8EO',
                'user_id': 'user3',
                'timestamp': 1588687728925
            }
        ]
        
        results = []
        for review in reviews:
            # Extract aspects
            aspect_result = self.aspect_extractor.process_review(
                review['text'],
                f"{review['user_id']}_{review['timestamp']}"
            )
            
            # Analyze sentiment
            sentiment_result = self.sentiment_analyzer.process_review_with_aspects(
                review['text'],
                aspect_result['aspects'],
                f"{review['user_id']}_{review['timestamp']}",
                review['asin']
            )
            
            results.append(sentiment_result)
        
        # Verify all reviews were processed
        assert len(results) == len(reviews)
        
        # Verify each result has the expected structure
        for result in results:
            assert 'aspects' in result
            assert 'overall_sentiment' in result
            assert isinstance(result['overall_sentiment'], (int, float))
    
    def test_data_ingestion_integration(self):
        """Test integration with data ingestion format."""
        # Sample data in the format from data_ingest/download_and_upload.py
        sample_review = {
            'rating': 5.0,
            'title': 'Such a lovely scent but not overpowering.',
            'text': "This spray is really nice. It smells really good, goes on really fine, and does the trick. I will say it feels like you need a lot of it though to get the texture I want. I have a lot of hair, medium thickness. I am comparing to other brands with yucky chemicals so I'm gonna stick with this. Try it!",
            'asin': 'B00YQ6X8EO',
            'parent_asin': 'B00YQ6X8EO',
            'timestamp': 1588687728923,
            'user_id': 'AGKHLEW2SOWHNMFQIJGBECAF7INQ',
            'verified_purchase': True
        }
        
        # Process the review
        aspect_result = self.aspect_extractor.process_review(
            sample_review['text'],
            f"{sample_review['user_id']}_{sample_review['timestamp']}"
        )
        
        sentiment_result = self.sentiment_analyzer.process_review_with_aspects(
            sample_review['text'],
            aspect_result['aspects'],
            f"{sample_review['user_id']}_{sample_review['timestamp']}",
            sample_review['asin']
        )
        
        # Verify the result structure matches expected format
        expected_structure = {
            'review_id': str,
            'asin': str,
            'aspects': dict,
            'overall_sentiment': (int, float)
        }
        
        for key, expected_type in expected_structure.items():
            assert key in sentiment_result, f"Missing key: {key}"
            assert isinstance(sentiment_result[key], expected_type), \
                f"Wrong type for {key}: {type(sentiment_result[key])}"
    
    @patch('boto3.resource')
    def test_lambda_handler_integration(self, mock_boto3):
        """Test Lambda handler integration."""
        # Mock DynamoDB
        mock_table = Mock()
        mock_table.update_item.return_value = {}
        mock_boto3.return_value.Table.return_value = mock_table
        
        # Mock environment variables
        with patch.dict(os.environ, {'DYNAMODB_TABLE': 'test_table'}):
            from handler import lambda_handler
            
            # Test event
            event = {
                'text': 'Battery life is great but the camera is blurry.',
                'asin': 'B00YQ6X8EO',
                'parent_asin': 'B00YQ6X8EO',
                'user_id': 'test_user',
                'timestamp': 1588687728923
            }
            
            # Call the handler
            result = lambda_handler(event, None)
            
            # Verify response structure
            assert 'statusCode' in result
            assert result['statusCode'] == 200
            assert 'body' in result
            
            # Parse the response body
            response_body = json.loads(result['body'])
            assert 'review_id' in response_body
            assert 'asin' in response_body
            assert 'aspects' in response_body
    
    def test_error_handling_integration(self):
        """Test error handling in the integration pipeline."""
        # Test with malformed review data
        malformed_review = {
            'text': '',  # Empty text
            'asin': 'B00YQ6X8EO',
            'parent_asin': 'B00YQ6X8EO',
            'user_id': 'test_user',
            'timestamp': 1588687728923
        }
        
        # Should handle empty text gracefully
        aspect_result = self.aspect_extractor.process_review(
            malformed_review['text'],
            f"{malformed_review['user_id']}_{malformed_review['timestamp']}"
        )
        
        # Should return empty aspects for empty text
        assert aspect_result['aspects'] == {}
        assert aspect_result['aspect_count'] == 0
        
        # Sentiment analysis should also handle empty text
        sentiment_result = self.sentiment_analyzer.process_review_with_aspects(
            malformed_review['text'],
            aspect_result['aspects'],
            f"{malformed_review['user_id']}_{malformed_review['timestamp']}",
            malformed_review['asin']
        )
        
        assert sentiment_result['aspects'] == {}
        assert sentiment_result['overall_sentiment'] == 0.0
    
    def test_performance_integration(self):
        """Test performance of the integrated pipeline."""
        import time
        
        # Create test data
        reviews = [
            'Battery life is great but the camera is blurry.',
            'The screen quality is excellent and the design is beautiful.',
            'Fast shipping and good packaging, but the price is too high.',
            'Easy to use interface, but customer service is terrible.',
            'Great performance and build quality, worth the money.'
        ] * 10  # 50 reviews total
        
        start_time = time.time()
        
        for i, review_text in enumerate(reviews):
            # Extract aspects
            aspect_result = self.aspect_extractor.process_review(review_text, f"review_{i}")
            
            # Analyze sentiment
            self.sentiment_analyzer.process_review_with_aspects(
                review_text,
                aspect_result['aspects'],
                f"review_{i}",
                f"ASIN_{i % 5}"  # 5 different ASINs
            )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 50 reviews in less than 60 seconds
        assert processing_time < 60, f"Processing took too long: {processing_time:.2f} seconds"
        
        # Average processing time should be less than 1.2 seconds per review
        avg_time = processing_time / len(reviews)
        assert avg_time < 1.2, f"Average processing time too slow: {avg_time:.3f} seconds per review"
    
    def test_data_quality_integration(self):
        """Test data quality in the integrated pipeline."""
        # Test with various review qualities
        test_cases = [
            {
                'text': 'Excellent product! Highly recommended.',
                'expected_aspects': ['product', 'quality'],
                'expected_sentiment': 'positive'
            },
            {
                'text': 'Terrible quality, waste of money.',
                'expected_aspects': ['quality', 'value'],
                'expected_sentiment': 'negative'
            },
            {
                'text': 'It is okay, nothing special.',
                'expected_aspects': ['product'],
                'expected_sentiment': 'neutral'
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            # Process the review
            aspect_result = self.aspect_extractor.process_review(
                test_case['text'], f"quality_test_{i}"
            )
            
            sentiment_result = self.sentiment_analyzer.process_review_with_aspects(
                test_case['text'],
                aspect_result['aspects'],
                f"quality_test_{i}",
                f"ASIN_{i}"
            )
            
            # Verify aspects were extracted
            assert len(aspect_result['aspects']) > 0, \
                f"No aspects extracted for: {test_case['text']}"
            
            # Verify sentiment analysis was performed
            assert len(sentiment_result['aspects']) > 0, \
                f"No aspect sentiments found for: {test_case['text']}"
            
            # Verify overall sentiment is reasonable
            overall_sentiment = sentiment_result['overall_sentiment']
            assert -1.0 <= overall_sentiment <= 1.0, \
                f"Invalid overall sentiment: {overall_sentiment}"


if __name__ == "__main__":
    pytest.main([__file__])
