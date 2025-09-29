"""
API Gateway Lambda handler for sentiment insights API.

This module handles API Gateway requests and queries DynamoDB for sentiment insights.
"""

import json
import boto3
from boto3.dynamodb.conditions import Key
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import traceback


class SentimentAPIHandler:
    """Handler for sentiment insights API requests."""
    
    def __init__(self):
        """Initialize the API handler."""
        self.dynamodb = boto3.resource('dynamodb')
        self.table_name = os.environ.get('DYNAMODB_TABLE', 'product_sentiment_insights')
        self.table = self.dynamodb.Table(self.table_name)
    
    def get_product_sentiment(self, asin: str, feature: Optional[str] = None, 
                            window: Optional[str] = "30d") -> Dict[str, Any]:
        """
        Get sentiment insights for a product.
        
        Args:
            asin: Product ASIN
            feature: Optional specific feature to filter
            window: Time window for analysis
            
        Returns:
            Dictionary with sentiment insights
        """
        try:
            # Calculate time window - if None, use a very long time window (100 years)
            if window is None:
                days = 365 * 100  # 100 years
            else:
                days = self._parse_time_window(window)
            cutoff_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            # Query DynamoDB
            if feature:
                # Get specific feature
                response = self.table.get_item(
                    Key={'parent_asin': asin, 'feature': feature}
                )
                
                if 'Item' not in response:
                    return self._create_error_response("Feature not found", "FEATURE_NOT_FOUND")
                
                item = response['Item']
                if float(item.get('last_updated', 0)) < cutoff_time:
                    return self._create_error_response("No recent data", "NO_RECENT_DATA")
                
                return {
                    'asin': asin,
                    'parent_asin': asin,
                    'features': {
                        feature: self._format_feature_sentiment(item)
                    },
                    'total_reviews': int(item.get('agg_score_count', 0)),
                    'last_updated': datetime.fromtimestamp(float(item.get('last_updated', 0)) / 1000).isoformat(),
                    'category': item.get('category', 'Unknown')
                }
            else:
                # Get all features
                # Correct KeyConditionExpression usage with boto3 Key
                response = self.table.query(
                    KeyConditionExpression=Key('parent_asin').eq(asin)
                )
                
                if not response['Items']:
                    return self._create_error_response("Product not found", "PRODUCT_NOT_FOUND")
                
                features = {}
                total_reviews = 0
                last_updated = 0
                category = 'Unknown'
                
                for item in response['Items']:
                    if float(item.get('last_updated', 0)) >= cutoff_time:
                        feature_name = item['feature']
                        features[feature_name] = self._format_feature_sentiment(item)
                        total_reviews += int(item.get('agg_score_count', 0))
                        last_updated = max(last_updated, float(item.get('last_updated', 0)))
                        category = item.get('category', category)
                
                if not features:
                    return self._create_error_response("No recent data", "NO_RECENT_DATA")
                
                # Calculate overall sentiment
                overall_sentiment = sum(
                    f['score'] * f['count'] for f in features.values()
                ) / sum(f['count'] for f in features.values()) if features else 0
                
                return {
                    'asin': asin,
                    'parent_asin': asin,
                    'features': features,
                    'overall_sentiment': overall_sentiment,
                    'total_reviews': total_reviews,
                    'last_updated': datetime.fromtimestamp(last_updated / 1000).isoformat(),
                    'category': category
                }
        
        except Exception as e:
            print(f"Error getting product sentiment: {e}")
            return self._create_error_response("Internal server error", "INTERNAL_ERROR")
    
    def get_top_features(self, asin: str, limit: int = 10, sort: str = "score") -> Dict[str, Any]:
        """
        Get top features for a product.
        
        Args:
            asin: Product ASIN
            limit: Maximum number of features to return
            sort: Sort order (score, count, name)
            
        Returns:
            Dictionary with top features
        """
        try:
            response = self.table.query(
                KeyConditionExpression=Key('parent_asin').eq(asin)
            )
            
            if not response['Items']:
                return self._create_error_response("Product not found", "PRODUCT_NOT_FOUND")
            
            # Format and sort features
            features = []
            for item in response['Items']:
                feature_data = self._format_feature_sentiment(item)
                feature_data['feature'] = item['feature']
                features.append(feature_data)
            
            # Sort features
            if sort == "score":
                features.sort(key=lambda x: abs(x['score']), reverse=True)
            elif sort == "count":
                features.sort(key=lambda x: x['count'], reverse=True)
            elif sort == "name":
                features.sort(key=lambda x: x['feature'])
            
            # Limit results
            features = features[:limit]
            
            return {
                'asin': asin,
                'features': features,
                'total_features': len(response['Items'])
            }
        
        except Exception as e:
            print(f"Error getting top features: {e}")
            return self._create_error_response("Internal server error", "INTERNAL_ERROR")
    
    def search_features(self, query: str, category: Optional[str] = None, 
                       limit: int = 20) -> Dict[str, Any]:
        """
        Search for features across categories.
        
        Args:
            query: Search query
            category: Optional category filter
            limit: Maximum number of results
            
        Returns:
            Dictionary with search results
        """
        try:
            # This is a simplified search - in production, you'd use Elasticsearch or similar
            # For now, we'll scan the table (not recommended for large datasets)
            response = self.table.scan()
            
            results = []
            for item in response['Items']:
                feature = item['feature']
                if query.lower() in feature.lower():
                    if category is None or item.get('category') == category:
                        results.append({
                            'asin': item['parent_asin'],
                            'feature': feature,
                            'score': float(item.get('agg_score_sum', 0)) / max(float(item.get('agg_score_count', 1)), 1),
                            'count': int(item.get('agg_score_count', 0)),
                            'category': item.get('category', 'Unknown')
                        })
            
            # Sort by score and limit
            results.sort(key=lambda x: abs(x['score']), reverse=True)
            results = results[:limit]
            
            return {
                'query': query,
                'results': results,
                'total_results': len(results)
            }
        
        except Exception as e:
            print(f"Error searching features: {e}")
            return self._create_error_response("Internal server error", "INTERNAL_ERROR")
    
    def _format_feature_sentiment(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Format feature sentiment data."""
        count = float(item.get('agg_score_count', 0))
        score_sum = float(item.get('agg_score_sum', 0))
        avg_score = score_sum / max(count, 1)
        
        return {
            'score': avg_score,
            'count': int(count),
            'positive_snippets': item.get('positive_snippets', [])[:5],  # Top 5
            'negative_snippets': item.get('negative_snippets', [])[:5],  # Top 5
            'trend': self._calculate_trend(item)  # Simplified trend calculation
        }
    
    def _calculate_trend(self, item: Dict[str, Any]) -> str:
        """Calculate sentiment trend (simplified)."""
        # This is a placeholder - in production, you'd analyze historical data
        score = float(item.get('agg_score_sum', 0)) / max(float(item.get('agg_score_count', 1)), 1)
        if score > 0.5:
            return "increasing"
        elif score < -0.5:
            return "decreasing"
        else:
            return "stable"
    
    def _parse_time_window(self, window: str) -> int:
        """Parse time window string to days."""
        if window.endswith('d'):
            return int(window[:-1])
        elif window.endswith('w'):
            return int(window[:-1]) * 7
        elif window.endswith('m'):
            return int(window[:-1]) * 30
        elif window.endswith('y'):
            return int(window[:-1]) * 365
        else:
            return 30  # Default to 30 days
    
    def _create_error_response(self, message: str, code: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'error': message,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }


def lambda_handler(event, context):
    """
    AWS Lambda handler for API Gateway requests.
    
    Args:
        event: API Gateway event
        context: Lambda context
        
    Returns:
        API Gateway response
    """
    try:
        handler = SentimentAPIHandler()
        
        # Parse the request
        http_method = event.get('httpMethod', 'GET')
        path = event.get('path', '')
        path_parameters = event.get('pathParameters') or {}
        query_parameters = event.get('queryStringParameters') or {}
        
        # Route the request
        if path.startswith('/sentiment/product/') and path.endswith('/top-features'):
            # GET /sentiment/product/{asin}/top-features
            asin = path_parameters.get('asin')
            limit = int(query_parameters.get('limit', 10))
            sort = query_parameters.get('sort', 'score')
            
            result = handler.get_top_features(asin, limit, sort)
            
        elif path.startswith('/sentiment/product/'):
            # GET /sentiment/product/{asin}
            asin = path_parameters.get('asin')
            feature = query_parameters.get('feature')
            window = query_parameters.get('window')  # Don't set default here, let the method handle it
            
            result = handler.get_product_sentiment(asin, feature, window)
            
        elif path == '/sentiment/search':
            # GET /sentiment/search
            query = query_parameters.get('query')
            category = query_parameters.get('category')
            limit = int(query_parameters.get('limit', 20))
            
            if not query:
                result = handler._create_error_response("Query parameter required", "MISSING_QUERY")
            else:
                result = handler.search_features(query, category, limit)
        
        elif path == '/health':
            # GET /health
            result = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat()
            }
        
        else:
            result = handler._create_error_response("Not found", "NOT_FOUND")
        
        # Determine status code
        status_code = 200
        if 'error' in result:
            if result['code'] == 'PRODUCT_NOT_FOUND':
                status_code = 404
            elif result['code'] == 'MISSING_QUERY':
                status_code = 400
            else:
                status_code = 500
        
        # Return API Gateway response
        return {
            'statusCode': status_code,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'GET,POST,OPTIONS'
            },
            'body': json.dumps(result)
        }
    
    except Exception as e:
        print(f"Lambda handler error: {e}")
        print(traceback.format_exc())
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'code': 'INTERNAL_ERROR',
                'timestamp': datetime.now().isoformat()
            })
        }


def test_local():
    """Test function for local development."""
    # Mock API Gateway event
    event = {
        'httpMethod': 'GET',
        'path': '/sentiment/product/B00YQ6X8EO',
        'pathParameters': {'asin': 'B00YQ6X8EO'},
        'queryStringParameters': {'window': '30d'}
    }
    
    result = lambda_handler(event, None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    test_local()
