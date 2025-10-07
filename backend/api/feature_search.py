"""
Feature Search API Endpoints
Handles cross-product feature search and comparison
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any, List
import requests
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# AWS API Gateway URL (from your existing deployment)
API_BASE_URL = os.getenv('API_BASE_URL', 'https://f3157r5ca4.execute-api.us-east-1.amazonaws.com/dev')

@router.get("/features/search")
async def search_features(
    query: str = Query(..., description="Feature search query"),
    category: Optional[str] = Query(None, description="Product category filter"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results"),
    window: Optional[str] = Query(None, description="Time window filter")
):
    """
    Search for features across all products
    """
    try:
        url = f"{API_BASE_URL}/sentiment/search"
        params = {
            'query': query,
            'limit': limit
        }
        
        if category:
            params['category'] = category
        if window:
            params['window'] = window
            
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "data": data,
                "query": query,
                "category": category,
                "limit": limit,
                "window": window
            }
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"API Error: {response.text}"
            )
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search features: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/features/compare")
async def compare_features(
    features: str = Query(..., description="Comma-separated list of features to compare"),
    category: Optional[str] = Query(None, description="Product category filter"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of products per feature")
):
    """
    Compare sentiment scores for multiple features
    """
    try:
        feature_list = [f.strip() for f in features.split(',')]
        
        results = {}
        for feature in feature_list:
            url = f"{API_BASE_URL}/sentiment/search"
            params = {
                'query': feature,
                'limit': limit
            }
            
            if category:
                params['category'] = category
                
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results[feature] = data
            else:
                results[feature] = {"error": f"Failed to fetch data for {feature}"}
        
        return {
            "success": True,
            "comparison": results,
            "features": feature_list,
            "category": category
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/features/trends")
async def get_feature_trends(
    feature: str = Query(..., description="Feature name to analyze trends"),
    category: Optional[str] = Query(None, description="Product category filter"),
    window: str = Query("1y", description="Time window for trend analysis")
):
    """
    Get sentiment trends for a specific feature over time
    """
    try:
        url = f"{API_BASE_URL}/sentiment/search"
        params = {
            'query': feature,
            'window': window,
            'limit': 50
        }
        
        if category:
            params['category'] = category
            
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Process data to extract trends
            trends = {
                "feature": feature,
                "category": category,
                "window": window,
                "total_mentions": len(data.get('results', [])),
                "average_sentiment": 0,
                "trend_data": []
            }
            
            if data.get('results'):
                sentiments = [item.get('sentiment', 0) for item in data['results']]
                trends['average_sentiment'] = sum(sentiments) / len(sentiments) if sentiments else 0
                
                # Group by time periods (simplified)
                for item in data['results']:
                    trends['trend_data'].append({
                        "asin": item.get('asin', ''),
                        "sentiment": item.get('sentiment', 0),
                        "timestamp": item.get('timestamp', ''),
                        "snippet": item.get('snippet', '')
                    })
            
            return {
                "success": True,
                "trends": trends
            }
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"API Error: {response.text}"
            )
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch feature trends: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )



