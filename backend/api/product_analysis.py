"""
Product Analysis API Endpoints
Handles ASIN-based sentiment analysis requests
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
import requests
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# AWS API Gateway URL (from your existing deployment)
API_BASE_URL = os.getenv('API_BASE_URL', 'https://f3157r5ca4.execute-api.us-east-1.amazonaws.com/dev')

@router.get("/product/{asin}")
async def get_product_analysis(
    asin: str,
    window: Optional[str] = Query(None, description="Time window: 1m, 3m, 6m, 1y, 10y, or None for all time")
):
    """
    Get comprehensive sentiment analysis for a specific product (ASIN)
    """
    try:
        # Call your existing AWS API
        url = f"{API_BASE_URL}/sentiment/product/{asin}"
        params = {}
        if window:
            params['window'] = window
            
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "data": data,
                "asin": asin,
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
            detail=f"Failed to fetch product analysis: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/product/{asin}/top-features")
async def get_top_features(
    asin: str,
    limit: int = Query(10, ge=1, le=50, description="Number of top features to return")
):
    """
    Get top features by sentiment for a specific product
    """
    try:
        url = f"{API_BASE_URL}/sentiment/product/{asin}/top-features"
        params = {'limit': limit}
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "data": data,
                "asin": asin,
                "limit": limit
            }
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"API Error: {response.text}"
            )
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch top features: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/product/{asin}/features")
async def get_product_features(asin: str):
    """
    Get all features for a specific product
    """
    try:
        url = f"{API_BASE_URL}/sentiment/product/{asin}"
        
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            # Extract features from the response
            features = []
            if 'features' in data:
                for feature in data['features']:
                    features.append({
                        "name": feature.get('feature', ''),
                        "sentiment": feature.get('sentiment', 0),
                        "count": feature.get('count', 0),
                        "snippets": feature.get('snippets', [])
                    })
            
            return {
                "success": True,
                "features": features,
                "asin": asin
            }
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"API Error: {response.text}"
            )
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch product features: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )



