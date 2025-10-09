"""
Product Analysis API Endpoints
Handles ASIN-based sentiment analysis requests
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
import sys
import os
from dotenv import load_dotenv

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

load_dotenv()

router = APIRouter()

# Import data processor
from data_processor import data_processor

@router.get("/product/{asin}")
async def get_product_analysis(
    asin: str,
    window: Optional[str] = Query(None, description="Time window: 1m, 3m, 6m, 1y, 10y, or None for all time")
):
    """
    Get comprehensive sentiment analysis for a specific product (ASIN)
    """
    try:
        # Use real data from the data processor
        product_data = data_processor.get_product_analysis(asin, window)
        
        return {
            "success": True,
            "data": product_data,
            "asin": asin,
            "window": window
        }
            
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
        # Mock data for top features
        mock_features = [
            {"feature": "battery life", "sentiment": 0.8, "mentions": 45},
            {"feature": "build quality", "sentiment": 0.7, "mentions": 38},
            {"feature": "camera", "sentiment": 0.6, "mentions": 32},
            {"feature": "price", "sentiment": -0.3, "mentions": 28},
            {"feature": "screen", "sentiment": 0.5, "mentions": 25},
            {"feature": "performance", "sentiment": 0.4, "mentions": 22},
            {"feature": "design", "sentiment": 0.3, "mentions": 20},
            {"feature": "connectivity", "sentiment": 0.2, "mentions": 18},
            {"feature": "storage", "sentiment": 0.1, "mentions": 15},
            {"feature": "software", "sentiment": -0.1, "mentions": 12}
        ]
        
        return {
            "success": True,
            "data": mock_features[:limit],
            "asin": asin,
            "limit": limit
        }
            
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
        # Mock data for product features
        mock_features = [
            {
                "name": "battery life",
                "sentiment": 0.8,
                "count": 45,
                "snippets": ["Great battery life", "Lasts all day", "Excellent battery performance"]
            },
            {
                "name": "build quality",
                "sentiment": 0.7,
                "count": 38,
                "snippets": ["Solid build", "Well constructed", "Durable design"]
            },
            {
                "name": "camera",
                "sentiment": 0.6,
                "count": 32,
                "snippets": ["Good camera", "Nice photos", "Decent image quality"]
            },
            {
                "name": "price",
                "sentiment": -0.3,
                "count": 28,
                "snippets": ["A bit expensive", "Pricey", "Good value for money"]
            },
            {
                "name": "screen",
                "sentiment": 0.5,
                "count": 25,
                "snippets": ["Nice display", "Good screen", "Clear visuals"]
            }
        ]
        
        return {
            "success": True,
            "features": mock_features,
            "asin": asin
        }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )



