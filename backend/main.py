"""
FastAPI Backend for Sentiment-Driven Product Feature Insights
Author: Shivam Kumar - IIT Gandhinagar
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import API routers
from api.product_analysis import router as product_router
from api.feature_search import router as feature_router
from api.chat_assistant import router as chat_router
from core.database import get_database

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment-Driven Product Feature Insights API",
    description="AI-powered product analytics platform for manufacturers and sellers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001", 
        "http://127.0.0.1:3000",
        "http://sentiment-analysis-alb-1018237225.us-east-1.elb.amazonaws.com",  # ALB domain
        "https://sentiment-analysis-alb-1018237225.us-east-1.elb.amazonaws.com"  # ALB domain with HTTPS
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(product_router, prefix="/api/v1", tags=["Product Analysis"])
app.include_router(feature_router, prefix="/api/v1", tags=["Feature Search"])
app.include_router(chat_router, prefix="/api/v1", tags=["AI Chat Assistant"])

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Sentiment-Driven Product Feature Insights API",
        "author": "Shivam Kumar - IIT Gandhinagar",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "product_analysis": "/api/v1/product",
            "feature_search": "/api/v1/features",
            "chat_assistant": "/api/v1/chat"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "sentiment-insights-api"}

@app.get("/api/health")
async def api_health_check():
    """API health check endpoint for ALB"""
    return {"status": "healthy", "service": "sentiment-insights-api"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
