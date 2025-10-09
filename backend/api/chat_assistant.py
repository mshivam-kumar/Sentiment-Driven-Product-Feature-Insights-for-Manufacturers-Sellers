"""
AI Chat Assistant API Endpoints
Handles RAG-powered conversational interface
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional
import sys
import os
from dotenv import load_dotenv

# Add ml_models to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml_models'))

load_dotenv()

router = APIRouter()

# Initialize RAG system (lazy loading)
rag_system = None

def get_rag_system():
    """Get or initialize RAG system"""
    global rag_system
    if rag_system is None:
        try:
            from rag_module import RAGSystem
            # Try to use fine-tuned model if available
            fine_tuned_path = "./fine_tuned_tinyllama"
            rag_system = RAGSystem(fine_tuned_model_path=fine_tuned_path)
            
            # Load some sample reviews for demonstration with proper structure
            sample_reviews = [
                {
                    "asin": "B08JTNQFZY",
                    "parent_asin": "B08JTNQFZY",
                    "review_text": "Great product! The battery life is excellent and the design is sleek. Highly recommend for anyone looking for quality.",
                    "text": "Great product! The battery life is excellent and the design is sleek. Highly recommend for anyone looking for quality.",
                    "sentiment": 0.8,
                    "sentiment_score": 0.8,
                    "rating": 5,
                    "features": ["battery", "design", "quality"]
                },
                {
                    "asin": "B08JTNQFZY",
                    "parent_asin": "B08JTNQFZY", 
                    "review_text": "The customer service is outstanding. They were very helpful when I had questions about the product.",
                    "text": "The customer service is outstanding. They were very helpful when I had questions about the product.",
                    "sentiment": 0.9,
                    "sentiment_score": 0.9,
                    "rating": 5,
                    "features": ["customer_service", "service"]
                },
                {
                    "asin": "B08JTNQFZY",
                    "parent_asin": "B08JTNQFZY",
                    "review_text": "Good value for money. The product works as expected and the price is reasonable.",
                    "text": "Good value for money. The product works as expected and the price is reasonable.",
                    "sentiment": 0.7,
                    "sentiment_score": 0.7,
                    "rating": 4,
                    "features": ["value_for_money", "price"]
                },
                {
                    "asin": "B00YQ6X8EO",
                    "parent_asin": "B00YQ6X8EO",
                    "review_text": "The battery life could be better. It drains quickly with heavy usage.",
                    "text": "The battery life could be better. It drains quickly with heavy usage.",
                    "sentiment": -0.3,
                    "sentiment_score": -0.3,
                    "rating": 2,
                    "features": ["battery", "performance"]
                },
                {
                    "asin": "B00YQ6X8EO",
                    "parent_asin": "B00YQ6X8EO",
                    "review_text": "Love the compact size! Perfect for travel and fits easily in my bag.",
                    "text": "Love the compact size! Perfect for travel and fits easily in my bag.",
                    "sentiment": 0.6,
                    "sentiment_score": 0.6,
                    "rating": 4,
                    "features": ["size", "portability", "design"]
                },
                {
                    "asin": "B07LCHCD6Q",
                    "parent_asin": "B07LCHCD6Q",
                    "review_text": "Amazing quality! The build is solid and it feels premium. Worth every penny.",
                    "text": "Amazing quality! The build is solid and it feels premium. Worth every penny.",
                    "sentiment": 0.9,
                    "sentiment_score": 0.9,
                    "rating": 5,
                    "features": ["quality", "build", "premium"]
                },
                {
                    "asin": "B07LCHCD6Q",
                    "parent_asin": "B07LCHCD6Q",
                    "review_text": "Fast shipping and great packaging. The product arrived in perfect condition.",
                    "text": "Fast shipping and great packaging. The product arrived in perfect condition.",
                    "sentiment": 0.8,
                    "sentiment_score": 0.8,
                    "rating": 5,
                    "features": ["shipping", "packaging", "delivery"]
                }
            ]
            
            # Load sample reviews into RAG system
            if hasattr(rag_system, 'load_reviews'):
                rag_system.load_reviews(sample_reviews)
                print(f"✅ Loaded {len(sample_reviews)} sample reviews into RAG system")
            else:
                print("⚠️ RAG system doesn't support load_reviews method")
            
            print("✅ RAG system initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize RAG system: {e}")
            rag_system = None
    return rag_system

@router.post("/chat/query")
async def chat_query(
    question: str = Body(..., description="User's question"),
    use_transformer: bool = Body(True, description="Use transformer-based generation"),
    session_id: Optional[str] = Body(None, description="Session ID for conversation history")
):
    """
    Process a chat query using RAG system
    """
    try:
        rag = get_rag_system()
        
        if rag is None:
            # Fallback response when RAG is not available
            return {
                "success": True,
                "answer": "I'm sorry, the AI assistant is currently unavailable. Please try again later or use the Product Analysis and Feature Search features.",
                "generation_method": "fallback",
                "is_fine_tuned": False,
                "model_type": "unavailable",
                "supporting_reviews": [],
                "session_id": session_id
            }
        
        # Check if the query processing fails
        try:
            # Process the query
            response = rag.query(question, use_transformer=use_transformer)
            
            return {
                "success": True,
                "answer": response.get('answer', 'No response generated'),
                "generation_method": response.get('generation_method', 'unknown'),
                "is_fine_tuned": response.get('is_fine_tuned', False),
                "model_type": response.get('model_type', 'unknown'),
                "supporting_reviews": response.get('supporting_reviews', []),
                "session_id": session_id,
                "timestamp": response.get('timestamp', None)
            }
        except Exception as rag_error:
            # Fallback response when RAG processing fails
            return {
                "success": True,
                "answer": f"I'm having trouble processing your question right now. Here's what I can tell you: Based on our product analysis, customers generally appreciate good quality products with reliable battery life and clear displays. For more specific insights, please use the Product Analysis or Feature Search features.",
                "generation_method": "fallback",
                "is_fine_tuned": False,
                "model_type": "fallback",
                "supporting_reviews": [],
                "session_id": session_id,
                "error": str(rag_error)
            }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat query: {str(e)}"
        )

@router.get("/chat/status")
async def chat_status():
    """
    Get the status of the AI chat assistant
    """
    try:
        rag = get_rag_system()
        
        if rag is None:
            return {
                "available": False,
                "status": "RAG system not initialized",
                "features": {
                    "transformer_generation": False,
                    "semantic_search": False,
                    "fine_tuned_model": False
                }
            }
        
        return {
            "available": True,
            "status": "RAG system ready",
            "features": {
                "transformer_generation": True,
                "semantic_search": True,
                "fine_tuned_model": getattr(rag, 'is_fine_tuned', False)
            },
            "model_info": {
                "generation_model": getattr(rag, 'generation_model_name', 'Unknown'),
                "embeddings_model": getattr(rag, 'model_name', 'Unknown'),
                "reviews_loaded": len(getattr(rag, 'reviews_data', []))
            }
        }
        
    except Exception as e:
        return {
            "available": False,
            "status": f"Error: {str(e)}",
            "features": {
                "transformer_generation": False,
                "semantic_search": False,
                "fine_tuned_model": False
            }
        }

@router.post("/chat/example-questions")
async def get_example_questions():
    """
    Get example questions for the chat assistant
    """
    return {
        "success": True,
        "examples": [
            {
                "category": "Product Quality",
                "questions": [
                    "What do customers say about product quality?",
                    "How do customers rate the build quality?",
                    "What are the main quality issues mentioned?"
                ]
            },
            {
                "category": "Design & Features",
                "questions": [
                    "How do customers feel about the design?",
                    "What features do customers like most?",
                    "What design improvements do customers suggest?"
                ]
            },
            {
                "category": "Performance",
                "questions": [
                    "How does the product perform according to customers?",
                    "What performance issues are mentioned?",
                    "How satisfied are customers with performance?"
                ]
            },
            {
                "category": "Value & Price",
                "questions": [
                    "Is the product worth the price according to customers?",
                    "How do customers feel about the value for money?",
                    "What do customers say about pricing?"
                ]
            }
        ]
    }

@router.post("/chat/clear-session")
async def clear_session(
    session_id: str = Body(..., description="Session ID to clear")
):
    """
    Clear conversation history for a session
    """
    # In a real implementation, you'd store session data in a database
    # For now, we'll just return success
    return {
        "success": True,
        "message": f"Session {session_id} cleared successfully"
    }
