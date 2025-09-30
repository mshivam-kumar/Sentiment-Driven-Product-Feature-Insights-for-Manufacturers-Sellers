"""
RAG (Retrieval-Augmented Generation) Module for Sentiment Analysis

This module provides context-aware responses by combining:
1. Review text retrieval using semantic search
2. LLM-generated responses based on retrieved context
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Optional imports for RAG functionality
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("RAG dependencies not installed. Install with: pip install sentence-transformers scikit-learn")


@dataclass
class ReviewContext:
    """Represents a review with its context."""
    text: str
    sentiment: float
    asin: str
    rating: int
    relevance_score: float


class RAGSystem:
    """
    Retrieval-Augmented Generation system for sentiment analysis.
    
    Provides intelligent responses by:
    1. Retrieving relevant reviews based on user queries
    2. Generating contextual insights about products/features
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """Initialize the RAG system."""
        self.model_name = model_name
        self.embeddings_model = None
        self.review_embeddings = None
        self.reviews_data = []
        
        if RAG_AVAILABLE:
            try:
                self.embeddings_model = SentenceTransformer(model_name)
                print(f"✅ RAG system initialized with {model_name}")
            except Exception as e:
                print(f"❌ Failed to load embeddings model: {e}")
                # Don't modify global variable, just set local flag
                self.embeddings_model = None
    
    def load_reviews(self, reviews_data: List[Dict[str, Any]]):
        """Load review data for RAG processing."""
        self.reviews_data = reviews_data
        
        if not RAG_AVAILABLE:
            print("⚠️ RAG not available, using simple text matching")
            return
        
        # Extract and embed review texts
        review_texts = []
        for review in reviews_data:
            text = review.get('text', '') or review.get('review_text', '')
            if text and len(text.strip()) > 10:  # Only meaningful reviews
                review_texts.append(text.strip())
        
        if review_texts:
            print(f"📊 Embedding {len(review_texts)} reviews...")
            self.review_embeddings = self.embeddings_model.encode(review_texts, normalize_embeddings=True, show_progress_bar=True)
            print("✅ Reviews embedded successfully")
        else:
            print("⚠️ No valid review texts found")
    
    def search_relevant_reviews(self, query: str, top_k: int = 10) -> List[ReviewContext]:
        """
        Search for reviews relevant to the query.
        
        Args:
            query: User's question or search term
            top_k: Number of top results to return
            
        Returns:
            List of relevant review contexts
        """
        if not RAG_AVAILABLE or self.review_embeddings is None:
            return self._simple_text_search(query, top_k)
        
        # Encode the query
        query_embedding = self.embeddings_model.encode([query], normalize_embeddings=True)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.review_embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build context objects
        contexts = []
        for idx in top_indices:
            if idx < len(self.reviews_data):
                review = self.reviews_data[idx]
                text = review.get('text', '') or review.get('review_text', '')
                sentiment = review.get('sentiment_score', 0.0)
                asin = review.get('parent_asin', 'Unknown')
                rating = review.get('rating', 0)
                
                contexts.append(ReviewContext(
                    text=text,
                    sentiment=sentiment,
                    asin=asin,
                    rating=rating,
                    relevance_score=float(similarities[idx])
                ))
        
        return contexts
    
    def _simple_text_search(self, query: str, top_k: int = 5) -> List[ReviewContext]:
        """Fallback text search when RAG is not available."""
        query_lower = query.lower()
        scored_reviews = []
        
        for review in self.reviews_data:
            text = review.get('text', '') or review.get('review_text', '')
            if not text:
                continue
            
            # Simple keyword matching
            text_lower = text.lower()
            score = 0
            
            # Count query word matches
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 2:  # Skip short words
                    score += text_lower.count(word)
            
            if score > 0:
                scored_reviews.append((score, review))
        
        # Sort by score and take top-k
        scored_reviews.sort(key=lambda x: x[0], reverse=True)
        
        contexts = []
        for score, review in scored_reviews[:top_k]:
            text = review.get('text', '') or review.get('review_text', '')
            sentiment = review.get('sentiment_score', 0.0)
            asin = review.get('parent_asin', 'Unknown')
            rating = review.get('rating', 0)
            
            contexts.append(ReviewContext(
                text=text,
                sentiment=sentiment,
                asin=asin,
                rating=rating,
                relevance_score=float(score)
            ))
        
        return contexts
    
    def generate_insight(self, query: str, contexts: List[ReviewContext]) -> str:
        """
        Generate an insight based on retrieved contexts.
        
        Args:
            query: User's question
            contexts: Retrieved review contexts
            
        Returns:
            Generated insight text
        """
        if not contexts:
            return "I couldn't find relevant information to answer your question. Try asking about specific product features like 'battery life', 'design', 'quality', or 'value for money'."
        
        # Extract ASIN from query if mentioned
        asin_pattern = r'B[A-Z0-9]{9}'
        mentioned_asins = re.findall(asin_pattern, query)
        
        # Filter contexts by ASIN if mentioned
        if mentioned_asins:
            contexts = [ctx for ctx in contexts if ctx.asin in mentioned_asins]
            if not contexts:
                return f"No specific reviews found for product {mentioned_asins[0]}. Try asking about general product features."
        
        # Analyze the contexts
        sentiments = [ctx.sentiment for ctx in contexts]
        ratings = [ctx.rating for ctx in contexts if ctx.rating > 0]
        
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        # Generate insight based on patterns
        insight_parts = []
        
        # More specific sentiment analysis
        positive_count = sum(1 for s in sentiments if s > 0.2)
        negative_count = sum(1 for s in sentiments if s < -0.2)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        if mentioned_asins:
            insight_parts.append(f"For product {mentioned_asins[0]}:")
        
        # Detailed sentiment breakdown
        if positive_count > negative_count:
            if positive_count == len(sentiments):
                insight_parts.append("All customers are very positive about this.")
            else:
                insight_parts.append(f"Most customers are positive ({positive_count}/{len(sentiments)} reviews positive).")
        elif negative_count > positive_count:
            if negative_count == len(sentiments):
                insight_parts.append("All customers have concerns about this.")
            else:
                insight_parts.append(f"Many customers have concerns ({negative_count}/{len(sentiments)} reviews negative).")
        else:
            insight_parts.append(f"Customer opinions are mixed ({positive_count} positive, {negative_count} negative, {neutral_count} neutral).")
        
        # Rating analysis with more context
        if avg_rating >= 4.5:
            insight_parts.append(f"Excellent average rating of {avg_rating:.1f}/5 stars.")
        elif avg_rating >= 4.0:
            insight_parts.append(f"Good average rating of {avg_rating:.1f}/5 stars.")
        elif avg_rating >= 3.0:
            insight_parts.append(f"Average rating of {avg_rating:.1f}/5 stars.")
        elif avg_rating >= 2.0:
            insight_parts.append(f"Below average rating of {avg_rating:.1f}/5 stars.")
        else:
            insight_parts.append(f"Poor average rating of {avg_rating:.1f}/5 stars.")
        
        # Extract specific themes based on query
        query_lower = query.lower()
        specific_themes = []
        
        if 'quality' in query_lower:
            quality_reviews = [ctx for ctx in contexts if 'quality' in ctx.text.lower()]
            if quality_reviews:
                quality_sentiment = sum(ctx.sentiment for ctx in quality_reviews) / len(quality_reviews)
                if quality_sentiment > 0.3:
                    specific_themes.append("quality is praised")
                elif quality_sentiment < -0.3:
                    specific_themes.append("quality concerns raised")
                else:
                    specific_themes.append("quality opinions are mixed")
        
        if 'design' in query_lower:
            design_reviews = [ctx for ctx in contexts if any(word in ctx.text.lower() for word in ['design', 'look', 'appearance', 'style'])]
            if design_reviews:
                design_sentiment = sum(ctx.sentiment for ctx in design_reviews) / len(design_reviews)
                if design_sentiment > 0.3:
                    specific_themes.append("design is well-received")
                elif design_sentiment < -0.3:
                    specific_themes.append("design issues mentioned")
                else:
                    specific_themes.append("design feedback is mixed")
        
        if 'battery' in query_lower:
            battery_reviews = [ctx for ctx in contexts if 'battery' in ctx.text.lower()]
            if battery_reviews:
                battery_sentiment = sum(ctx.sentiment for ctx in battery_reviews) / len(battery_reviews)
                if battery_sentiment > 0.3:
                    specific_themes.append("battery life is praised")
                elif battery_sentiment < -0.3:
                    specific_themes.append("battery life concerns")
                else:
                    specific_themes.append("battery life feedback is mixed")
        
        if 'size' in query_lower:
            size_reviews = [ctx for ctx in contexts if any(word in ctx.text.lower() for word in ['size', 'big', 'small', 'large', 'compact'])]
            if size_reviews:
                size_sentiment = sum(ctx.sentiment for ctx in size_reviews) / len(size_reviews)
                if size_sentiment > 0.3:
                    specific_themes.append("size is appreciated")
                elif size_sentiment < -0.3:
                    specific_themes.append("size issues mentioned")
                else:
                    specific_themes.append("size feedback is mixed")
        
        if specific_themes:
            insight_parts.append(f"Regarding your question: {', '.join(specific_themes)}.")
        
        # Key themes from reviews
        common_words = self._extract_common_themes(contexts)
        if common_words:
            insight_parts.append(f"Common themes: {', '.join(common_words[:3])}")
        
        # Sample review quote with better selection
        if contexts:
            # Select the most relevant review
            best_context = max(contexts, key=lambda x: x.relevance_score)
            quote = best_context.text[:150] + "..." if len(best_context.text) > 150 else best_context.text
            insight_parts.append(f"Example review: \"{quote}\"")
        
        return " ".join(insight_parts)
    
    def _extract_common_themes(self, contexts: List[ReviewContext]) -> List[str]:
        """Extract common themes from review contexts."""
        all_text = " ".join([ctx.text for ctx in contexts])
        
        # Simple keyword extraction (could be enhanced with NLP)
        words = re.findall(r'\b\w+\b', all_text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        filtered_words = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Count word frequency
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return most common words
        return sorted(word_counts.keys(), key=lambda x: word_counts[x], reverse=True)[:5]
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Main query interface for RAG system.
        
        Args:
            question: User's question
            top_k: Number of relevant reviews to retrieve
            
        Returns:
            Dictionary with answer and supporting evidence
        """
        # Search for relevant reviews
        contexts = self.search_relevant_reviews(question, top_k)
        
        # Generate insight
        insight = self.generate_insight(question, contexts)
        
        return {
            'question': question,
            'answer': insight,
            'supporting_reviews': [
                {
                    'text': ctx.text[:300] + "..." if len(ctx.text) > 300 else ctx.text,
                    'sentiment': ctx.sentiment,
                    'rating': ctx.rating,
                    'asin': ctx.asin,
                    'relevance_score': ctx.relevance_score
                }
                for ctx in contexts
            ],
            'rag_available': RAG_AVAILABLE
        }


# Example usage
if __name__ == "__main__":
    # Example usage
    rag = RAGSystem()
    
    # Load sample data
    sample_reviews = [
        {
            'text': 'Great product, excellent quality and fast delivery!',
            'sentiment_score': 0.8,
            'parent_asin': 'B123456',
            'rating': 5
        },
        {
            'text': 'Poor quality, broke after one week of use.',
            'sentiment_score': -0.7,
            'parent_asin': 'B123456',
            'rating': 2
        }
    ]
    
    rag.load_reviews(sample_reviews)
    
    # Query the system
    result = rag.query("What do customers say about the quality?")
    print("Question:", result['question'])
    print("Answer:", result['answer'])
    print("RAG Available:", result['rag_available'])
