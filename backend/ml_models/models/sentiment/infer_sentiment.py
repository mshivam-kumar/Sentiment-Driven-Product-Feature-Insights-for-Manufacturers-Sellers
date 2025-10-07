"""
Baseline Sentiment Analysis for Product Features

This module implements sentence-level sentiment analysis using pre-trained models
and maps sentiment scores to extracted aspects.
"""

import json
import re
from typing import Dict, List, Tuple
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


class SentimentAnalyzer:
    """Sentiment analyzer for product features."""
    
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: Hugging Face model name for sentiment analysis
        """
        self.model_name = model_name
        
        # Load pre-trained sentiment model
        try:
            # Use a widely available English SST-2 model
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                top_k=None  # replaces return_all_scores=True
            )
        except Exception as e:
            print(f"Failed to load sentiment model: {e}")
            # Fallback to a robust open model
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                top_k=None
            )
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using simple regex.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if text is None:
            return []
        if not isinstance(text, str):
            text = str(text)
        # Capture sentences including optional terminal punctuation
        matches = re.findall(r"[^.!?]+[.!?]?", text)
        cleaned = []
        for m in matches:
            s = m.strip()
            if not s:
                continue
            # Remove trailing period only; keep '?' and '!' as tests expect
            if s.endswith('.'):
                s = s[:-1]
            cleaned.append(s)
        return cleaned
    
    def analyze_sentence_sentiment(self, sentence: str) -> float:
        """
        Analyze sentiment of a single sentence.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Sentiment score between -1 (negative) and +1 (positive)
        """
        if not sentence.strip():
            return 0.0
        
        try:
            outputs = self.sentiment_pipeline(sentence)
            # outputs is a list with one element (because one input), which is a list of {label, score}
            if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], list):
                scores = outputs[0]
            else:
                scores = outputs
            positive_score = 0.0
            negative_score = 0.0
            # Normalize labels and pick scores
            for item in scores:
                label = str(item.get("label", "")).upper()
                score = float(item.get("score", 0.0))
                if label in ("POSITIVE", "LABEL_2"):  # common mappings
                    positive_score = max(positive_score, score)
                elif label in ("NEGATIVE", "LABEL_0"):
                    negative_score = max(negative_score, score)
                elif label in ("NEUTRAL", "LABEL_1"):
                    # ignore neutral for signed score, treat as 0
                    pass
            # Neutral keyword heuristic
            lower_text = sentence.lower()
            if re.search(r"\b(ok|okay|fine|average|decent|meh|not bad)\b", lower_text):
                return 0.0
            if positive_score > negative_score:
                return positive_score
            if negative_score > positive_score:
                return -negative_score
            return 0.0
        except Exception as e:
            print(f"Error analyzing sentiment for '{sentence}': {e}")
            return 0.0
    
    def map_sentiment_to_aspects(self, text: str, aspects: Dict[str, float]) -> Dict[str, Dict]:
        """
        Map sentiment scores to extracted aspects.
        
        Args:
            text: Original review text
            aspects: Dictionary of extracted aspects and their confidence scores
            
        Returns:
            Dictionary mapping aspects to sentiment information
        """
        sentences = self.split_into_sentences(text)
        aspect_sentiments = {}
        
        for aspect, confidence in aspects.items():
            # Find sentences that mention this aspect
            relevant_sentences = []
            for sentence in sentences:
                # Simple keyword matching (can be improved with more sophisticated NLP)
                if any(keyword in sentence.lower() for keyword in aspect.split('_')):
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                # Calculate average sentiment for relevant sentences
                sentiments = [self.analyze_sentence_sentiment(s) for s in relevant_sentences]
                avg_sentiment = np.mean(sentiments) if sentiments else 0.0
                
                # Find the most relevant sentence
                best_sentence = max(relevant_sentences, 
                                  key=lambda s: self.analyze_sentence_sentiment(s))
                
                aspect_sentiments[aspect] = {
                    "score": float(avg_sentiment),
                    "sentence": best_sentence,
                    "confidence": float(confidence),
                    "relevant_sentences": relevant_sentences
                }
            else:
                # If no specific sentences found, use overall text sentiment
                overall_sentiment = self.analyze_sentence_sentiment(text if isinstance(text, str) else "")
                aspect_sentiments[aspect] = {
                    "score": float(overall_sentiment),
                    "sentence": (text[:100] + "..." if isinstance(text, str) and len(text) > 100 else (text or "")),
                    "confidence": float(confidence),
                    "relevant_sentences": [text] if isinstance(text, str) else []
                }
        
        return aspect_sentiments
    
    def process_review_with_aspects(self, review_text: str, aspects: Dict[str, float], 
                                  review_id: str = None, asin: str = None) -> Dict:
        """
        Process a review with pre-extracted aspects and return sentiment mapping.
        
        Args:
            review_text: The review text
            aspects: Dictionary of extracted aspects
            review_id: Optional review ID
            asin: Optional product ASIN
            
        Returns:
            Dictionary with aspect-sentiment mapping
        """
        # Guard against None review_text
        safe_text = review_text if isinstance(review_text, str) else ""
        aspect_sentiments = self.map_sentiment_to_aspects(safe_text, aspects)
        
        return {
            "review_id": review_id,
            "asin": asin,
            "text": safe_text,
            "aspects": aspect_sentiments,
            "overall_sentiment": self.analyze_sentence_sentiment(safe_text)
        }


def main():
    """Main function for testing the sentiment analyzer."""
    analyzer = SentimentAnalyzer()
    
    # Test with sample data
    sample_data = [
        {
            "text": "Battery life is great but the camera is blurry.",
            "aspects": {"battery_life": 0.8, "camera_quality": 0.7}
        },
        {
            "text": "The screen quality is excellent and the design is beautiful.",
            "aspects": {"display_quality": 0.9, "design": 0.8}
        },
        {
            "text": "Fast shipping and good packaging, but the price is too high.",
            "aspects": {"delivery": 0.6, "packaging": 0.7, "value_for_money": 0.8}
        }
    ]
    
    print("=== Sentiment Analysis Results ===")
    for i, data in enumerate(sample_data):
        result = analyzer.process_review_with_aspects(
            data["text"], 
            data["aspects"], 
            f"review_{i}"
        )
        
        print(f"\nReview {i+1}: {data['text']}")
        print(f"Overall sentiment: {result['overall_sentiment']:.2f}")
        print("Aspect sentiments:")
        for aspect, info in result["aspects"].items():
            print(f"  {aspect}: {info['score']:.2f} - '{info['sentence']}'")


if __name__ == "__main__":
    main()
