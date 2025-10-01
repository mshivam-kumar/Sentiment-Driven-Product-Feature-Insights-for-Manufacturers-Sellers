"""
Training Data Preparation for Fine-tuning TinyLlama

This script prepares product review data for fine-tuning by creating
conversational training examples in the TinyLlama chat format.
"""

import json
import os
from typing import List, Dict, Any
import random

class TrainingDataPreparer:
    """Prepare training data for fine-tuning."""
    
    def __init__(self):
        self.training_examples = []
        
    def create_training_examples(self, reviews_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Create training examples from review data."""
        print(f"📊 Processing {len(reviews_data)} reviews for training...")
        
        training_examples = []
        
        for review in reviews_data:
            text = review.get('text', '') or review.get('review_text', '')
            sentiment = review.get('sentiment_score', 0.0)
            rating = review.get('rating', 0)
            asin = review.get('parent_asin', 'Unknown')
            
            if not text or len(text.strip()) < 20:
                continue
                
            # Create multiple training examples per review
            examples = self._create_examples_for_review(text, sentiment, rating, asin)
            training_examples.extend(examples)
        
        print(f"✅ Created {len(training_examples)} training examples")
        return training_examples
    
    def _create_examples_for_review(self, text: str, sentiment: float, rating: int, asin: str) -> List[Dict[str, str]]:
        """Create multiple training examples for a single review."""
        examples = []
        
        # Clean and truncate text
        clean_text = text.strip()[:300]  # Limit length
        
        # Example 1: Sentiment Analysis
        sentiment_label = self._get_sentiment_label(sentiment)
        examples.append({
            "text": f"<|user|>\nAnalyze the sentiment of this product review: {clean_text}\n<|assistant|>\nThis review shows a {sentiment_label} sentiment with a score of {sentiment:.2f}. The customer rated the product {rating}/5 stars."
        })
        
        # Example 2: Quality Assessment
        if 'quality' in text.lower() or 'good' in text.lower() or 'bad' in text.lower():
            quality_sentiment = "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"
            examples.append({
                "text": f"<|user|>\nWhat does this customer say about product quality: {clean_text}\n<|assistant|>\nThe customer's feedback on quality is {quality_sentiment}. They rated it {rating}/5 stars and the sentiment score is {sentiment:.2f}."
            })
        
        # Example 3: Recommendation
        if rating >= 4:
            examples.append({
                "text": f"<|user|>\nShould I buy this product based on this review: {clean_text}\n<|assistant|>\nYes, this product is recommended. The customer gave it {rating}/5 stars with a {sentiment_label} sentiment ({sentiment:.2f})."
            })
        elif rating <= 2:
            examples.append({
                "text": f"<|user|>\nShould I buy this product based on this review: {clean_text}\n<|assistant|>\nNo, this product is not recommended. The customer gave it only {rating}/5 stars with a {sentiment_label} sentiment ({sentiment:.2f})."
            })
        
        # Example 4: Feature Analysis
        features = self._extract_features(text)
        if features:
            feature_text = ", ".join(features[:3])  # Top 3 features
            examples.append({
                "text": f"<|user|>\nWhat features does this review mention: {clean_text}\n<|assistant|>\nThis review mentions: {feature_text}. The overall sentiment is {sentiment_label} ({sentiment:.2f}) with a {rating}/5 star rating."
            })
        
        # Example 5: Summary
        examples.append({
            "text": f"<|user|>\nSummarize this product review: {clean_text}\n<|assistant|>\nThis is a {sentiment_label} review ({sentiment:.2f}) where the customer rated the product {rating}/5 stars. The review discusses various aspects of the product."
        })
        
        return examples
    
    def _get_sentiment_label(self, sentiment: float) -> str:
        """Convert sentiment score to label."""
        if sentiment > 0.3:
            return "very positive"
        elif sentiment > 0.1:
            return "positive"
        elif sentiment < -0.3:
            return "very negative"
        elif sentiment < -0.1:
            return "negative"
        else:
            return "neutral"
    
    def _extract_features(self, text: str) -> List[str]:
        """Extract mentioned features from review text."""
        features = []
        text_lower = text.lower()
        
        feature_keywords = {
            'quality': ['quality', 'well made', 'durable', 'solid'],
            'design': ['design', 'look', 'appearance', 'style', 'beautiful'],
            'performance': ['performance', 'works', 'function', 'effective'],
            'value': ['value', 'price', 'worth', 'expensive', 'cheap'],
            'delivery': ['delivery', 'shipping', 'fast', 'slow'],
            'customer service': ['service', 'support', 'help', 'response'],
            'size': ['size', 'big', 'small', 'large', 'compact'],
            'battery': ['battery', 'power', 'charge', 'life'],
            'comfort': ['comfortable', 'comfort', 'easy', 'convenient']
        }
        
        for feature, keywords in feature_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                features.append(feature)
        
        return features
    
    def save_training_data(self, training_examples: List[Dict[str, str]], output_file: str = "training_data.json"):
        """Save training data to file."""
        print(f"💾 Saving {len(training_examples)} training examples to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_examples, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Training data saved to {output_file}")
        return output_file
    
    def load_sample_data(self) -> List[Dict[str, Any]]:
        """Load sample review data for training."""
        sample_reviews = [
            {
                'text': 'Excellent product! Great quality and fast delivery. Highly recommend.',
                'sentiment_score': 0.9,
                'parent_asin': 'B08JTNQFZY',
                'rating': 5
            },
            {
                'text': 'Poor quality, broke after one week. Not worth the money.',
                'sentiment_score': -0.8,
                'parent_asin': 'B08JTNQFZY',
                'rating': 2
            },
            {
                'text': 'Good value for money. Works as expected but could be better.',
                'sentiment_score': 0.2,
                'parent_asin': 'B08JTNQFZY',
                'rating': 4
            },
            {
                'text': 'Amazing design and quality! Love the style and it works perfectly.',
                'sentiment_score': 0.7,
                'parent_asin': 'B097YYB2GV',
                'rating': 5
            },
            {
                'text': 'Average product. Nothing special but does the job.',
                'sentiment_score': 0.0,
                'parent_asin': 'B00YQ6X8EO',
                'rating': 3
            },
            {
                'text': 'Terrible customer service. Product is okay but support is awful.',
                'sentiment_score': -0.5,
                'parent_asin': 'B081TJ8YS3',
                'rating': 2
            },
            {
                'text': 'Perfect size and very comfortable to use. Great battery life too!',
                'sentiment_score': 0.8,
                'parent_asin': 'B08BZ63GMJ',
                'rating': 5
            },
            {
                'text': 'Too expensive for what you get. Quality is decent but overpriced.',
                'sentiment_score': -0.3,
                'parent_asin': 'B00R8DXL44',
                'rating': 3
            }
        ]
        
        return sample_reviews


def main():
    """Main function to prepare training data."""
    print("🎯 Preparing Training Data for TinyLlama Fine-tuning")
    print("=" * 60)
    
    # Initialize preparer
    preparer = TrainingDataPreparer()
    
    # Load sample data (replace with your actual data)
    reviews_data = preparer.load_sample_data()
    
    # Create training examples
    training_examples = preparer.create_training_examples(reviews_data)
    
    # Save training data
    output_file = preparer.save_training_data(training_examples)
    
    print(f"\n🎉 Training data preparation complete!")
    print(f"📁 Training examples saved to: {output_file}")
    print(f"📊 Total examples: {len(training_examples)}")
    print(f"🚀 Ready for fine-tuning!")


if __name__ == "__main__":
    main()
