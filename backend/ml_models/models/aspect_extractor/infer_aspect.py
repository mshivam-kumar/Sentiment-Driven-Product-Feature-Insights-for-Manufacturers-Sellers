"""
Baseline Aspect Extraction using spaCy + (optional) YAKE

This module implements a fast, interpretable baseline for extracting product aspects
from review text using spaCy for noun phrase extraction and YAKE for keyword scoring.
If YAKE is not available in the runtime environment, the extractor gracefully
falls back to noun-phrase-only scoring so inference Lambdas can run without
bundling heavy dependencies.
"""

try:
    import yake  # type: ignore
    _YAKE_AVAILABLE = True
except Exception as _e:
    print(f"YAKE not available, falling back to noun-phrase-only extraction: {_e}")
    yake = None  # type: ignore
    _YAKE_AVAILABLE = False
import json
import re
from typing import List, Dict, Set
from collections import Counter


class AspectExtractor:
    """Baseline aspect extractor using spaCy and YAKE."""
    
    def __init__(self, language="en", max_ngram_size=3, deduplication_threshold=0.7):
        """
        Initialize the aspect extractor.
        
        Args:
            language: Language for YAKE keyword extraction
            max_ngram_size: Maximum n-gram size for YAKE
            deduplication_threshold: Threshold for YAKE deduplication
        """
        # Lazily load spaCy to avoid hard dependency issues in constrained envs
        self.nlp = None
        try:
            import spacy  # type: ignore
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                print(f"spaCy model load failed, falling back to regex noun phrase extraction: {e}")
                self.nlp = None
        except Exception as e:
            print(f"spaCy not available, falling back to regex noun phrase extraction: {e}")
        
        # Initialize YAKE if available
        self.yake_extractor = None
        if _YAKE_AVAILABLE:
            try:
                self.yake_extractor = yake.KeywordExtractor(
                    lan=language,
                    n=max_ngram_size,
                    dedupLim=deduplication_threshold,
                    top=20
                )
            except Exception as e:
                print(f"YAKE initialization failed, continuing without YAKE: {e}")
                self.yake_extractor = None
        
        # Canonicalization dictionary for synonyms
        self.canonicalization_dict = {
            'battery': 'battery_life',
            'battery life': 'battery_life',
            'battery performance': 'battery_life',
            'camera': 'camera_quality',
            'camera quality': 'camera_quality',
            'picture quality': 'camera_quality',
            'photo quality': 'camera_quality',
            'screen': 'display_quality',
            'display': 'display_quality',
            'screen quality': 'display_quality',
            'display quality': 'display_quality',
            'price': 'value_for_money',
            'cost': 'value_for_money',
            'value': 'value_for_money',
            'shipping': 'delivery',
            'delivery': 'delivery',
            'packaging': 'packaging',
            'box': 'packaging',
            'size': 'size_fit',
            'fit': 'size_fit',
            'comfort': 'comfort',
            'ease of use': 'usability',
            'usability': 'usability',
            'user interface': 'usability',
            'ui': 'usability',
            'performance': 'performance',
            'speed': 'performance',
            'quality': 'build_quality',
            'build quality': 'build_quality',
            'durability': 'build_quality',
            'design': 'design',
            'looks': 'design',
            'appearance': 'design',
            'customer service': 'customer_service',
            'support': 'customer_service',
            'warranty': 'warranty',
            'return policy': 'return_policy',
            'returns': 'return_policy'
        }
    
    def extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases using spaCy.
        
        Args:
            text: Input text to process
            
        Returns:
            List of noun phrases
        """
        noun_phrases: List[str] = []
        if self.nlp is not None:
            doc = self.nlp(text)
            for chunk in doc.noun_chunks:
                phrase = chunk.text.strip().lower()
                phrase = re.sub(r'\b(the|a|an|this|that|these|those)\b', '', phrase).strip()
                if len(phrase) > 2:
                    noun_phrases.append(phrase)
            return noun_phrases
        # Fallback: more selective regex to capture meaningful product aspects
        # Look for common product-related patterns
        product_patterns = [
            r'\b(battery\s+life|battery\s+performance|battery\s+power)\b',
            r'\b(camera\s+quality|camera\s+performance|camera\s+resolution)\b',
            r'\b(design\s+quality|build\s+quality|construction\s+quality)\b',
            r'\b(sound\s+quality|audio\s+quality|sound\s+performance)\b',
            r'\b(display\s+quality|screen\s+quality|display\s+resolution)\b',
            r'\b(performance|speed|efficiency|reliability)\b',
            r'\b(comfort|comfortable|ergonomics|fit)\b',
            r'\b(durability|longevity|sturdiness|robustness)\b',
            r'\b(ease\s+of\s+use|usability|user\s+experience)\b',
            r'\b(value\s+for\s+money|value|price\s+performance)\b',
            r'\b(customer\s+service|support|warranty)\b',
            r'\b(shipping|delivery|packaging)\b',
            r'\b(quality|material|finish|texture)\b',
            r'\b(size|weight|dimensions|portability)\b',
            r'\b(style|appearance|look|aesthetics)\b'
        ]
        
        for pattern in product_patterns:
            matches = re.findall(pattern, text.lower())
            noun_phrases.extend(matches)
        
        # Also look for simple meaningful single words
        meaningful_words = [
            'battery', 'camera', 'design', 'quality', 'performance', 'sound', 
            'display', 'screen', 'comfort', 'durability', 'usability', 'value',
            'service', 'shipping', 'material', 'size', 'weight', 'style',
            'price', 'cost', 'warranty', 'support', 'delivery', 'packaging'
        ]
        
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            if word in meaningful_words and len(word) > 3:
                noun_phrases.append(word)
        
        return list(dict.fromkeys(noun_phrases))
    
    def extract_keywords(self, text: str) -> List[tuple]:
        """
        Extract keywords using YAKE.
        
        Args:
            text: Input text to process
            
        Returns:
            List of (keyword, score) tuples
        """
        if self.yake_extractor is None:
            return []
        keywords = self.yake_extractor.extract_keywords(text)
        return keywords
    
    def canonicalize_aspect(self, aspect: str) -> str:
        """
        Canonicalize aspect names using the synonym dictionary.
        
        Args:
            aspect: Raw aspect name
            
        Returns:
            Canonicalized aspect name
        """
        aspect_lower = aspect.lower().strip()
        return self.canonicalization_dict.get(aspect_lower, aspect_lower)
    
    def extract_aspects(self, text: str) -> Dict[str, float]:
        """
        Extract aspects from text using both spaCy and YAKE.
        
        Args:
            text: Input review text
            
        Returns:
            Dictionary mapping aspect names to confidence scores
        """
        # Extract noun phrases
        noun_phrases = self.extract_noun_phrases(text)
        
        # Extract keywords with YAKE
        yake_keywords = self.extract_keywords(text)
        
        # Combine and score aspects
        aspect_scores = {}
        
        # Score noun phrases (higher weight for spaCy results)
        for phrase in noun_phrases:
            canonical = self.canonicalize_aspect(phrase)
            aspect_scores[canonical] = aspect_scores.get(canonical, 0) + 0.8
        
        # Score YAKE keywords
        for keyword, score in yake_keywords:
            canonical = self.canonicalize_aspect(keyword)
            aspect_scores[canonical] = aspect_scores.get(canonical, 0) + score * 0.2
        
        # Filter out aspects with very low scores and meaningless aspects
        filtered_aspects = {}
        for k, v in aspect_scores.items():
            # Only keep meaningful aspects
            if (v > 0.3 and 
                len(k) > 3 and 
                not k in ['the', 'and', 'or', 'but', 'for', 'with', 'this', 'that', 'these', 'those'] and
                not re.match(r'^[a-z]+\s+[a-z]+$', k) or  # Allow meaningful bigrams
                k in ['battery', 'camera', 'design', 'quality', 'performance', 'sound', 'display', 'comfort', 'durability', 'value', 'service', 'shipping', 'material', 'size', 'weight', 'style']):
                filtered_aspects[k] = v
        
        # Fallback: if nothing extracted, use a generic 'product' aspect so downstream stays robust
        if not filtered_aspects and text and len(text.split()) >= 3:
            filtered_aspects = {"product": 0.5}
        
        return filtered_aspects
    
    def process_review(self, review_text: str, review_id: str = None) -> Dict:
        """
        Process a single review and extract aspects.
        
        Args:
            review_text: The review text
            review_id: Optional review ID
            
        Returns:
            Dictionary with extracted aspects and metadata
        """
        aspects = self.extract_aspects(review_text)
        
        return {
            "review_id": review_id,
            "text": review_text,
            "aspects": aspects,
            "aspect_count": len(aspects)
        }


def main():
    """Main function for testing the aspect extractor."""
    extractor = AspectExtractor()
    
    # Test with sample reviews
    sample_reviews = [
        "Battery life is great but the camera is blurry.",
        "The screen quality is excellent and the design is beautiful.",
        "Fast shipping and good packaging, but the price is too high.",
        "Easy to use interface, but customer service is terrible.",
        "Great performance and build quality, worth the money."
    ]
    
    print("=== Aspect Extraction Results ===")
    for i, review in enumerate(sample_reviews):
        result = extractor.process_review(review, f"review_{i}")
        print(f"\nReview {i+1}: {review}")
        print(f"Aspects: {result['aspects']}")
        print(f"Count: {result['aspect_count']}")


if __name__ == "__main__":
    main()
