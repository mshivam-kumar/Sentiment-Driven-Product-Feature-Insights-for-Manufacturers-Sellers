"""
Unit tests for aspect extraction functionality.
"""

import pytest
import sys
import os

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models', 'aspect_extractor'))

from infer_aspect import AspectExtractor


class TestAspectExtraction:
    """Test class for aspect extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = AspectExtractor()
    
    def test_extract_noun_phrases(self):
        """Test noun phrase extraction."""
        text = "Battery life is great but the camera is blurry."
        phrases = self.extractor.extract_noun_phrases(text)
        
        assert len(phrases) > 0
        assert "battery life" in phrases or "camera" in phrases
    
    def test_extract_keywords(self):
        """Test keyword extraction with YAKE."""
        text = "The screen quality is excellent and the design is beautiful."
        keywords = self.extractor.extract_keywords(text)
        
        assert len(keywords) > 0
        assert isinstance(keywords, list)
        assert all(isinstance(kw, tuple) and len(kw) == 2 for kw in keywords)
    
    def test_canonicalize_aspect(self):
        """Test aspect canonicalization."""
        # Test known mappings
        assert self.extractor.canonicalize_aspect("battery") == "battery_life"
        assert self.extractor.canonicalize_aspect("camera") == "camera_quality"
        assert self.extractor.canonicalize_aspect("screen") == "display_quality"
        
        # Test unknown aspect
        assert self.extractor.canonicalize_aspect("unknown_feature") == "unknown_feature"
    
    def test_extract_aspects(self):
        """Test complete aspect extraction."""
        text = "Battery life is great but the camera is blurry."
        aspects = self.extractor.extract_aspects(text)
        
        assert isinstance(aspects, dict)
        assert len(aspects) > 0
        assert all(isinstance(score, (int, float)) for score in aspects.values())
    
    def test_process_review(self):
        """Test complete review processing."""
        review_text = "The screen quality is excellent and the design is beautiful."
        result = self.extractor.process_review(review_text, "test_review_1")
        
        assert isinstance(result, dict)
        assert "review_id" in result
        assert "text" in result
        assert "aspects" in result
        assert "aspect_count" in result
        assert result["review_id"] == "test_review_1"
        assert result["text"] == review_text
        assert isinstance(result["aspects"], dict)
        assert isinstance(result["aspect_count"], int)
    
    def test_empty_text(self):
        """Test handling of empty text."""
        result = self.extractor.process_review("", "empty_review")
        
        assert result["aspects"] == {}
        assert result["aspect_count"] == 0
    
    def test_short_text(self):
        """Test handling of very short text."""
        result = self.extractor.process_review("Good", "short_review")
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert "aspects" in result
    
    def test_special_characters(self):
        """Test handling of special characters."""
        text = "The product's battery-life is great! (5/5 stars)"
        result = self.extractor.process_review(text, "special_chars")
        
        assert isinstance(result, dict)
        assert "aspects" in result


class TestAspectExtractionIntegration:
    """Integration tests for aspect extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = AspectExtractor()
    
    def test_sample_reviews(self):
        """Test with sample review data."""
        sample_reviews = [
            "Battery life is great but the camera is blurry.",
            "The screen quality is excellent and the design is beautiful.",
            "Fast shipping and good packaging, but the price is too high.",
            "Easy to use interface, but customer service is terrible.",
            "Great performance and build quality, worth the money."
        ]
        
        for i, review in enumerate(sample_reviews):
            result = self.extractor.process_review(review, f"review_{i}")
            
            assert result["aspects"] != {}, f"No aspects found for review: {review}"
            assert result["aspect_count"] > 0, f"No aspects counted for review: {review}"
    
    def test_aspect_consistency(self):
        """Test that similar reviews produce consistent aspects."""
        review1 = "Battery life is excellent"
        review2 = "The battery performance is great"
        
        result1 = self.extractor.process_review(review1, "review_1")
        result2 = self.extractor.process_review(review2, "review_2")
        
        # Both should extract battery-related aspects
        aspects1 = set(result1["aspects"].keys())
        aspects2 = set(result2["aspects"].keys())
        
        # Should have some overlap in battery-related aspects
        battery_aspects1 = {a for a in aspects1 if "battery" in a}
        battery_aspects2 = {a for a in aspects2 if "battery" in a}
        
        assert len(battery_aspects1) > 0, "No battery aspects found in first review"
        assert len(battery_aspects2) > 0, "No battery aspects found in second review"
    
    def test_performance(self):
        """Test performance with multiple reviews."""
        import time
        
        reviews = [
            "Battery life is great but the camera is blurry.",
            "The screen quality is excellent and the design is beautiful.",
            "Fast shipping and good packaging, but the price is too high.",
            "Easy to use interface, but customer service is terrible.",
            "Great performance and build quality, worth the money."
        ] * 10  # 50 reviews total
        
        start_time = time.time()
        
        for i, review in enumerate(reviews):
            self.extractor.process_review(review, f"perf_review_{i}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 50 reviews in less than 10 seconds
        assert processing_time < 10, f"Processing took too long: {processing_time:.2f} seconds"
        
        # Average processing time should be less than 0.2 seconds per review
        avg_time = processing_time / len(reviews)
        assert avg_time < 0.2, f"Average processing time too slow: {avg_time:.3f} seconds per review"


if __name__ == "__main__":
    pytest.main([__file__])
