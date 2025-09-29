"""
Unit tests for sentiment analysis functionality.
"""

import pytest
import sys
import os

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models', 'sentiment'))

from infer_sentiment import SentimentAnalyzer


class TestSentimentAnalysis:
    """Test class for sentiment analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    def test_split_into_sentences(self):
        """Test sentence splitting."""
        text = "Battery life is great. The camera is blurry. Overall, it's a good product!"
        sentences = self.analyzer.split_into_sentences(text)
        
        assert len(sentences) == 3
        assert "Battery life is great" in sentences
        assert "The camera is blurry" in sentences
        assert "Overall, it's a good product!" in sentences
    
    def test_analyze_sentence_sentiment(self):
        """Test sentence sentiment analysis."""
        # Positive sentence
        pos_score = self.analyzer.analyze_sentence_sentiment("This is excellent!")
        assert pos_score > 0, f"Positive sentence should have positive score, got {pos_score}"
        
        # Negative sentence
        neg_score = self.analyzer.analyze_sentence_sentiment("This is terrible!")
        assert neg_score < 0, f"Negative sentence should have negative score, got {neg_score}"
        
        # Neutral sentence
        neu_score = self.analyzer.analyze_sentence_sentiment("This is okay.")
        assert -0.5 < neu_score < 0.5, f"Neutral sentence should have neutral score, got {neu_score}"
    
    def test_map_sentiment_to_aspects(self):
        """Test sentiment mapping to aspects."""
        text = "Battery life is great but the camera is blurry."
        aspects = {"battery_life": 0.8, "camera_quality": 0.7}
        
        result = self.analyzer.map_sentiment_to_aspects(text, aspects)
        
        assert isinstance(result, dict)
        assert "battery_life" in result
        assert "camera_quality" in result
        
        # Check structure of aspect sentiment data
        for aspect, sentiment_info in result.items():
            assert "score" in sentiment_info
            assert "sentence" in sentiment_info
            assert "confidence" in sentiment_info
            assert "relevant_sentences" in sentiment_info
            assert isinstance(sentiment_info["score"], (int, float))
            assert isinstance(sentiment_info["sentence"], str)
            assert isinstance(sentiment_info["confidence"], (int, float))
            assert isinstance(sentiment_info["relevant_sentences"], list)
    
    def test_process_review_with_aspects(self):
        """Test complete review processing with aspects."""
        review_text = "Battery life is great but the camera is blurry."
        aspects = {"battery_life": 0.8, "camera_quality": 0.7}
        
        result = self.analyzer.process_review_with_aspects(
            review_text, aspects, "test_review_1", "B00YQ6X8EO"
        )
        
        assert isinstance(result, dict)
        assert result["review_id"] == "test_review_1"
        assert result["asin"] == "B00YQ6X8EO"
        assert result["text"] == review_text
        assert "aspects" in result
        assert "overall_sentiment" in result
        assert isinstance(result["overall_sentiment"], (int, float))
        
        # Check aspect sentiment structure
        for aspect, sentiment_info in result["aspects"].items():
            assert "score" in sentiment_info
            assert "sentence" in sentiment_info
            assert "confidence" in sentiment_info
            assert "relevant_sentences" in sentiment_info
    
    def test_empty_text(self):
        """Test handling of empty text."""
        result = self.analyzer.process_review_with_aspects("", {}, "empty_review")
        
        assert result["aspects"] == {}
        assert result["overall_sentiment"] == 0.0
    
    def test_no_aspects(self):
        """Test handling of reviews with no aspects."""
        text = "This is a general comment."
        aspects = {}
        
        result = self.analyzer.process_review_with_aspects(text, aspects, "no_aspects")
        
        assert result["aspects"] == {}
        assert isinstance(result["overall_sentiment"], (int, float))
    
    def test_sentiment_score_range(self):
        """Test that sentiment scores are in expected range."""
        test_cases = [
            ("This is excellent!", 0.5, 1.0),
            ("This is terrible!", -1.0, -0.5),
            ("This is okay.", -0.5, 0.5),
            ("I love this product!", 0.5, 1.0),
            ("I hate this product!", -1.0, -0.5)
        ]
        
        for text, min_score, max_score in test_cases:
            score = self.analyzer.analyze_sentence_sentiment(text)
            assert min_score <= score <= max_score, \
                f"Score {score} for '{text}' not in expected range [{min_score}, {max_score}]"


class TestSentimentAnalysisIntegration:
    """Integration tests for sentiment analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    def test_sample_data(self):
        """Test with sample review data."""
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
        
        for i, data in enumerate(sample_data):
            result = self.analyzer.process_review_with_aspects(
                data["text"], data["aspects"], f"sample_{i}"
            )
            
            assert len(result["aspects"]) > 0, f"No aspect sentiments found for sample {i}"
            
            # Check that each aspect has sentiment data
            for aspect in data["aspects"]:
                assert aspect in result["aspects"], f"Aspect {aspect} not found in result"
                sentiment_info = result["aspects"][aspect]
                assert "score" in sentiment_info
                assert "sentence" in sentiment_info
                assert "confidence" in sentiment_info
    
    def test_sentiment_consistency(self):
        """Test that similar reviews produce consistent sentiment."""
        review1 = "Battery life is excellent"
        review2 = "The battery performance is great"
        aspects = {"battery_life": 0.8}
        
        result1 = self.analyzer.process_review_with_aspects(review1, aspects, "review_1")
        result2 = self.analyzer.process_review_with_aspects(review2, aspects, "review_2")
        
        # Both should have positive sentiment for battery_life
        if "battery_life" in result1["aspects"] and "battery_life" in result2["aspects"]:
            score1 = result1["aspects"]["battery_life"]["score"]
            score2 = result2["aspects"]["battery_life"]["score"]
            
            # Both should be positive (or at least not negative)
            assert score1 > -0.5, f"Battery sentiment too negative in review 1: {score1}"
            assert score2 > -0.5, f"Battery sentiment too negative in review 2: {score2}"
    
    def test_performance(self):
        """Test performance with multiple reviews."""
        import time
        
        reviews = [
            ("Battery life is great but the camera is blurry.", {"battery_life": 0.8, "camera_quality": 0.7}),
            ("The screen quality is excellent and the design is beautiful.", {"display_quality": 0.9, "design": 0.8}),
            ("Fast shipping and good packaging, but the price is too high.", {"delivery": 0.6, "packaging": 0.7, "value_for_money": 0.8}),
            ("Easy to use interface, but customer service is terrible.", {"usability": 0.7, "customer_service": 0.8}),
            ("Great performance and build quality, worth the money.", {"performance": 0.8, "build_quality": 0.7, "value_for_money": 0.9})
        ] * 10  # 50 reviews total
        
        start_time = time.time()
        
        for i, (text, aspects) in enumerate(reviews):
            self.analyzer.process_review_with_aspects(text, aspects, f"perf_review_{i}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 50 reviews in less than 30 seconds
        assert processing_time < 30, f"Processing took too long: {processing_time:.2f} seconds"
        
        # Average processing time should be less than 0.6 seconds per review
        avg_time = processing_time / len(reviews)
        assert avg_time < 0.6, f"Average processing time too slow: {avg_time:.3f} seconds per review"
    
    def test_error_handling(self):
        """Test error handling with malformed input."""
        # Test with None values
        try:
            result = self.analyzer.process_review_with_aspects(None, {}, "none_text")
            # Should handle gracefully
            assert isinstance(result, dict)
        except Exception as e:
            # Should raise a meaningful error
            assert "text" in str(e).lower() or "none" in str(e).lower()
        
        # Test with invalid aspect data
        try:
            result = self.analyzer.process_review_with_aspects(
                "Test text", {"invalid_aspect": "not_a_number"}, "invalid_aspects"
            )
            # Should handle gracefully
            assert isinstance(result, dict)
        except Exception as e:
            # Should handle gracefully or raise meaningful error
            assert isinstance(e, (TypeError, ValueError, KeyError))


if __name__ == "__main__":
    pytest.main([__file__])
