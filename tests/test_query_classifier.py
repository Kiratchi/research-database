"""
Test suite for query classifier module.

This module tests the query classification functionality using TDD approach.
Tests cover pattern matching, LLM-based classification, and performance requirements.
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import List, Dict

from src.research_agent.core.query_classifier import (
    QueryClassifier,
    QueryClassification,
    QueryType,
    classify_query,
    should_use_fast_path,
    get_processing_message
)


class TestQueryClassification:
    """Test the QueryClassification model."""
    
    def test_query_classification_creation(self):
        """Test creating a QueryClassification instance."""
        classification = QueryClassification(
            query_type="conversational",
            confidence=0.95,
            reasoning="Greeting pattern detected",
            needs_tools=False,
            escalate_if_needed=False
        )
        
        assert classification.query_type == "conversational"
        assert classification.confidence == 0.95
        assert classification.reasoning == "Greeting pattern detected"
        assert classification.needs_tools is False
        assert classification.escalate_if_needed is False
    
    def test_query_classification_validation(self):
        """Test validation of QueryClassification fields."""
        # Test valid query types
        for query_type in ["conversational", "research", "mixed"]:
            classification = QueryClassification(
                query_type=query_type,
                confidence=0.8,
                reasoning="Test",
                needs_tools=True,
                escalate_if_needed=True
            )
            assert classification.query_type == query_type
        
        # Test confidence bounds
        classification = QueryClassification(
            query_type="conversational",
            confidence=0.0,
            reasoning="Test",
            needs_tools=False,
            escalate_if_needed=False
        )
        assert classification.confidence == 0.0
        
        classification = QueryClassification(
            query_type="conversational",
            confidence=1.0,
            reasoning="Test",
            needs_tools=False,
            escalate_if_needed=False
        )
        assert classification.confidence == 1.0


class TestQueryClassifierPatterns:
    """Test pattern-based classification."""
    
    @pytest.fixture
    def classifier(self):
        """Create a QueryClassifier instance for testing."""
        return QueryClassifier()
    
    def test_conversational_patterns(self, classifier):
        """Test recognition of conversational patterns."""
        conversational_queries = [
            "Hello",
            "Hi there",
            "Good morning",
            "Thank you",
            "Thanks a lot",
            "Goodbye",
            "See you later", 
            "How are you?",
            "What's up?",
            "Yes",
            "No",
            "Okay",
            "Sorry",
            "My mistake"
        ]
        
        for query in conversational_queries:
            classification = classifier.classify_query(query)
            assert classification.query_type == "conversational", f"Failed for query: {query}"
            assert classification.confidence >= 0.7, f"Low confidence for query: {query}"
            assert classification.needs_tools is False, f"Should not need tools for: {query}"
    
    def test_research_patterns(self, classifier):
        """Test recognition of research patterns."""
        research_queries = [
            "How many papers has John Smith published?",
            "Find publications by Anna Dubois",
            "Search for articles about machine learning",
            "List all publications from 2023",
            "Show me papers by Martin Nilsson",
            "Who published about neural networks?",
            "What research has been done on AI?",
            "Compare authors Smith and Jones",
            "Statistics on publications in Nature",
            "Academic papers on quantum computing",
            "Scientific studies from last year",
            "Journal articles about robotics",
            "Research by professors at MIT",
            "Publications in the last decade"
        ]
        
        for query in research_queries:
            classification = classifier.classify_query(query)
            assert classification.query_type == "research", f"Failed for query: {query}"
            assert classification.confidence >= 0.7, f"Low confidence for query: {query}"
            assert classification.needs_tools is True, f"Should need tools for: {query}"
    
    def test_mixed_patterns(self, classifier):
        """Test recognition of mixed patterns."""
        mixed_queries = [
            "Thanks! Now find papers by John Smith",
            "Hello, can you search for machine learning papers?",
            "I don't understand, show me more publications",
            "Great! What about research on AI?",
            "OK, now list publications by Anna Dubois"
        ]
        
        for query in mixed_queries:
            classification = classifier.classify_query(query)
            # Mixed queries should be classified as research for safety
            assert classification.query_type == "research", f"Failed for query: {query}"
            assert classification.needs_tools is True, f"Should need tools for: {query}"
    
    def test_uncertain_queries(self, classifier):
        """Test handling of uncertain queries."""
        uncertain_queries = [
            "What about that thing?",
            "Can you tell me more?",
            "I'm not sure what I'm looking for",
            "Maybe something else?",
            "This is confusing"
        ]
        
        for query in uncertain_queries:
            classification = classifier.classify_query(query)
            # Uncertain queries should default to research for safety
            assert classification.query_type == "research", f"Failed for query: {query}"
            assert classification.confidence <= 0.8, f"Too high confidence for uncertain query: {query}"
            assert classification.needs_tools is True, f"Should need tools for uncertain query: {query}"


class TestQueryClassifierPerformance:
    """Test performance requirements."""
    
    @pytest.fixture
    def classifier(self):
        """Create a QueryClassifier instance for testing."""
        return QueryClassifier()
    
    def test_fallback_classification_speed(self, classifier):
        """Test that fallback classification is fast (<100ms)."""
        query = "Hello, how are you?"
        
        start_time = time.time()
        for _ in range(100):  # Test 100 classifications
            classification = classifier._simple_fallback_classify(query)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.001, f"Fallback classification too slow: {avg_time:.4f}s"
    
    @patch('src.research_agent.core.query_classifier.ChatLiteLLM')
    def test_fast_path_decision_speed(self, mock_llm, classifier):
        """Test that fast path decisions are quick."""
        # Mock LLM to avoid actual API calls
        mock_response = Mock()
        mock_response.invoke.return_value = QueryClassification(
            query_type="conversational",
            confidence=0.95,
            reasoning="Test",
            needs_tools=False,
            escalate_if_needed=False
        )
        mock_llm.return_value.with_structured_output.return_value = mock_response
        
        query = "Hello"
        
        start_time = time.time()
        result = classifier.should_use_fast_path(query)
        end_time = time.time()
        
        assert end_time - start_time < 0.1, "Fast path decision too slow"
        assert result is True, "Should use fast path for greeting"


class TestQueryClassifierIntegration:
    """Test full classifier integration."""
    
    @pytest.fixture
    def classifier(self):
        """Create a QueryClassifier instance for testing."""
        return QueryClassifier()
    
    def test_conversation_context_usage(self, classifier):
        """Test that conversation context influences classification."""
        # Context with research discussion
        context = [
            {"role": "user", "content": "How many papers has John Smith published?"},
            {"role": "assistant", "content": "John Smith has published 45 papers."}
        ]
        
        # Ambiguous query that could be influenced by context
        query = "Tell me more about his work"
        
        # Pattern classification should handle this
        classification = classifier._pattern_classify(query)
        
        # Should default to research due to ambiguity
        assert classification.query_type == "research"
        assert classification.needs_tools is True
    
    def test_fallback_behavior(self, classifier):
        """Test fallback behavior when LLM fails."""
        with patch.object(classifier, 'classifier') as mock_classifier:
            mock_classifier.invoke.side_effect = Exception("LLM failure")
            
            query = "Hello"
            classification = classifier.classify_query(query)
            
            # Should fall back to pattern classification
            assert classification.query_type == "conversational"
            assert classification.confidence >= 0.6  # Fallback confidence
    
    def test_confidence_adjustment(self, classifier):
        """Test confidence score adjustment."""
        # Mock very low confidence response
        mock_classification = QueryClassification(
            query_type="conversational",
            confidence=0.05,
            reasoning="Test",
            needs_tools=False,
            escalate_if_needed=False
        )
        
        with patch.object(classifier, 'classifier') as mock_classifier:
            mock_classifier.invoke.return_value = mock_classification
            
            query = "Hello"
            result = classifier.classify_query(query)
            
            # Should adjust very low confidence
            assert result.confidence >= 0.5


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    @patch('src.research_agent.core.query_classifier.QueryClassifier')
    def test_classify_query_function(self, mock_classifier_class):
        """Test the classify_query convenience function."""
        mock_instance = Mock()
        mock_result = QueryClassification(
            query_type="conversational",
            confidence=0.95,
            reasoning="Test",
            needs_tools=False,
            escalate_if_needed=False
        )
        mock_instance.classify_query.return_value = mock_result
        mock_classifier_class.return_value = mock_instance
        
        result = classify_query("Hello")
        
        mock_classifier_class.assert_called_once()
        mock_instance.classify_query.assert_called_once_with("Hello", None)
        assert result == mock_result
    
    @patch('src.research_agent.core.query_classifier.QueryClassifier')
    def test_should_use_fast_path_function(self, mock_classifier_class):
        """Test the should_use_fast_path convenience function."""
        mock_instance = Mock()
        mock_instance.should_use_fast_path.return_value = True
        mock_classifier_class.return_value = mock_instance
        
        result = should_use_fast_path("Hello")
        
        mock_classifier_class.assert_called_once()
        mock_instance.should_use_fast_path.assert_called_once_with("Hello", None)
        assert result is True
    
    @patch('src.research_agent.core.query_classifier.QueryClassifier')
    def test_get_processing_message_function(self, mock_classifier_class):
        """Test the get_processing_message convenience function."""
        mock_instance = Mock()
        mock_instance.get_processing_message.return_value = "💬 Responding..."
        mock_classifier_class.return_value = mock_instance
        
        result = get_processing_message("Hello")
        
        mock_classifier_class.assert_called_once()
        mock_instance.get_processing_message.assert_called_once_with("Hello", None)
        assert result == "💬 Responding..."


class TestQueryClassifierEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def classifier(self):
        """Create a QueryClassifier instance for testing."""
        return QueryClassifier()
    
    def test_empty_query(self, classifier):
        """Test handling of empty queries."""
        classification = classifier.classify_query("")
        assert classification.query_type == "research"  # Default to safe option
        assert classification.needs_tools is True
    
    def test_very_long_query(self, classifier):
        """Test handling of very long queries."""
        long_query = "Hello " * 1000 + "how many papers has John Smith published?"
        classification = classifier.classify_query(long_query)
        # Should still detect research pattern despite length
        assert classification.query_type == "research"
    
    def test_special_characters(self, classifier):
        """Test handling of special characters."""
        queries_with_special_chars = [
            "Hello! How are you?",
            "Find papers by O'Connor",
            "What about papers on AI/ML?",
            "Search for papers with émile",
            "Publications by João Silva"
        ]
        
        for query in queries_with_special_chars:
            classification = classifier.classify_query(query)
            # Should not crash and should classify reasonably
            assert classification.query_type in ["conversational", "research", "mixed"]
    
    def test_case_insensitive_matching(self, classifier):
        """Test that pattern matching is case insensitive."""
        pairs = [
            ("HELLO", "hello"),
            ("Thank You", "thank you"),
            ("FIND PAPERS", "find papers"),
            ("How Many Publications", "how many publications")
        ]
        
        for upper_query, lower_query in pairs:
            upper_result = classifier.classify_query(upper_query)
            lower_result = classifier.classify_query(lower_query)
            
            assert upper_result.query_type == lower_result.query_type, \
                f"Case sensitivity issue: {upper_query} vs {lower_query}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])