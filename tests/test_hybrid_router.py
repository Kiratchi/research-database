"""
Test suite for hybrid router module.

This module tests the hybrid routing system that intelligently routes queries
between fast-path and full workflow using TDD approach.
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict

from src.research_agent.core.hybrid_router import (
    HybridRouter,
    process_hybrid_query,
    stream_hybrid_query
)
from src.research_agent.core.query_classifier import QueryClassification
from src.research_agent.core.fast_path_workflow import FastPathResponse


class TestHybridRouter:
    """Test the HybridRouter class."""
    
    @pytest.fixture
    def mock_es_client(self):
        """Create a mock Elasticsearch client."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        return mock_client
    
    @pytest.fixture
    def router(self, mock_es_client):
        """Create a HybridRouter instance for testing."""
        return HybridRouter(es_client=mock_es_client)
    
    def test_router_initialization(self, router):
        """Test that router initializes correctly."""
        assert router.query_classifier is not None
        assert router.conversational_workflow is not None
        assert router.is_initialized() is True
    
    def test_router_initialization_without_es(self):
        """Test router initialization without Elasticsearch client."""
        router = HybridRouter()
        assert router.query_classifier is not None
        assert router.conversational_workflow is not None
        assert router.is_initialized() is True
    
    def test_conversational_query_routing(self, router):
        """Test routing of conversational queries to fast path."""
        # Mock the query classifier to return conversational
        mock_classification = QueryClassification(
            query_type="conversational",
            confidence=0.95,
            reasoning="Simple greeting",
            needs_tools=False,
            escalate_if_needed=False
        )
        
        # Mock the fast path workflow
        mock_fast_response = FastPathResponse(
            response="Hello! How can I help you today?",
            response_time=0.3,
            escalate=False,
            escalation_reason="",
            metadata={"workflow_type": "fast_path"}
        )
        
        with patch.object(router.query_classifier, 'classify_query', return_value=mock_classification), \
             patch.object(router.conversational_workflow, 'process_query', return_value=mock_fast_response):
            
            result = router.process_query("Hello")
            
            assert result['success'] is True
            assert result['response'] == "Hello! How can I help you today?"
            assert result['metadata']['workflow_type'] == 'fast_path'
            assert result['metadata']['response_time'] < 1.0
    
    def test_research_query_routing(self, router):
        """Test routing of research queries to full workflow."""
        # Mock the query classifier to return research
        mock_classification = QueryClassification(
            query_type="research",
            confidence=0.9,
            reasoning="Asks about publications",
            needs_tools=True,
            escalate_if_needed=True
        )
        
        with patch.object(router.query_classifier, 'classify_query', return_value=mock_classification), \
             patch('src.research_agent.core.hybrid_router.run_research_query') as mock_research:
            
            mock_research.return_value = {
                'response': 'John Smith has published 45 papers.',
                'plan': ['Search for author John Smith', 'Count publications'],
                'past_steps': [('search', 'found 45 papers')],
                'total_results': 45
            }
            
            result = router.process_query("How many papers has John Smith published?")
            
            assert result['success'] is True
            assert result['response'] == 'John Smith has published 45 papers.'
            assert result['metadata']['workflow_type'] == 'full_workflow'
            assert 'plan' in result['metadata']
    
    def test_escalation_from_fast_path(self, router):
        """Test escalation from fast path to full workflow."""
        # Mock classification as conversational
        mock_classification = QueryClassification(
            query_type="conversational",
            confidence=0.8,
            reasoning="Conversational but needs escalation",
            needs_tools=False,
            escalate_if_needed=True
        )
        
        # Mock fast path response with escalation
        mock_fast_response = FastPathResponse(
            response="I'd be happy to help you with that!",
            response_time=0.2,
            escalate=True,
            escalation_reason="Query requires database access",
            metadata={"workflow_type": "fast_path", "escalated": True}
        )
        
        with patch.object(router.query_classifier, 'classify_query', return_value=mock_classification), \
             patch.object(router.conversational_workflow, 'process_query', return_value=mock_fast_response), \
             patch('src.research_agent.core.hybrid_router.run_research_query') as mock_research:
            
            mock_research.return_value = {
                'response': 'Found 10 papers about machine learning.',
                'plan': ['Search for ML papers'],
                'past_steps': [('search', 'found papers')],
                'total_results': 10
            }
            
            result = router.process_query("Thanks! Now find papers about machine learning")
            
            assert result['success'] is True
            assert result['response'] == 'Found 10 papers about machine learning.'
            assert result['metadata']['workflow_type'] == 'escalated'
            assert result['metadata']['escalation_reason'] == 'Query requires database access'
    
    def test_performance_statistics_tracking(self, router):
        """Test that performance statistics are tracked."""
        initial_stats = router.get_performance_stats()
        
        # Mock fast path query
        mock_classification = QueryClassification(
            query_type="conversational",
            confidence=0.9,
            reasoning="Simple greeting",
            needs_tools=False,
            escalate_if_needed=False
        )
        
        mock_fast_response = FastPathResponse(
            response="Hello!",
            response_time=0.5,
            escalate=False,
            escalation_reason="",
            metadata={"workflow_type": "fast_path"}
        )
        
        with patch.object(router.query_classifier, 'classify_query', return_value=mock_classification), \
             patch.object(router.conversational_workflow, 'process_query', return_value=mock_fast_response):
            
            router.process_query("Hello")
            
            stats = router.get_performance_stats()
            
            assert stats['fast_path_count'] == initial_stats['fast_path_count'] + 1
            assert stats['total_queries'] == initial_stats['total_queries'] + 1
            assert stats['avg_fast_path_time'] > 0
    
    def test_should_use_fast_path_logic(self, router):
        """Test the fast path decision logic."""
        # High confidence conversational query
        mock_classification = QueryClassification(
            query_type="conversational",
            confidence=0.9,
            reasoning="Simple greeting",
            needs_tools=False,
            escalate_if_needed=False
        )
        
        with patch.object(router.query_classifier, 'classify_query', return_value=mock_classification):
            assert router.should_use_fast_path(mock_classification) is True
        
        # Low confidence conversational query
        mock_classification.confidence = 0.6
        assert router.should_use_fast_path(mock_classification) is False
        
        # Research query
        mock_classification.query_type = "research"
        mock_classification.confidence = 0.9
        mock_classification.needs_tools = True
        assert router.should_use_fast_path(mock_classification) is False
    
    def test_error_handling(self, router):
        """Test error handling in hybrid router."""
        with patch.object(router.query_classifier, 'classify_query', side_effect=Exception("Classification error")):
            result = router.process_query("Hello")
            
            assert result['success'] is False
            assert "error" in result['error'].lower()
            assert result['metadata']['workflow_type'] == 'error'
    
    def test_processing_message_generation(self, router):
        """Test appropriate processing message generation."""
        # Mock conversational classification
        mock_classification = QueryClassification(
            query_type="conversational",
            confidence=0.9,
            reasoning="Simple greeting",
            needs_tools=False,
            escalate_if_needed=False
        )
        
        with patch.object(router.query_classifier, 'classify_query', return_value=mock_classification):
            message = router.get_processing_message("Hello")
            assert "💬" in message or "Responding" in message
        
        # Mock research classification
        mock_classification.query_type = "research"
        mock_classification.needs_tools = True
        
        with patch.object(router.query_classifier, 'classify_query', return_value=mock_classification):
            message = router.get_processing_message("Find papers")
            assert "🔍" in message or "Researching" in message
    
    def test_router_info(self, router):
        """Test router information gathering."""
        info = router.get_router_info()
        
        assert 'initialized' in info
        assert 'research_agent_initialized' in info
        assert 'performance_stats' in info
        assert 'index_name' in info
        assert info['initialized'] is True
    
    @pytest.mark.asyncio
    async def test_streaming_query_conversational(self, router):
        """Test streaming for conversational queries."""
        # Mock classification and fast path response
        mock_classification = QueryClassification(
            query_type="conversational",
            confidence=0.9,
            reasoning="Simple greeting",
            needs_tools=False,
            escalate_if_needed=False
        )
        
        async def mock_stream_fast_path(query, history):
            yield {"type": "status", "content": "💬 Responding..."}
            yield {"type": "response_chunk", "content": "Hello!"}
            yield {"type": "final", "content": {"response": "Hello!", "response_time": 0.5}}
        
        with patch.object(router.query_classifier, 'classify_query', return_value=mock_classification), \
             patch.object(router.conversational_workflow, 'stream_query', side_effect=mock_stream_fast_path):
            
            events = []
            async for event in router.stream_query("Hello"):
                events.append(event)
            
            assert len(events) >= 3  # classification, status, response_chunk, final
            assert events[0]['type'] == 'classification'
            assert any(e['type'] == 'status' for e in events)
            assert any(e['type'] == 'final' for e in events)
    
    @pytest.mark.asyncio
    async def test_streaming_query_research(self, router):
        """Test streaming for research queries."""
        # Mock classification and research response
        mock_classification = QueryClassification(
            query_type="research",
            confidence=0.9,
            reasoning="Asks about publications",
            needs_tools=True,
            escalate_if_needed=True
        )
        
        async def mock_stream_research(query):
            yield {"planner": {"plan": ["Search for papers"]}}
            yield {"agent": {"response": "Searching..."}}
            yield {"__end__": {"response": "Found 45 papers"}}
        
        with patch.object(router.query_classifier, 'classify_query', return_value=mock_classification), \
             patch.object(router.research_agent, 'stream_query', side_effect=mock_stream_research):
            
            events = []
            async for event in router.stream_query("Find papers"):
                events.append(event)
            
            assert len(events) >= 2  # classification + research events
            assert events[0]['type'] == 'classification'
            assert any(e['type'] == 'plan' for e in events)
    
    def test_conversation_history_support(self, router):
        """Test that conversation history is properly passed through."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        mock_classification = QueryClassification(
            query_type="conversational",
            confidence=0.9,
            reasoning="Follow-up greeting",
            needs_tools=False,
            escalate_if_needed=False
        )
        
        mock_fast_response = FastPathResponse(
            response="How can I help you?",
            response_time=0.3,
            escalate=False,
            escalation_reason="",
            metadata={"workflow_type": "fast_path"}
        )
        
        with patch.object(router.query_classifier, 'classify_query', return_value=mock_classification) as mock_classify, \
             patch.object(router.conversational_workflow, 'process_query', return_value=mock_fast_response) as mock_process:
            
            router.process_query("How are you?", history)
            
            # Verify conversation history was passed to classifier
            mock_classify.assert_called_with("How are you?", history)
            # Verify conversation history was passed to fast path
            mock_process.assert_called_with("How are you?", history)


class TestHybridRouterPerformance:
    """Test performance requirements for hybrid router."""
    
    @pytest.fixture
    def router(self):
        """Create a HybridRouter instance for testing."""
        return HybridRouter()
    
    def test_fast_path_performance(self, router):
        """Test that fast path queries meet performance requirements."""
        mock_classification = QueryClassification(
            query_type="conversational",
            confidence=0.9,
            reasoning="Simple greeting",
            needs_tools=False,
            escalate_if_needed=False
        )
        
        mock_fast_response = FastPathResponse(
            response="Hello!",
            response_time=0.5,
            escalate=False,
            escalation_reason="",
            metadata={"workflow_type": "fast_path"}
        )
        
        with patch.object(router.query_classifier, 'classify_query', return_value=mock_classification), \
             patch.object(router.conversational_workflow, 'process_query', return_value=mock_fast_response):
            
            start_time = time.time()
            result = router.process_query("Hello")
            end_time = time.time()
            
            total_time = end_time - start_time
            assert total_time < 2.0, f"Fast path too slow: {total_time:.3f}s"
            assert result['metadata']['response_time'] < 1.0
    
    def test_classification_performance(self, router):
        """Test that query classification is fast."""
        queries = ["Hello", "How are you?", "Thank you", "Goodbye"]
        
        for query in queries:
            start_time = time.time()
            with patch.object(router.query_classifier, 'classify_query') as mock_classify:
                mock_classify.return_value = QueryClassification(
                    query_type="conversational",
                    confidence=0.9,
                    reasoning="Test",
                    needs_tools=False,
                    escalate_if_needed=False
                )
                
                router.should_use_fast_path(router.query_classifier.classify_query(query))
            
            end_time = time.time()
            classification_time = end_time - start_time
            assert classification_time < 0.5, f"Classification too slow for '{query}': {classification_time:.3f}s"


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    @patch('src.research_agent.core.hybrid_router.HybridRouter')
    def test_process_hybrid_query_function(self, mock_router_class):
        """Test the process_hybrid_query convenience function."""
        mock_instance = Mock()
        mock_result = {
            'success': True,
            'response': 'Hello!',
            'metadata': {'workflow_type': 'fast_path'}
        }
        mock_instance.process_query.return_value = mock_result
        mock_router_class.return_value = mock_instance
        
        result = process_hybrid_query("Hello")
        
        mock_router_class.assert_called_once_with(es_client=None, index_name="research-publications-static")
        mock_instance.process_query.assert_called_once_with("Hello", None)
        assert result == mock_result
    
    @pytest.mark.asyncio
    @patch('src.research_agent.core.hybrid_router.HybridRouter')
    async def test_stream_hybrid_query_function(self, mock_router_class):
        """Test the stream_hybrid_query convenience function."""
        mock_instance = Mock()
        
        # Mock async generator
        async def mock_stream_query(query, history):
            yield {"type": "status", "content": "Processing..."}
            yield {"type": "final", "content": "Hello!"}
        
        mock_instance.stream_query = mock_stream_query
        mock_router_class.return_value = mock_instance
        
        events = []
        async for event in stream_hybrid_query("Hello"):
            events.append(event)
        
        mock_router_class.assert_called_once_with(es_client=None, index_name="research-publications-static")
        assert len(events) == 2
        assert events[0]["type"] == "status"
        assert events[1]["type"] == "final"


class TestHybridRouterEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_initialization_with_invalid_es_client(self):
        """Test initialization with invalid Elasticsearch client."""
        mock_es = Mock()
        mock_es.ping.return_value = False
        
        router = HybridRouter(es_client=mock_es)
        
        # Should still initialize other components
        assert router.query_classifier is not None
        assert router.conversational_workflow is not None
        assert router.is_initialized() is True
    
    def test_research_agent_unavailable(self):
        """Test handling when research agent is unavailable."""
        router = HybridRouter()
        router.research_agent = None
        
        mock_classification = QueryClassification(
            query_type="research",
            confidence=0.9,
            reasoning="Asks about publications",
            needs_tools=True,
            escalate_if_needed=True
        )
        
        with patch.object(router.query_classifier, 'classify_query', return_value=mock_classification):
            result = router.process_query("Find papers")
            
            assert result['success'] is False
            assert "not initialized" in result['error'].lower()
    
    def test_empty_query_handling(self):
        """Test handling of empty queries."""
        router = HybridRouter()
        
        mock_classification = QueryClassification(
            query_type="research",
            confidence=0.5,
            reasoning="Empty query defaults to research",
            needs_tools=True,
            escalate_if_needed=True
        )
        
        with patch.object(router.query_classifier, 'classify_query', return_value=mock_classification), \
             patch('src.research_agent.core.hybrid_router.run_research_query') as mock_research:
            
            mock_research.return_value = {
                'response': 'I can help you search for publications.',
                'plan': [],
                'past_steps': [],
                'total_results': 0
            }
            
            result = router.process_query("")
            
            assert result['success'] is True
            assert result['response'] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])