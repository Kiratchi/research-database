"""
Test suite for fast-path workflow module.

This module tests the fast conversational response workflow using TDD approach.
Tests cover response generation, escalation logic, and performance requirements.
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict

from src.research_agent.core.fast_path_workflow import (
    ConversationalWorkflow,
    FastPathResponse,
    process_conversational_query,
    stream_conversational_query,
    create_conversational_workflow,
    get_workflow_memory_summary
)


class TestFastPathResponse:
    """Test the FastPathResponse model."""
    
    def test_fast_path_response_creation(self):
        """Test creating a FastPathResponse instance."""
        response = FastPathResponse(
            response="Hello! How can I help you today?",
            response_time=0.5,
            escalate=False,
            escalation_reason="",
            metadata={"workflow_type": "fast_path"}
        )
        
        assert response.response == "Hello! How can I help you today?"
        assert response.response_time == 0.5
        assert response.escalate is False
        assert response.escalation_reason == ""
        assert response.metadata["workflow_type"] == "fast_path"
    
    def test_fast_path_response_escalation(self):
        """Test FastPathResponse with escalation."""
        response = FastPathResponse(
            response="I'd be happy to help you with that! Let me search the database.",
            response_time=0.3,
            escalate=True,
            escalation_reason="Query requires database access",
            metadata={"workflow_type": "fast_path", "escalated": True}
        )
        
        assert response.escalate is True
        assert response.escalation_reason == "Query requires database access"
        assert response.metadata["escalated"] is True


class TestConversationalWorkflow:
    """Test the ConversationalWorkflow class."""
    
    @pytest.fixture
    def workflow(self):
        """Create a ConversationalWorkflow instance for testing."""
        return ConversationalWorkflow()
    
    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response for testing."""
        mock = Mock()
        mock.content = "Hello! How can I help you today?"
        return mock
    
    def test_workflow_initialization(self, workflow):
        """Test that workflow initializes correctly."""
        assert workflow.llm is not None
        assert workflow.memory is not None
        assert workflow.conversation_chain is not None
        assert hasattr(workflow, 'memory')
        assert workflow.memory.memory_key == "chat_history"
    
    def test_conversational_query_processing(self, workflow):
        """Test basic conversational query processing."""
        with patch.object(workflow, 'conversation_chain') as mock_chain:
            mock_response = {"response": "Hello! How can I help you today?"}
            mock_chain.invoke.return_value = mock_response
            
            result = workflow.process_query("Hello")
            
            assert result.response == "Hello! How can I help you today?"
            assert result.response_time < 2.0  # Should be fast
            assert result.escalate is False
            assert result.metadata["workflow_type"] == "fast_path"
    
    def test_escalation_detection(self, workflow):
        """Test detection of queries that need escalation."""
        with patch.object(workflow, 'conversation_chain') as mock_chain:
            mock_response = {"response": "ESCALATE: Query requires database search"}
            mock_chain.invoke.return_value = mock_response
            
            result = workflow.process_query("Find papers by John Smith")
            
            assert result.escalate is True
            assert result.escalation_reason == "Query requires database search"
            assert result.response == "I'd be happy to help you with that! Let me search the publications database for you."
    
    def test_memory_initialization_from_history(self, workflow):
        """Test memory initialization from conversation history."""
        history = [
            {"role": "user", "content": "Hello", "timestamp": "2023-01-01"},
            {"role": "assistant", "content": "Hi! How can I help?", "timestamp": "2023-01-01"},
            {"role": "user", "content": "How are you?", "timestamp": "2023-01-01"}
        ]
        
        workflow._initialize_memory_from_history(history)
        
        memory_summary = workflow.get_memory_summary()
        assert memory_summary["total_messages"] == 3
        assert memory_summary["user_messages"] == 2
        assert memory_summary["ai_messages"] == 1
    
    def test_memory_content_truncation(self, workflow):
        """Test that very long messages are truncated in memory."""
        long_content = "This is a very long message that should be truncated " * 20
        history = [
            {"role": "user", "content": long_content, "timestamp": "2023-01-01"}
        ]
        
        workflow._initialize_memory_from_history(history)
        
        # Check that message was truncated to 500 characters + "..."
        messages = workflow.memory.chat_memory.messages
        assert len(messages) == 1
        assert len(messages[0].content) <= 503  # 500 + "..."
        assert "..." in messages[0].content
    
    def test_error_handling(self, workflow):
        """Test error handling in conversational workflow."""
        with patch.object(workflow, 'conversation_chain') as mock_chain:
            mock_chain.invoke.side_effect = Exception("LLM error")
            
            result = workflow.process_query("Hello")
            
            assert result.escalate is True
            assert "error" in result.escalation_reason.lower()
            assert result.metadata["workflow_type"] == "fast_path"
    
    @pytest.mark.asyncio
    async def test_streaming_query(self, workflow):
        """Test streaming query processing."""
        mock_response = "Hello! How can I help you today?"
        
        with patch.object(workflow, 'conversation_chain') as mock_chain:
            # Mock async streaming
            async def mock_astream(input_data):
                for chunk in ["Hello!", " How can", " I help", " you today?"]:
                    yield {"response": chunk}
            
            mock_chain.astream = mock_astream
            
            events = []
            async for event in workflow.stream_query("Hello"):
                events.append(event)
            
            # Should have status, response chunks, and final event
            assert len(events) >= 3
            assert events[0]["type"] == "status"
            assert any(event["type"] == "response_chunk" for event in events)
            assert events[-1]["type"] == "final"
    
    @pytest.mark.asyncio
    async def test_streaming_escalation(self, workflow):
        """Test streaming with escalation."""
        with patch.object(workflow, 'conversation_chain') as mock_chain:
            # Mock async streaming that returns escalation
            async def mock_astream(input_data):
                yield {"response": "ESCALATE: Need database access"}
            
            mock_chain.astream = mock_astream
            
            events = []
            async for event in workflow.stream_query("Find papers"):
                events.append(event)
            
            # Should have escalation event
            escalation_events = [e for e in events if e.get("type") == "escalation"]
            assert len(escalation_events) == 1
            assert escalation_events[0]["content"]["reason"] == "Need database access"


class TestConversationalWorkflowPerformance:
    """Test performance requirements for conversational workflow."""
    
    @pytest.fixture
    def workflow(self):
        """Create a ConversationalWorkflow instance for testing."""
        return ConversationalWorkflow()
    
    def test_response_time_performance(self, workflow):
        """Test that conversational responses are fast (<2s)."""
        with patch.object(workflow, 'conversation_chain') as mock_chain:
            mock_response = Mock()
            mock_response.content = "Hello! How can I help you today?"
            mock_chain.invoke.return_value = mock_response
            
            start_time = time.time()
            result = workflow.process_query("Hello")
            end_time = time.time()
            
            total_time = end_time - start_time
            assert total_time < 2.0, f"Response too slow: {total_time:.3f}s"
            assert result.response_time < 2.0, f"Reported response time too slow: {result.response_time:.3f}s"
    
    def test_multiple_queries_performance(self, workflow):
        """Test performance with multiple consecutive queries."""
        with patch.object(workflow, 'conversation_chain') as mock_chain:
            mock_response = Mock()
            mock_response.content = "Test response"
            mock_chain.invoke.return_value = mock_response
            
            queries = ["Hello", "How are you?", "Thank you", "Goodbye"]
            response_times = []
            
            for query in queries:
                start_time = time.time()
                result = workflow.process_query(query)
                end_time = time.time()
                
                response_times.append(end_time - start_time)
            
            # All responses should be fast
            for i, response_time in enumerate(response_times):
                assert response_time < 2.0, f"Query {i} too slow: {response_time:.3f}s"
            
            # Average response time should be very fast
            avg_time = sum(response_times) / len(response_times)
            assert avg_time < 1.0, f"Average response time too slow: {avg_time:.3f}s"


class TestConversationalWorkflowMemory:
    """Test LangChain Memory functionality in conversational workflow."""
    
    @pytest.fixture
    def workflow(self):
        """Create a ConversationalWorkflow instance for testing."""
        return ConversationalWorkflow()
    
    def test_memory_persistence_across_queries(self, workflow):
        """Test that memory persists across multiple queries."""
        with patch.object(workflow, 'conversation_chain') as mock_chain:
            # First query
            mock_chain.invoke.return_value = {"response": "Hello! I'm Claude."}
            result1 = workflow.process_query("What's your name?")
            
            # Second query - memory should have the previous conversation
            mock_chain.invoke.return_value = {"response": "I just told you, I'm Claude!"}
            result2 = workflow.process_query("What's your name again?")
            
            # Check that memory contains both exchanges
            memory_summary = workflow.get_memory_summary()
            assert memory_summary["total_messages"] == 4  # 2 user + 2 assistant
            assert memory_summary["user_messages"] == 2
            assert memory_summary["ai_messages"] == 2
    
    def test_memory_clear_functionality(self, workflow):
        """Test clearing conversation memory."""
        with patch.object(workflow, 'conversation_chain') as mock_chain:
            mock_chain.invoke.return_value = {"response": "Hello!"}
            workflow.process_query("Hi")
            
            # Memory should have messages
            memory_summary = workflow.get_memory_summary()
            assert memory_summary["total_messages"] > 0
            
            # Clear memory
            workflow.clear_memory()
            
            # Memory should be empty
            memory_summary = workflow.get_memory_summary()
            assert memory_summary["total_messages"] == 0
    
    def test_memory_instance_access(self, workflow):
        """Test direct access to memory instance."""
        memory = workflow.get_conversation_memory()
        assert memory is not None
        assert hasattr(memory, 'chat_memory')
        assert hasattr(memory, 'memory_key')
        assert memory.memory_key == "history"


class TestConversationalWorkflowIntegration:
    """Test integration scenarios for conversational workflow."""
    
    @pytest.fixture
    def workflow(self):
        """Create a ConversationalWorkflow instance for testing."""
        return ConversationalWorkflow()
    
    def test_context_aware_responses(self, workflow):
        """Test that workflow uses conversation context."""
        history = [
            {"role": "user", "content": "My name is John"},
            {"role": "assistant", "content": "Nice to meet you, John!"}
        ]
        
        with patch.object(workflow, 'conversation_chain') as mock_chain:
            mock_response = Mock()
            mock_response.content = "Hello again, John!"
            mock_chain.invoke.return_value = mock_response
            
            result = workflow.process_query("Hello", history)
            
            # Should initialize memory with history
            memory_summary = workflow.get_memory_summary()
            assert memory_summary["total_messages"] == 2
            assert memory_summary["user_messages"] == 1
            assert memory_summary["ai_messages"] == 1
    
    def test_no_context_handling(self, workflow):
        """Test handling when no conversation context is provided."""
        with patch.object(workflow, 'conversation_chain') as mock_chain:
            mock_response = Mock()
            mock_response.content = "Hello! How can I help you today?"
            mock_chain.invoke.return_value = mock_response
            
            result = workflow.process_query("Hello", None)
            
            # Should handle None context gracefully
            memory_summary = workflow.get_memory_summary()
            assert memory_summary["total_messages"] == 0
    
    def test_escalation_with_context(self, workflow):
        """Test escalation detection with conversation context."""
        history = [
            {"role": "user", "content": "I'm researching machine learning"},
            {"role": "assistant", "content": "That's a fascinating field!"}
        ]
        
        with patch.object(workflow, 'conversation_chain') as mock_chain:
            mock_response = Mock()
            mock_response.content = "ESCALATE: User wants to find ML papers"
            mock_chain.invoke.return_value = mock_response
            
            result = workflow.process_query("Show me some papers", history)
            
            assert result.escalate is True
            assert "ML papers" in result.escalation_reason


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    @patch('src.research_agent.core.fast_path_workflow.ConversationalWorkflow')
    def test_process_conversational_query_function(self, mock_workflow_class):
        """Test the process_conversational_query convenience function."""
        mock_instance = Mock()
        mock_result = FastPathResponse(
            response="Hello!",
            response_time=0.5,
            escalate=False,
            escalation_reason="",
            metadata={"workflow_type": "fast_path"}
        )
        mock_instance.process_query.return_value = mock_result
        mock_workflow_class.return_value = mock_instance
        
        result = process_conversational_query("Hello")
        
        mock_workflow_class.assert_called_once()
        mock_instance.process_query.assert_called_once_with("Hello", None)
        assert result == mock_result
    
    def test_create_conversational_workflow_function(self):
        """Test the create_conversational_workflow convenience function."""
        workflow = create_conversational_workflow()
        
        assert isinstance(workflow, ConversationalWorkflow)
        assert workflow.memory is not None
        assert workflow.conversation_chain is not None
    
    def test_get_workflow_memory_summary_function(self):
        """Test the get_workflow_memory_summary convenience function."""
        workflow = create_conversational_workflow()
        
        # Add some test data to memory
        workflow.memory.chat_memory.add_user_message("Hello")
        workflow.memory.chat_memory.add_ai_message("Hi there!")
        
        summary = get_workflow_memory_summary(workflow)
        
        assert summary["total_messages"] == 2
        assert summary["user_messages"] == 1
        assert summary["ai_messages"] == 1
    
    @pytest.mark.asyncio
    @patch('src.research_agent.core.fast_path_workflow.ConversationalWorkflow')
    async def test_stream_conversational_query_function(self, mock_workflow_class):
        """Test the stream_conversational_query convenience function."""
        mock_instance = Mock()
        
        # Mock async generator
        async def mock_stream_query(query, history):
            yield {"type": "status", "content": "Processing..."}
            yield {"type": "final", "content": "Hello!"}
        
        mock_instance.stream_query = mock_stream_query
        mock_workflow_class.return_value = mock_instance
        
        events = []
        async for event in stream_conversational_query("Hello"):
            events.append(event)
        
        mock_workflow_class.assert_called_once()
        assert len(events) == 2
        assert events[0]["type"] == "status"
        assert events[1]["type"] == "final"


class TestConversationalWorkflowEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def workflow(self):
        """Create a ConversationalWorkflow instance for testing."""
        return ConversationalWorkflow()
    
    def test_empty_query_handling(self, workflow):
        """Test handling of empty queries."""
        with patch.object(workflow, 'conversation_chain') as mock_chain:
            mock_response = Mock()
            mock_response.content = "I'm here to help! What would you like to know?"
            mock_chain.invoke.return_value = mock_response
            
            result = workflow.process_query("")
            
            assert result.response is not None
            assert result.escalate is False
    
    def test_very_long_query_handling(self, workflow):
        """Test handling of very long queries."""
        long_query = "Hello " * 1000
        
        with patch.object(workflow, 'conversation_chain') as mock_chain:
            mock_response = Mock()
            mock_response.content = "I understand you're greeting me!"
            mock_chain.invoke.return_value = mock_response
            
            result = workflow.process_query(long_query)
            
            assert result.response is not None
            assert result.response_time < 5.0  # Should still be reasonably fast
    
    def test_unicode_handling(self, workflow):
        """Test handling of unicode characters."""
        unicode_queries = [
            "Hola! ¿Cómo estás?",
            "Bonjour! Comment allez-vous?",
            "こんにちは！元気ですか？",
            "Привет! Как дела?"
        ]
        
        for query in unicode_queries:
            with patch.object(workflow, 'conversation_chain') as mock_chain:
                mock_response = Mock()
                mock_response.content = "Hello! I understand you're greeting me."
                mock_chain.invoke.return_value = mock_response
                
                result = workflow.process_query(query)
                
                assert result.response is not None
                assert result.escalate is False
    
    def test_llm_initialization_failure(self):
        """Test graceful handling of LLM initialization failure."""
        with patch('src.research_agent.core.fast_path_workflow.ChatLiteLLM') as mock_llm:
            mock_llm.side_effect = Exception("LLM initialization failed")
            
            # Should not crash during initialization
            workflow = ConversationalWorkflow()
            
            # Should handle queries gracefully even without LLM
            result = workflow.process_query("Hello")
            
            assert result.escalate is True
            assert "error" in result.escalation_reason.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])