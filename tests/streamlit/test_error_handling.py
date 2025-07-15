"""
Test suite for error handling and loading states (Phase 1.3)

Following TDD approach from STREAMLIT_CHAT_TDD_PLAN.md
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from streamlit_agent import StreamlitAgent
from elasticsearch import Elasticsearch, ConnectionError, NotFoundError


class TestAgentInitializationFailure:
    """Test graceful handling of agent init failures"""
    
    def test_agent_initialization_failure_no_es_client(self):
        """Test graceful handling when ES client fails to initialize"""
        # Test with None ES client
        agent = StreamlitAgent(es_client=None)
        
        # Should still initialize but with limitations
        assert agent.es_client is None
        assert agent.is_initialized() is True  # Should work without ES for testing
    
    def test_agent_initialization_failure_bad_es_client(self):
        """Test graceful handling when ES client connection fails"""
        # Mock ES client that fails ping
        mock_es_client = Mock()
        mock_es_client.ping.return_value = False
        
        agent = StreamlitAgent(es_client=mock_es_client)
        
        # Should initialize but ES operations will fail
        assert agent.es_client == mock_es_client
        assert agent.is_initialized() is True
    
    def test_agent_initialization_failure_exception(self):
        """Test graceful handling when ResearchAgent initialization throws exception"""
        # Mock ES client that raises exception during agent creation
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        
        with patch('streamlit_agent.ResearchAgent') as mock_research_agent:
            mock_research_agent.side_effect = Exception("ResearchAgent initialization failed")
            
            agent = StreamlitAgent(es_client=mock_es_client)
            
            # Should handle the exception gracefully
            assert agent.research_agent is None
            assert agent.is_initialized() is False
    
    def test_get_agent_info_when_not_initialized(self):
        """Test agent info when agent is not initialized"""
        with patch('streamlit_agent.ResearchAgent') as mock_research_agent:
            mock_research_agent.side_effect = Exception("Failed to initialize")
            
            agent = StreamlitAgent()
            info = agent.get_agent_info()
            
            assert info['initialized'] is False
            assert 'error' in info


class TestQueryExecutionErrors:
    """Test handling of query execution errors"""
    
    @pytest.mark.asyncio
    async def test_query_execution_error_no_agent(self):
        """Test query execution when agent is not initialized"""
        agent = StreamlitAgent()
        agent.research_agent = None  # Force uninitialized state
        
        result = await agent.process_query("test query")
        
        assert result['success'] is False
        assert 'not initialized' in result['error'].lower()
        assert result['response'] is None
    
    @pytest.mark.asyncio
    async def test_query_execution_error_research_agent_exception(self):
        """Test query execution when ResearchAgent throws exception"""
        # Mock ES client
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        
        agent = StreamlitAgent(es_client=mock_es_client)
        
        # Mock research agent that throws exception
        mock_research_agent = Mock()
        mock_research_agent.query = AsyncMock(side_effect=Exception("Query execution failed"))
        agent.research_agent = mock_research_agent
        
        result = await agent.process_query("test query")
        
        assert result['success'] is False
        assert 'Query execution failed' in result['error']
        assert 'traceback' in result
    
    @pytest.mark.asyncio
    async def test_streaming_query_error_no_agent(self):
        """Test streaming query when agent is not initialized"""
        agent = StreamlitAgent()
        agent.research_agent = None  # Force uninitialized state
        
        events = []
        async for event in agent.stream_query("test query"):
            events.append(event)
        
        assert len(events) == 1
        assert events[0]['type'] == 'error'
        assert 'not initialized' in events[0]['content'].lower()
    
    @pytest.mark.asyncio
    async def test_streaming_query_error_research_agent_exception(self):
        """Test streaming query when ResearchAgent throws exception"""
        # Mock ES client
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        
        agent = StreamlitAgent(es_client=mock_es_client)
        
        # Mock research agent that throws exception during streaming
        async def mock_stream_query(query):
            raise Exception("Streaming failed")
        
        mock_research_agent = Mock()
        mock_research_agent.stream_query = mock_stream_query
        agent.research_agent = mock_research_agent
        
        events = []
        async for event in agent.stream_query("test query"):
            events.append(event)
        
        assert len(events) == 1
        assert events[0]['type'] == 'error'
        assert 'Streaming failed' in events[0]['content']
        assert 'traceback' in events[0]
    
    def test_sync_query_error_no_event_loop(self):
        """Test sync query execution when event loop is not available"""
        agent = StreamlitAgent()
        agent.research_agent = None  # Force uninitialized state
        
        result = agent.run_sync_query("test query")
        
        assert result['success'] is False
        assert result['response'] is None
    
    def test_sync_query_error_asyncio_exception(self):
        """Test sync query execution when asyncio throws exception"""
        # Mock ES client
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        
        agent = StreamlitAgent(es_client=mock_es_client)
        
        # Mock asyncio.run to raise exception
        with patch('asyncio.run', side_effect=Exception("Asyncio failed")):
            with patch('asyncio.get_event_loop', side_effect=RuntimeError("No event loop")):
                with patch('asyncio.new_event_loop') as mock_new_loop:
                    with patch('asyncio.set_event_loop'):
                        mock_new_loop.return_value.run_until_complete.side_effect = Exception("Loop failed")
                        
                        result = agent.run_sync_query("test query")
                        
                        assert result['success'] is False
                        assert 'Error running sync query' in result['error']
                        assert 'traceback' in result


class TestLoadingStates:
    """Test loading indicators during processing"""
    
    def test_agent_initialization_loading_state(self):
        """Test loading state during agent initialization"""
        # This test would verify that loading indicators are shown
        # during agent initialization in the Streamlit interface
        pass
    
    def test_query_processing_loading_state(self):
        """Test loading state during query processing"""
        # This test would verify that loading indicators are shown
        # during query execution in the Streamlit interface
        pass
    
    def test_streaming_query_progressive_loading(self):
        """Test progressive loading during streaming query"""
        # This test would verify that streaming updates show progressive
        # loading states as the query is processed
        pass


class TestErrorRecovery:
    """Test error recovery mechanisms"""
    
    @pytest.mark.asyncio
    async def test_elasticsearch_connection_recovery(self):
        """Test recovery from Elasticsearch connection errors"""
        # Mock ES client that initially fails but then succeeds
        mock_es_client = Mock()
        mock_es_client.ping.side_effect = [False, True]  # First fails, then succeeds
        
        agent = StreamlitAgent(es_client=mock_es_client)
        
        # First check should show not connected
        info = agent.get_agent_info()
        assert info['es_client_connected'] is False
        
        # Second check should show connected
        info = agent.get_agent_info()
        assert info['es_client_connected'] is True
    
    @pytest.mark.asyncio
    async def test_query_retry_after_failure(self):
        """Test query retry mechanism after failure"""
        # Mock ES client
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        
        agent = StreamlitAgent(es_client=mock_es_client)
        
        # Mock research agent that fails first time, succeeds second time
        mock_research_agent = Mock()
        mock_research_agent.query = AsyncMock(side_effect=[
            Exception("First failure"),
            {"response": "Success on retry", "plan": [], "past_steps": []}
        ])
        agent.research_agent = mock_research_agent
        
        # First query should fail
        result1 = await agent.process_query("test query")
        assert result1['success'] is False
        
        # Second query should succeed
        result2 = await agent.process_query("test query")
        assert result2['success'] is True
        assert result2['response'] == "Success on retry"
    
    def test_session_state_recovery_after_error(self):
        """Test session state recovery after errors"""
        # This test would verify that session state is properly maintained
        # even after errors occur
        pass


class TestUserFriendlyErrorMessages:
    """Test user-friendly error message formatting"""
    
    def test_connection_error_message(self):
        """Test user-friendly message for connection errors"""
        # Test that connection errors are formatted in a user-friendly way
        pass
    
    def test_timeout_error_message(self):
        """Test user-friendly message for timeout errors"""
        # Test that timeout errors are formatted in a user-friendly way
        pass
    
    def test_query_format_error_message(self):
        """Test user-friendly message for query format errors"""
        # Test that query format errors are formatted in a user-friendly way
        pass


# Fixtures for error handling tests
@pytest.fixture
def mock_failing_es_client():
    """Mock Elasticsearch client that fails operations"""
    client = Mock()
    client.ping.return_value = False
    client.search.side_effect = ConnectionError("Connection failed")
    return client


@pytest.fixture
def mock_timeout_es_client():
    """Mock Elasticsearch client that times out"""
    client = Mock()
    client.ping.return_value = True
    client.search.side_effect = TimeoutError("Request timed out")
    return client


@pytest.fixture
def mock_intermittent_es_client():
    """Mock Elasticsearch client with intermittent failures"""
    client = Mock()
    client.ping.side_effect = [True, False, True, True]  # Intermittent failures
    return client