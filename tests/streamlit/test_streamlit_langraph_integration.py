"""
Test suite for Streamlit-LangGraph integration (Phase 1.1)

Following TDD approach from STREAMLIT_CHAT_TDD_PLAN.md
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.research_agent.core.workflow import ResearchAgent


class TestResearchAgentInitialization:
    """Test ResearchAgent can be initialized in Streamlit context"""
    
    def test_research_agent_initialization(self):
        """Test ResearchAgent can be initialized in Streamlit context"""
        # Mock ES client
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        
        # Should be able to create ResearchAgent instance
        agent = ResearchAgent(
            es_client=mock_es_client,
            index_name="test-index",
            recursion_limit=10
        )
        
        assert agent is not None
        assert agent.es_client == mock_es_client
        assert agent.index_name == "test-index"
        assert agent.recursion_limit == 10
        assert agent.app is not None
    
    def test_research_agent_initialization_no_es_client(self):
        """Test ResearchAgent can be initialized without ES client"""
        agent = ResearchAgent()
        
        assert agent is not None
        assert agent.es_client is None
        assert agent.index_name == "research-publications-static"
        assert agent.recursion_limit == 50
        assert agent.app is not None


class TestAgentQueryExecution:
    """Test agent can process queries and return results"""
    
    @pytest.mark.asyncio
    async def test_agent_query_execution(self):
        """Test agent can process queries and return results"""
        # Mock ES client
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        
        # Create agent
        agent = ResearchAgent(es_client=mock_es_client)
        
        # Mock the app.ainvoke method
        mock_result = {
            "response": "Test response",
            "plan": ["Step 1", "Step 2"],
            "past_steps": [],
            "error": None
        }
        
        with patch.object(agent.app, 'ainvoke', new_callable=AsyncMock) as mock_ainvoke:
            mock_ainvoke.return_value = mock_result
            
            result = await agent.query("test query")
            
            assert result == mock_result
            assert mock_ainvoke.called
    
    @pytest.mark.asyncio
    async def test_agent_streaming_query(self):
        """Test agent can stream query results"""
        # Mock ES client
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        
        # Create agent
        agent = ResearchAgent(es_client=mock_es_client)
        
        # Mock streaming events
        mock_events = [
            {"planner": {"plan": ["Step 1", "Step 2"]}},
            {"agent": {"response": "Intermediate result"}},
            {"__end__": {"response": "Final result"}}
        ]
        
        async def mock_astream(state, config):
            for event in mock_events:
                yield event
        
        with patch.object(agent.app, 'astream', side_effect=mock_astream):
            events = []
            async for event in agent.stream_query("test query"):
                events.append(event)
            
            assert len(events) == 3
            assert events[0] == mock_events[0]
            assert events[1] == mock_events[1]
            assert events[2] == mock_events[2]


class TestSessionStateIntegration:
    """Test agent works with Streamlit session state"""
    
    def test_session_state_integration(self):
        """Test agent works with Streamlit session state"""
        # Mock streamlit session state
        mock_session_state = {}
        
        with patch('streamlit.session_state', mock_session_state):
            # Mock ES client
            mock_es_client = Mock()
            mock_es_client.ping.return_value = True
            
            # Create agent and store in session state
            agent = ResearchAgent(es_client=mock_es_client)
            mock_session_state['research_agent'] = agent
            
            # Verify agent is stored correctly
            assert 'research_agent' in mock_session_state
            assert mock_session_state['research_agent'] == agent
            assert isinstance(mock_session_state['research_agent'], ResearchAgent)
    
    def test_session_state_persistence(self):
        """Test agent instance persists across session state access"""
        # Mock streamlit session state
        mock_session_state = {}
        
        with patch('streamlit.session_state', mock_session_state):
            # Mock ES client
            mock_es_client = Mock()
            mock_es_client.ping.return_value = True
            
            # Create and store agent
            agent = ResearchAgent(es_client=mock_es_client)
            mock_session_state['research_agent'] = agent
            
            # Retrieve agent from session state
            retrieved_agent = mock_session_state['research_agent']
            
            # Should be the same instance
            assert retrieved_agent is agent
            assert id(retrieved_agent) == id(agent)


class TestStreamlitAgentBridge:
    """Test the StreamlitAgent bridge class"""
    
    def test_streamlit_agent_bridge_initialization(self):
        """Test StreamlitAgent bridge can be initialized"""
        # This will test the bridge class once we create it
        pass
    
    def test_streamlit_agent_bridge_query_processing(self):
        """Test StreamlitAgent bridge can process queries"""
        # This will test query processing through the bridge
        pass
    
    def test_streamlit_agent_bridge_error_handling(self):
        """Test StreamlitAgent bridge handles errors gracefully"""
        # This will test error handling in the bridge
        pass


# Fixtures for tests
@pytest.fixture
def mock_es_client():
    """Mock Elasticsearch client for testing"""
    client = Mock()
    client.ping.return_value = True
    return client


@pytest.fixture
def sample_query():
    """Sample query for testing"""
    return "How many papers has Christian Fager published?"


@pytest.fixture
def sample_agent_response():
    """Sample agent response for testing"""
    return {
        "response": "Christian Fager has published 42 papers.",
        "plan": [
            "Search for publications by Christian Fager",
            "Count the total number of results"
        ],
        "past_steps": [
            {"step": "Search for publications by Christian Fager", "result": "Found 42 papers"},
            {"step": "Count the total number of results", "result": "Total: 42"}
        ],
        "error": None,
        "total_results": 42
    }