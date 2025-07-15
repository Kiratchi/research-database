"""
Test configuration and fixtures for the research agent.
"""

import pytest
import os
import sys
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List

# Add src to path for tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from research_agent.core.models import Plan, Response, Act
from research_agent.core.state import PlanExecuteState
from research_agent.utils.config import ElasticsearchConfig, OpenAIConfig, AgentConfig


@pytest.fixture
def mock_es_client():
    """Mock Elasticsearch client."""
    client = Mock()
    client.ping.return_value = True
    client.search.return_value = {
        "hits": {
            "total": {"value": 100},
            "hits": [
                {
                    "_source": {
                        "Title": "Test Paper",
                        "authors": ["Test Author"],
                        "Year": 2023
                    }
                }
            ]
        },
        "aggregations": {
            "years": {
                "buckets": [
                    {"key": "2023", "doc_count": 50},
                    {"key": "2022", "doc_count": 30}
                ]
            }
        }
    }
    return client


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI response."""
    response = Mock()
    response.content = '{"steps": ["Search for publications", "Count results"]}'
    return response


@pytest.fixture
def sample_plan() -> Plan:
    """Sample Plan object."""
    return Plan(steps=[
        "Search for publications by Christian Fager",
        "Count the total number of publications"
    ])


@pytest.fixture
def sample_response() -> Response:
    """Sample Response object."""
    return Response(response="Christian Fager has published 25 papers.")


@pytest.fixture
def sample_act_plan() -> Act:
    """Sample Act object with Plan."""
    return Act(action=Plan(steps=[
        "Search for more publications",
        "Analyze the results"
    ]))


@pytest.fixture
def sample_act_response() -> Act:
    """Sample Act object with Response."""
    return Act(action=Response(response="The analysis is complete."))


@pytest.fixture
def sample_state() -> PlanExecuteState:
    """Sample PlanExecuteState."""
    return {
        "input": "How many papers has Christian Fager published?",
        "plan": [
            "Search for publications by Christian Fager",
            "Count the total number of publications"
        ],
        "past_steps": [
            ("Search for publications by Christian Fager", "Found 25 publications")
        ],
        "response": None,
        "session_id": "test-session-123",
        "total_results": 25,
        "current_step": 1,
        "error": None
    }


@pytest.fixture
def sample_config():
    """Sample configuration."""
    return {
        "elasticsearch": ElasticsearchConfig(
            host="localhost:9200",
            username="test",
            password="test",
            index_name="test-index"
        ),
        "openai": OpenAIConfig(
            api_key="test-key",
            model_name="gpt-4o",
            temperature=0.0
        ),
        "agent": AgentConfig(
            recursion_limit=10,
            max_steps=5,
            timeout=60
        )
    }


@pytest.fixture
def mock_tool_results():
    """Mock tool execution results."""
    return {
        "search_publications": {
            "total_results": 100,
            "session_id": "test-session-123",
            "sample_results": [
                {
                    "_source": {
                        "Title": "Machine Learning Paper",
                        "authors": ["ML Author"],
                        "Year": 2023
                    }
                }
            ],
            "aggregations": {"years": [], "types": []}
        },
        "search_by_author": {
            "total_results": 25,
            "session_id": "test-session-456",
            "sample_results": [
                {
                    "_source": {
                        "Title": "Author Paper",
                        "authors": ["Christian Fager"],
                        "Year": 2023
                    }
                }
            ]
        }
    }


@pytest.fixture
def mock_agent_executor():
    """Mock agent executor."""
    executor = Mock()
    executor.ainvoke.return_value = {
        "messages": [
            Mock(content="Search completed successfully. Found 25 publications by Christian Fager.")
        ]
    }
    return executor


@pytest.fixture
def mock_planner():
    """Mock planner."""
    planner = Mock()
    planner.ainvoke.return_value = Plan(steps=[
        "Search for publications by Christian Fager",
        "Count the total number of publications"
    ])
    return planner


@pytest.fixture
def mock_replanner():
    """Mock replanner."""
    replanner = Mock()
    replanner.ainvoke.return_value = Act(action=Response(
        response="Christian Fager has published 25 papers."
    ))
    return replanner


@pytest.fixture
def mock_elasticsearch_tools():
    """Mock Elasticsearch tools."""
    tools = []
    
    # Mock search_publications tool
    search_tool = Mock()
    search_tool.name = "search_publications"
    search_tool.description = "Search for publications"
    search_tool.func = Mock(return_value="Search results: 100 publications found")
    tools.append(search_tool)
    
    # Mock search_by_author tool
    author_tool = Mock()
    author_tool.name = "search_by_author"
    author_tool.description = "Search by author"
    author_tool.func = Mock(return_value="Author results: 25 publications found")
    tools.append(author_tool)
    
    return tools


# Test markers
pytestmark = pytest.mark.asyncio


# Helper functions
def create_mock_state(**kwargs) -> PlanExecuteState:
    """Create a mock state with default values."""
    default_state = {
        "input": "test query",
        "plan": [],
        "past_steps": [],
        "response": None,
        "session_id": None,
        "total_results": None,
        "current_step": 0,
        "error": None
    }
    default_state.update(kwargs)
    return default_state


def assert_plan_structure(plan: Plan):
    """Assert that a plan has the correct structure."""
    assert isinstance(plan, Plan)
    assert hasattr(plan, 'steps')
    assert isinstance(plan.steps, list)
    assert len(plan.steps) > 0
    assert all(isinstance(step, str) for step in plan.steps)


def assert_response_structure(response: Response):
    """Assert that a response has the correct structure."""
    assert isinstance(response, Response)
    assert hasattr(response, 'response')
    assert isinstance(response.response, str)
    assert len(response.response) > 0


def assert_state_structure(state: PlanExecuteState):
    """Assert that a state has the correct structure."""
    required_keys = ["input", "plan", "past_steps", "response"]
    for key in required_keys:
        assert key in state
    
    assert isinstance(state["input"], str)
    assert isinstance(state["plan"], list)
    assert isinstance(state["past_steps"], list)


# Environment setup
def setup_test_environment():
    """Set up test environment variables."""
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["ES_HOST"] = "localhost:9200"
    os.environ["ES_INDEX"] = "test-index"


# Cleanup
def cleanup_test_environment():
    """Clean up test environment."""
    test_vars = ["OPENAI_API_KEY", "ES_HOST", "ES_INDEX"]
    for var in test_vars:
        if var in os.environ:
            del os.environ[var]


# Auto-setup for tests
@pytest.fixture(autouse=True)
def setup_and_cleanup():
    """Automatically setup and cleanup for each test."""
    setup_test_environment()
    yield
    cleanup_test_environment()


# Legacy fixtures for backward compatibility
@pytest.fixture(scope="session")
def es():
    """Legacy ES fixture for backward compatibility."""
    from elasticsearch import Elasticsearch
    return Elasticsearch(
        os.getenv("ES_HOST", "localhost:9200"),
        basic_auth=(os.getenv("ES_USER"), os.getenv("ES_PASS")),
        verify_certs=False,
    )


@pytest.fixture(autouse=True, scope="session")
def init_tools(es):
    """Legacy tools initialization for backward compatibility."""
    try:
        # Import legacy tools if they exist
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from agent_tools import initialize_tools
        initialize_tools(es)
    except ImportError:
        # Skip if legacy tools don't exist
        pass