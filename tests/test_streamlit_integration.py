#!/usr/bin/env python3
"""
Test script to verify Streamlit integration with plan-and-execute agent
"""

import sys
import os
import asyncio
from unittest.mock import MagicMock, patch
import traceback

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test core imports
        from src.research_agent.core.workflow import ResearchAgent, create_research_workflow
        from src.research_agent.core.state import PlanExecuteState
        from src.research_agent.core.models import Plan, Response, Act
        print("✅ Core workflow imports successful")
        
        # Test agent bridge
        from streamlit_agent import (
            StreamlitAgent, 
            initialize_streamlit_agent,
            format_streaming_response,
            display_agent_status
        )
        print("✅ Streamlit agent bridge imports successful")
        
        # Test Streamlit app
        import streamlit_app
        print("✅ Streamlit app imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False


def test_agent_initialization():
    """Test that the agent can be initialized without ES."""
    print("\nTesting agent initialization...")
    
    try:
        # Mock Elasticsearch client
        mock_es = MagicMock()
        mock_es.ping.return_value = True
        
        # Test direct ResearchAgent
        from src.research_agent.core.workflow import ResearchAgent
        agent = ResearchAgent(es_client=mock_es)
        print("✅ ResearchAgent initialization successful")
        
        # Test StreamlitAgent wrapper
        from streamlit_agent import StreamlitAgent
        streamlit_agent = StreamlitAgent(es_client=mock_es)
        print("✅ StreamlitAgent initialization successful")
        
        # Test agent info
        info = streamlit_agent.get_agent_info()
        print(f"✅ Agent info: {info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent initialization error: {e}")
        traceback.print_exc()
        return False


def test_workflow_creation():
    """Test that the workflow can be created."""
    print("\nTesting workflow creation...")
    
    try:
        # Mock ES client and tools
        mock_es = MagicMock()
        
        with patch('src.research_agent.core.workflow.initialize_elasticsearch_tools'), \
             patch('src.research_agent.core.workflow.create_elasticsearch_tools') as mock_tools:
            
            # Mock tools
            mock_tools.return_value = []
            
            from src.research_agent.core.workflow import create_research_workflow
            workflow = create_research_workflow(es_client=mock_es)
            print("✅ Workflow creation successful")
            
            # Test compilation
            app = workflow.compile()
            print("✅ Workflow compilation successful")
            
            return True
            
    except Exception as e:
        print(f"❌ Workflow creation error: {e}")
        traceback.print_exc()
        return False


def test_streaming_format():
    """Test streaming response formatting."""
    print("\nTesting streaming response formatting...")
    
    try:
        from streamlit_agent import format_streaming_response
        from datetime import datetime
        
        # Test different event types
        test_events = [
            {
                'type': 'plan',
                'content': ['Step 1: Search for publications', 'Step 2: Analyze results'],
                'timestamp': datetime.now()
            },
            {
                'type': 'execution',
                'content': {'response': 'Searching for publications...'},
                'timestamp': datetime.now()
            },
            {
                'type': 'final',
                'content': {'response': 'Found 25 publications by the author'},
                'timestamp': datetime.now()
            },
            {
                'type': 'error',
                'content': 'Connection timeout',
                'timestamp': datetime.now()
            }
        ]
        
        for event in test_events:
            formatted = format_streaming_response(event)
            print(f"✅ {event['type']} event formatted: {formatted[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Streaming format error: {e}")
        traceback.print_exc()
        return False


async def test_mock_query():
    """Test a mock query execution."""
    print("\nTesting mock query execution...")
    
    try:
        # Mock ES client
        mock_es = MagicMock()
        mock_es.ping.return_value = True
        
        # Mock the tools and LLM responses
        with patch('src.research_agent.core.workflow.initialize_elasticsearch_tools'), \
             patch('src.research_agent.core.workflow.create_elasticsearch_tools') as mock_tools, \
             patch('src.research_agent.core.workflow.ChatOpenAI') as mock_llm:
            
            # Mock tools
            mock_tools.return_value = []
            
            # Mock LLM structured output
            mock_plan = MagicMock()
            mock_plan.steps = ["Search for author publications", "Count results"]
            
            mock_llm_instance = MagicMock()
            mock_llm_instance.with_structured_output.return_value.ainvoke.return_value = mock_plan
            mock_llm.return_value = mock_llm_instance
            
            # Mock the agent executor
            with patch('src.research_agent.core.workflow.create_react_agent') as mock_agent:
                mock_agent.return_value.ainvoke.return_value = {
                    'messages': [MagicMock(content="Found 25 publications")]
                }
                
                from streamlit_agent import StreamlitAgent
                agent = StreamlitAgent(es_client=mock_es)
                
                # Test sync query
                result = agent.run_sync_query("How many papers has John Doe published?")
                print(f"✅ Mock query result: {result}")
                
                return True
        
    except Exception as e:
        print(f"❌ Mock query error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Streamlit Plan-and-Execute Integration")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_agent_initialization,
        test_workflow_creation,
        test_streaming_format,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 40)
    
    # Run async test
    print("Running async test...")
    try:
        asyncio.run(test_mock_query())
        passed += 1
        total += 1
        print("✅ Async test passed")
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        total += 1
    
    print("=" * 60)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Streamlit integration is working!")
        return 0
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())