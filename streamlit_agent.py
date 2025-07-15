"""
Streamlit-LangGraph Bridge for Research Publications Chat

This module provides a bridge between Streamlit and the LangGraph ResearchAgent,
handling async operations, session state, and streaming updates.
"""

import asyncio
import streamlit as st
from typing import Dict, Any, Optional, AsyncIterator
from datetime import datetime
import traceback
from elasticsearch import Elasticsearch

from src.research_agent.core.workflow import ResearchAgent


class StreamlitAgent:
    """
    Bridge between Streamlit and ResearchAgent that handles:
    - Async operations in Streamlit context
    - Session state management
    - Streaming updates
    - Error handling
    """
    
    def __init__(self, es_client: Optional[Elasticsearch] = None, 
                 index_name: str = "research-publications-static"):
        """
        Initialize the Streamlit agent bridge.
        
        Args:
            es_client: Elasticsearch client instance
            index_name: Name of the publications index
        """
        self.es_client = es_client
        self.index_name = index_name
        self.research_agent = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the ResearchAgent."""
        try:
            self.research_agent = ResearchAgent(
                es_client=self.es_client,
                index_name=self.index_name,
                recursion_limit=50
            )
        except Exception as e:
            st.error(f"Failed to initialize ResearchAgent: {str(e)}")
            self.research_agent = None
    
    def is_initialized(self) -> bool:
        """Check if the agent is properly initialized."""
        return self.research_agent is not None
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the ResearchAgent.
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary containing result and metadata
        """
        if not self.research_agent:
            return {
                'success': False,
                'error': 'ResearchAgent not initialized',
                'response': None,
                'metadata': {}
            }
        
        try:
            # Execute the query
            result = await self.research_agent.query(query)
            
            return {
                'success': True,
                'error': None,
                'response': result.get('response', 'No response generated'),
                'metadata': {
                    'plan': result.get('plan', []),
                    'past_steps': result.get('past_steps', []),
                    'total_results': result.get('total_results'),
                    'current_step': result.get('current_step', 0),
                    'session_id': result.get('session_id')
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response': None,
                'metadata': {},
                'traceback': traceback.format_exc()
            }
    
    async def stream_query(self, query: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream a query execution with real-time updates.
        
        Args:
            query: Natural language query
            
        Yields:
            Dictionary containing streaming updates
        """
        if not self.research_agent:
            yield {
                'type': 'error',
                'content': 'ResearchAgent not initialized',
                'timestamp': datetime.now()
            }
            return
        
        try:
            async for event in self.research_agent.stream_query(query):
                # Process different types of events
                for node_name, node_data in event.items():
                    if node_name == "__end__":
                        yield {
                            'type': 'final',
                            'content': node_data,
                            'timestamp': datetime.now()
                        }
                    elif node_name == "planner":
                        yield {
                            'type': 'plan',
                            'content': node_data.get('plan', []),
                            'timestamp': datetime.now()
                        }
                    elif node_name == "agent":
                        yield {
                            'type': 'execution',
                            'content': node_data,
                            'timestamp': datetime.now()
                        }
                    elif node_name == "replan":
                        yield {
                            'type': 'replan',
                            'content': node_data,
                            'timestamp': datetime.now()
                        }
                    else:
                        yield {
                            'type': 'step',
                            'content': node_data,
                            'node': node_name,
                            'timestamp': datetime.now()
                        }
                        
        except Exception as e:
            yield {
                'type': 'error',
                'content': str(e),
                'timestamp': datetime.now(),
                'traceback': traceback.format_exc()
            }
    
    def run_sync_query(self, query: str) -> Dict[str, Any]:
        """
        Run a query synchronously using asyncio.
        
        Args:
            query: Natural language query
            
        Returns:
            Query result
        """
        try:
            # Create new event loop if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async query
            return loop.run_until_complete(self.process_query(query))
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Error running sync query: {str(e)}",
                'response': None,
                'metadata': {},
                'traceback': traceback.format_exc()
            }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the agent state.
        
        Returns:
            Dictionary containing agent information
        """
        if not self.research_agent:
            return {
                'initialized': False,
                'error': 'ResearchAgent not initialized'
            }
        
        return {
            'initialized': True,
            'es_client_connected': self.es_client is not None and self.es_client.ping() if self.es_client else False,
            'index_name': self.index_name,
            'recursion_limit': self.research_agent.recursion_limit
        }


def get_streamlit_agent() -> StreamlitAgent:
    """
    Get or create a StreamlitAgent instance using Streamlit session state.
    
    Returns:
        StreamlitAgent instance
    """
    # Check if agent already exists in session state
    if 'streamlit_agent' not in st.session_state:
        # Initialize ES client if not already done
        es_client = None
        if 'es_client' in st.session_state:
            es_client = st.session_state.es_client
        
        # Create new agent
        agent = StreamlitAgent(es_client=es_client)
        st.session_state.streamlit_agent = agent
        
        return agent
    
    return st.session_state.streamlit_agent


def initialize_streamlit_agent(es_client: Elasticsearch, 
                              index_name: str = "research-publications-static") -> StreamlitAgent:
    """
    Initialize a StreamlitAgent with specific ES client and store in session state.
    
    Args:
        es_client: Elasticsearch client instance
        index_name: Name of the publications index
        
    Returns:
        Initialized StreamlitAgent
    """
    agent = StreamlitAgent(es_client=es_client, index_name=index_name)
    
    # Store in session state
    st.session_state.streamlit_agent = agent
    st.session_state.es_client = es_client
    
    return agent


def format_streaming_response(event: Dict[str, Any]) -> str:
    """
    Format streaming event for display in Streamlit.
    
    Args:
        event: Streaming event from StreamlitAgent
        
    Returns:
        Formatted string for display
    """
    event_type = event.get('type', 'unknown')
    timestamp = event.get('timestamp', datetime.now())
    
    if event_type == 'plan':
        plan_steps = event.get('content', [])
        if plan_steps:
            formatted_steps = "\n".join([f"  {i+1}. {step}" for i, step in enumerate(plan_steps)])
            return f"📋 **Plan Generated:**\n{formatted_steps}"
        return "📋 **Plan Generated** (empty)"
    
    elif event_type == 'execution':
        content = event.get('content', {})
        if isinstance(content, dict):
            if 'response' in content:
                return f"🔍 **Executing:** {content['response']}"
            return f"🔍 **Executing step...**"
        return f"🔍 **Executing:** {content}"
    
    elif event_type == 'replan':
        return "🔄 **Replanning based on results...**"
    
    elif event_type == 'final':
        content = event.get('content', {})
        if isinstance(content, dict) and 'response' in content:
            return f"✅ **Final Result:** {content['response']}"
        return f"✅ **Final Result:** {content}"
    
    elif event_type == 'error':
        error_msg = event.get('content', 'Unknown error')
        return f"❌ **Error:** {error_msg}"
    
    elif event_type == 'step':
        node = event.get('node', 'unknown')
        content = event.get('content', {})
        return f"⚡ **{node.title()}:** {content}"
    
    else:
        return f"ℹ️ **{event_type.title()}:** {event.get('content', 'No content')}"


def display_agent_status(agent: StreamlitAgent):
    """
    Display agent status information in Streamlit sidebar.
    
    Args:
        agent: StreamlitAgent instance
    """
    info = agent.get_agent_info()
    
    if info['initialized']:
        st.sidebar.success("✅ ResearchAgent Initialized")
        if info.get('es_client_connected'):
            st.sidebar.success("✅ Elasticsearch Connected")
        else:
            st.sidebar.warning("⚠️ Elasticsearch Not Connected")
        
        st.sidebar.info(f"Index: {info['index_name']}")
        st.sidebar.info(f"Recursion Limit: {info['recursion_limit']}")
    else:
        st.sidebar.error("❌ ResearchAgent Not Initialized")
        if 'error' in info:
            st.sidebar.error(f"Error: {info['error']}")