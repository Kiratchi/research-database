"""
Streamlit-LangGraph Bridge for Research Publications Chat

This module provides a bridge between Streamlit and the LangGraph ResearchAgent,
handling async operations, session state, and streaming updates.
"""

import asyncio
import streamlit as st
from typing import Dict, Any, Optional, AsyncIterator, List
from datetime import datetime
import traceback
from elasticsearch import Elasticsearch

from src.research_agent.core.workflow import ResearchAgent
from src.research_agent.core.hybrid_router import HybridRouter


class StreamlitAgent:
    """
    Bridge between Streamlit and HybridRouter that handles:
    - Async operations in Streamlit context
    - Session state management
    - Streaming updates
    - Error handling
    - Query classification and routing
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
        self.hybrid_router = None
        self.research_agent = None  # Keep for backward compatibility
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the HybridRouter and ResearchAgent."""
        try:
            # Initialize hybrid router
            self.hybrid_router = HybridRouter(
                es_client=self.es_client,
                index_name=self.index_name
            )
            
            # Initialize legacy research agent for backward compatibility
            self.research_agent = ResearchAgent(
                es_client=self.es_client,
                index_name=self.index_name,
                recursion_limit=50
            )
        except Exception as e:
            st.error(f"Failed to initialize agents: {str(e)}")
            self.hybrid_router = None
            self.research_agent = None
    
    def is_initialized(self) -> bool:
        """Check if the agent is properly initialized."""
        return self.hybrid_router is not None and self.hybrid_router.is_initialized()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation memory state."""
        if self.hybrid_router:
            return self.hybrid_router.get_memory_summary()
        return {"total_messages": 0, "user_messages": 0, "ai_messages": 0, "memory_buffer": None}
    
    def clear_memory(self):
        """Clear the conversation memory."""
        if self.hybrid_router:
            self.hybrid_router.clear_memory()
    
    def get_conversation_memory(self):
        """Get direct access to the conversation memory instance."""
        if self.hybrid_router:
            return self.hybrid_router.get_conversation_memory()
        return None
    
    def process_query(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Process a query through the HybridRouter with LangChain Memory.
        
        Args:
            query: Natural language query
            conversation_history: Optional conversation history for initialization (deprecated - use LangChain memory instead)
            
        Returns:
            Dictionary containing result and metadata
        """
        if not self.hybrid_router:
            return {
                'success': False,
                'error': 'HybridRouter not initialized',
                'response': None,
                'metadata': {}
            }
        
        try:
            # Use hybrid router for intelligent query processing
            # conversation_history is mainly for backwards compatibility and initial memory setup
            result = self.hybrid_router.process_query(query, conversation_history)
            
            # Add memory information to result metadata
            if result.get('success', False):
                memory_summary = self.get_memory_summary()
                result.setdefault('metadata', {})['memory_summary'] = memory_summary
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response': None,
                'metadata': {},
                'traceback': traceback.format_exc()
            }
    
    async def stream_query(self, query: str, conversation_history: Optional[List[Dict]] = None) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream a query execution with real-time updates and LangChain Memory.
        
        Args:
            query: Natural language query
            conversation_history: Optional conversation history for initialization (deprecated - use LangChain memory instead)
            
        Yields:
            Dictionary containing streaming updates
        """
        if not self.hybrid_router:
            yield {
                'type': 'error',
                'content': 'HybridRouter not initialized',
                'timestamp': datetime.now()
            }
            return
        
        try:
            # Stream through hybrid router with memory support
            async for event in self.hybrid_router.stream_query(query, conversation_history):
                # Add memory information to final events
                if event.get('type') == 'final':
                    memory_summary = self.get_memory_summary()
                    event.setdefault('metadata', {})['memory_summary'] = memory_summary
                yield event
                        
        except Exception as e:
            yield {
                'type': 'error',
                'content': str(e),
                'timestamp': datetime.now(),
                'traceback': traceback.format_exc()
            }
    
    def run_sync_query(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Run a query synchronously using the hybrid router with LangChain Memory.
        
        Args:
            query: Natural language query
            conversation_history: Optional conversation history for initialization (deprecated - use LangChain memory instead)
            
        Returns:
            Query result
        """
        try:
            # Direct sync query through hybrid router
            return self.process_query(query, conversation_history)
            
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
        if not self.hybrid_router:
            return {
                'initialized': False,
                'error': 'HybridRouter not initialized'
            }
        
        router_info = self.hybrid_router.get_router_info()
        
        return {
            'initialized': router_info['initialized'],
            'es_client_connected': router_info['es_client_connected'],
            'index_name': router_info['index_name'],
            'hybrid_router_enabled': True,
            'research_agent_initialized': router_info['research_agent_initialized'],
            'performance_stats': router_info['performance_stats'],
            'recursion_limit': 50  # Default value
        }
    
    def get_processing_message(self, query: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Get appropriate processing message for user feedback.
        
        Args:
            query: Natural language query
            conversation_history: Recent conversation history for context
            
        Returns:
            Processing message string
        """
        if not self.hybrid_router:
            return "🔍 Processing query..."
        
        return self.hybrid_router.get_processing_message(query, conversation_history)


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
        
        # Handle different node types with proper formatting
        if node == 'complete':
            # Extract the response from the complete step
            if isinstance(content, dict) and 'response' in content:
                return f"**Final Answer:**\n\n{content['response']}"
            else:
                return f"**Final Answer:**\n\n{content}"
        elif node == 'replan':
            # Format replan content
            if isinstance(content, dict):
                if 'response' in content:
                    return f"**Final Answer:**\n\n{content['response']}"
                elif 'plan' in content:
                    steps = content['plan']
                    if steps:
                        formatted_steps = "\n".join([f"  {i+1}. {step}" for i, step in enumerate(steps)])
                        return f"🔄 **Replanning:**\n{formatted_steps}"
                    return "🔄 **Replanning** (empty)"
            return f"🔄 **Replan:** {content}"
        else:
            # For other nodes, display as is
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