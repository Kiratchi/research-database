"""
Streamlit Chat Agent for Research Publications

A locally hosted chat interface for querying research publications
using natural language with LangGraph ResearchAgent backend.
"""

import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import traceback
import asyncio
import json

# Import our new streamlit agent bridge
from streamlit_agent import (
    get_streamlit_agent, 
    initialize_streamlit_agent,
    format_streaming_response,
    display_agent_status
)

# Import ES tools for statistics
from src.research_agent.tools.elasticsearch_tools import get_statistics_summary

# Configure Streamlit page
st.set_page_config(
    page_title="Research Publications Chat Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent_initialized' not in st.session_state:
    st.session_state.agent_initialized = False
if 'es_client' not in st.session_state:
    st.session_state.es_client = None
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False


@st.cache_resource
def initialize_elasticsearch():
    """Initialize Elasticsearch client (cached for performance)."""
    try:
        # Load environment variables
        load_dotenv(dotenv_path=".env", override=True)
        
        # Initialize Elasticsearch client
        es = Elasticsearch(
            hosts=[os.getenv('ES_HOST')],
            http_auth=(os.getenv('ES_USER'), os.getenv('ES_PASS')),
            verify_certs=False
        )
        
        # Test connection
        if not es.ping():
            st.error("❌ Could not connect to Elasticsearch")
            return None
        
        return es
        
    except Exception as e:
        st.error(f"❌ Error initializing Elasticsearch: {str(e)}")
        if st.session_state.debug_mode:
            st.code(traceback.format_exc())
        return None


def format_agent_response(response_data: dict) -> str:
    """
    Format agent response for display in Streamlit.
    
    Args:
        response_data: Response from StreamlitAgent
        
    Returns:
        Formatted string for display
    """
    if not response_data.get('success', False):
        return f"❌ Error: {response_data.get('error', 'Unknown error')}"
    
    response = response_data.get('response', 'No response generated')
    metadata = response_data.get('metadata', {})
    
    # Format the main response
    formatted_response = f"{response}\n\n"
    
    # Add plan information if available
    plan = metadata.get('plan', [])
    if plan:
        formatted_response += "📋 **Plan:**\n"
        for i, step in enumerate(plan, 1):
            formatted_response += f"  {i}. {step}\n"
        formatted_response += "\n"
    
    # Add execution steps if available
    past_steps = metadata.get('past_steps', [])
    if past_steps:
        formatted_response += "🔍 **Execution Steps:**\n"
        for i, step in enumerate(past_steps, 1):
            step_desc = step.get('step', 'Unknown step')
            step_result = step.get('result', 'No result')
            formatted_response += f"  {i}. {step_desc} → {step_result}\n"
        formatted_response += "\n"
    
    # Add result statistics if available
    total_results = metadata.get('total_results')
    if total_results is not None:
        formatted_response += f"📊 **Results:** {total_results}\n"
    
    return formatted_response


def display_database_stats(es_client):
    """Display database statistics in sidebar."""
    try:
        # Initialize tools with ES client
        from src.research_agent.tools.elasticsearch_tools import initialize_elasticsearch_tools
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        # Get statistics
        stats = get_statistics_summary()
        
        st.sidebar.markdown(f"""
        ### 📊 Database Stats
        - **Total Publications:** {stats.get('total_publications', 'N/A'):,}
        - **Latest Year:** {stats.get('latest_year', 'N/A')}
        - **Most Common Type:** {stats.get('most_common_type', 'N/A')}
        - **Authors:** {stats.get('total_authors', 'N/A'):,}
        """)
        
    except Exception as e:
        st.sidebar.error(f"Could not load database stats: {e}")
        if st.session_state.debug_mode:
            st.sidebar.code(traceback.format_exc())


def display_debug_panel(event_data=None, response_data=None):
    """
    Display comprehensive debug information.
    
    Args:
        event_data: Streaming event data
        response_data: Final response data
    """
    with st.expander("🔧 Debug Information", expanded=st.session_state.debug_mode):
        debug_tabs = st.tabs(["Response Data", "Event Stream", "System Info", "Raw Data"])
        
        with debug_tabs[0]:
            if response_data:
                st.subheader("Response Analysis")
                st.json({
                    "success": response_data.get('success', False),
                    "error": response_data.get('error'),
                    "response_length": len(response_data.get('response', '')),
                    "metadata": response_data.get('metadata', {}),
                    "has_plan": bool(response_data.get('metadata', {}).get('plan')),
                    "plan_steps": len(response_data.get('metadata', {}).get('plan', [])),
                    "execution_steps": len(response_data.get('metadata', {}).get('past_steps', []))
                })
                
                if response_data.get('traceback'):
                    st.subheader("Error Traceback")
                    st.code(response_data['traceback'])
        
        with debug_tabs[1]:
            if event_data:
                st.subheader("Event Stream")
                for i, event in enumerate(event_data):
                    st.write(f"**Event {i+1}:**")
                    st.json({
                        "type": event.get('type'),
                        "timestamp": str(event.get('timestamp')),
                        "content_type": type(event.get('content', '')).__name__,
                        "content_preview": str(event.get('content', ''))[:200] + "..." if len(str(event.get('content', ''))) > 200 else str(event.get('content', ''))
                    })
                    st.write("---")
        
        with debug_tabs[2]:
            st.subheader("System Information")
            if 'streamlit_agent' in st.session_state:
                agent = st.session_state.streamlit_agent
                agent_info = agent.get_agent_info()
                st.json({
                    "agent_initialized": agent_info.get('initialized', False),
                    "elasticsearch_connected": agent_info.get('es_client_connected', False),
                    "index_name": agent_info.get('index_name', 'N/A'),
                    "recursion_limit": agent_info.get('recursion_limit', 'N/A'),
                    "session_state_keys": list(st.session_state.keys()),
                    "messages_count": len(st.session_state.messages),
                    "debug_mode": st.session_state.debug_mode
                })
            else:
                st.warning("No agent in session state")
        
        with debug_tabs[3]:
            st.subheader("Raw Response Data")
            if response_data:
                st.code(json.dumps(response_data, indent=2, default=str))
            if event_data:
                st.subheader("Raw Event Data")
                st.code(json.dumps(event_data, indent=2, default=str))


def process_query_with_streaming(agent, query: str):
    """Process query with streaming updates and debug info."""
    # Create placeholders for streaming updates
    streaming_placeholder = st.empty()
    final_placeholder = st.empty()
    
    # Container for streaming updates and debug data
    streaming_updates = []
    event_data = []
    final_response = None
    
    try:
        # Run the streaming query
        async def stream_query():
            nonlocal final_response
            async for event in agent.stream_query(query):
                # Store event for debugging
                event_data.append(event)
                
                # Format and display streaming update
                formatted_event = format_streaming_response(event)
                streaming_updates.append(formatted_event)
                
                # Update the streaming display
                streaming_content = "\n\n".join(streaming_updates)
                streaming_placeholder.markdown(streaming_content)
                
                # If this is the final result, store it
                if event.get('type') == 'final':
                    final_response = event.get('content', {})
                    if isinstance(final_response, dict) and 'response' in final_response:
                        final_placeholder.markdown(f"**Final Answer:** {final_response['response']}")
        
        # Run the async streaming in sync context
        asyncio.run(stream_query())
        
        # Display debug information
        if st.session_state.debug_mode:
            # Create mock response data for debug panel
            response_data = {
                'success': True,
                'response': final_response.get('response', 'No response') if final_response else 'No final response',
                'metadata': final_response if final_response else {},
                'streaming_events': len(event_data),
                'streaming_updates': len(streaming_updates)
            }
            display_debug_panel(event_data=event_data, response_data=response_data)
        
        return "\n\n".join(streaming_updates)
        
    except Exception as e:
        error_msg = f"❌ Error during streaming: {str(e)}"
        streaming_placeholder.error(error_msg)
        
        # Show error in debug panel
        if st.session_state.debug_mode:
            error_response = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            display_debug_panel(response_data=error_response)
        
        return error_msg


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🔍 Research Publications Chat Agent")
    st.markdown("Ask questions about research publications in natural language!")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("""
        ### How to use:
        - Ask questions like "How many papers has Christian Fager published?"
        - Use author names, years, topics, or journals
        - The system uses LangGraph ResearchAgent with plan-and-execute
        
        ### Examples:
        - "List all articles by Anna Dubois"
        - "Find papers about machine learning from 2023"
        - "How many publications in Nature?"
        - "Compare publication counts between authors"
        """)
        
        # Debug controls
        st.header("🔧 Debug Options")
        st.session_state.debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
        
        if st.session_state.debug_mode:
            st.info("Debug mode enabled - detailed information will be shown")
        
        # Streaming control
        streaming_enabled = st.checkbox("Enable Streaming", value=True)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        # System status
        st.header("📊 System Status")
        if st.session_state.agent_initialized:
            st.success("✅ Agent Ready")
        else:
            st.warning("⚠️ Initializing...")
    
    # Initialize Elasticsearch and agent
    if not st.session_state.agent_initialized:
        with st.spinner("Initializing Research Agent..."):
            # Initialize Elasticsearch
            es_client = initialize_elasticsearch()
            
            if es_client is not None:
                # Initialize the StreamlitAgent
                agent = initialize_streamlit_agent(es_client)
                
                if agent.is_initialized():
                    st.session_state.agent_initialized = True
                    st.success("✅ Research Agent initialized successfully!")
                    
                    # Display database stats
                    display_database_stats(es_client)
                    
                else:
                    st.error("❌ Failed to initialize Research Agent.")
                    st.stop()
            else:
                st.error("❌ Failed to initialize Elasticsearch. Please check your configuration.")
                st.stop()
    
    # Get agent from session state
    if st.session_state.agent_initialized:
        agent = get_streamlit_agent()
        
        # Display agent status in sidebar
        display_agent_status(agent)
        
        # Chat interface
        st.markdown("### Chat Interface")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show debug info if available
                if message["role"] == "assistant" and "debug_info" in message:
                    display_debug_panel(response_data=message["debug_info"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about research publications..."):
            
            # Add user message to chat history
            st.session_state.messages.append({
                "role": "user", 
                "content": prompt,
                "timestamp": datetime.now()
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Process the query
            with st.chat_message("assistant"):
                with st.spinner("Processing query..."):
                    # Use streaming or synchronous processing
                    if streaming_enabled:
                        response_content = process_query_with_streaming(agent, prompt)
                        debug_info = None  # Debug info already shown in streaming
                    else:
                        # Synchronous processing
                        result = agent.run_sync_query(prompt)
                        response_content = format_agent_response(result)
                        st.markdown(response_content)
                        debug_info = result
                        
                        # Show debug info for sync queries
                        if st.session_state.debug_mode:
                            display_debug_panel(response_data=result)
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_content,
                    "timestamp": datetime.now(),
                    "debug_info": debug_info
                })


if __name__ == "__main__":
    main()