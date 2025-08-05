"""
Cleaned ReAct Workflow - Removed plan-execute artifacts
Removed: Complex state schema, verbose comments, plan-execute references
Kept: ReAct execution, conversation context, research quality
"""

from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import HumanMessage
import os
import uuid
import time
from dotenv import load_dotenv

# Import tools
from research_agent.tools import get_all_tools

# =============================================================================
# CLEANED STATE SCHEMA
# =============================================================================

class ReActState(TypedDict):
    """Clean ReAct state schema."""
    input: str
    response: Optional[str]
    session_id: str

# =============================================================================
# CONTEXT HELPER
# =============================================================================

def get_conversation_context(memory_manager, session_id: str, max_length: int = 2000) -> str:
    """Get conversation context with proper length management."""
    if not session_id or not memory_manager:
        return "No previous conversation context available."
    
    try:
        history = memory_manager.get_conversation_history_for_state(session_id)
        if not history:
            return "No previous conversation context available."
        
        # Build context from most recent messages
        context_parts = []
        current_length = 0
        
        for msg in reversed(history):
            role = msg["role"].title()
            content = msg["content"]
            
            new_part = f"- {role}: {content}\n"
            new_length = current_length + len(new_part)
            
            if new_length > max_length:
                remaining_space = max_length - current_length - len(f"- {role}: ") - 20
                if remaining_space > 100:
                    truncated_content = content[:remaining_space] + "..."
                    context_parts.insert(0, f"- {role}: {truncated_content}")
                break
            
            context_parts.insert(0, new_part.strip())
            current_length = new_length
        
        context = "\n".join(context_parts)
        print(f"Context built: {len(context)} chars from {len(context_parts)} messages")
        return context
        
    except Exception as e:
        print(f"Error getting conversation context: {e}")
        return "Error retrieving conversation context."

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

def create_react_llm(session_id: str = None) -> ChatLiteLLM:
    """Create LLM for ReAct agent."""
    
    config = {
        "model": "anthropic/claude-sonnet-3.7",
        "temperature": 0.1,
        "max_tokens": 4000,
    }
    
    metadata = {
        "component": "react_llm",
        "model_name": config["model"],
        "purpose": "react_execution"
    }
    
    if session_id:
        metadata.update({
            "session_id": session_id,
            "session_group": f"research-session-{session_id}"
        })
    
    try:
        llm = ChatLiteLLM(
            model=config["model"],
            api_key=os.getenv("LITELLM_API_KEY"),
            api_base=os.getenv("LITELLM_BASE_URL"),
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            timeout=60,
            metadata=metadata
        )
        
        return llm
        
    except Exception as e:
        raise e

# =============================================================================
# REACT PROMPT TEMPLATE
# =============================================================================

REACT_PROMPT_TEMPLATE = """You are an AI research assistant specializing in the Chalmers University research database with access to powerful research tools.

**RESEARCH QUESTION:** {query}

**CONVERSATION CONTEXT:**
{conversation_context}

**DATABASE SCOPE & LIMITATIONS:**
This database covers research from Chalmers University of Technology, Sweden. Key limitations:
- **Author scope**: Only includes publications from when authors worked at Chalmers or collaborated with Chalmers researchers
- **Temporal coverage**: Grant projects comprehensive from 2012+, student theses from 2009+
- **Publication definition**: Includes journal articles, conference papers, reports, theses, and other research outputs
- **Access limitations**: Items registered but full-text may not always be available
- **Institution-specific**: Excludes research from other universities
- **Content gaps**: May lack pre-prints, industry reports, software/datasets
- **Data freshness**: Potential delays in updates

**CONTEXT-AWARE RESEARCH APPROACH:**
The conversation context contains important information from our ongoing discussion. Use this context to:
- Understand references to people, publications, or topics mentioned earlier
- Avoid asking for clarification when context already provides answers
- Build upon previous research findings rather than starting from scratch
- Connect current research to information already discovered
- Address follow-up questions in the context of earlier research

**SYSTEMATIC RESEARCH METHODOLOGY:**
Think step-by-step and use a systematic approach:

1. **Query Analysis**: 
   - What specific information is requested?
   - How does conversation context inform this query?
   - What entities are involved (authors, publications, projects)?
   - What depth of analysis is needed?

2. **Strategic Tool Usage**:
   - Start with targeted searches using exact parameters
   - Use multiple search approaches for verification
   - Look for patterns, trends, and relationships
   - Cross-reference data for consistency

3. **Quality Research Standards**:
   - Focus on concrete findings: names, numbers, dates, affiliations
   - Provide comprehensive results with supporting evidence
   - Look for collaboration networks and research patterns
   - Include publication trends and impact indicators
   - Connect findings to broader research landscape

4. **Response Excellence**:
   - Lead with direct answer to the specific question
   - Structure information clearly with headers and formatting
   - Include relevant statistics, examples, and context
   - Address all aspects of the original query
   - Connect to previous conversation when relevant
   - **Only mention database limitations when directly relevant to understanding results**

**EXECUTION PRINCIPLES:**
- Use tools systematically and efficiently
- Verify important findings through multiple approaches
- Build comprehensive profiles with rich context
- Provide thorough analysis with detailed explanations
- Handle name variations and disambiguation carefully
- Assess data quality and completeness

Now execute your research using the available tools to provide a comprehensive, well-structured response."""

# =============================================================================
# CLEANED WORKFLOW CREATION
# =============================================================================

def create_react_workflow(
    es_client=None, 
    index_name: str = "research-publications-static", 
    session_id: str = None,
    memory_manager=None
) -> StateGraph:
    """Create clean ReAct workflow."""
    
    # Get tools
    if es_client:
        tools = get_all_tools(es_client=es_client, index_name=index_name)
    else:
        tools = get_all_tools()
    
    # Use memory manager
    if memory_manager is None:
        try:
            from .memory_singleton import get_global_memory_manager
            memory_manager = get_global_memory_manager()
            print("Using global memory manager singleton")
        except ImportError:
            print("Warning: Could not import global memory manager")
            memory_manager = None
    
    # Create ReAct LLM
    react_llm = create_react_llm(session_id)
    
    def react_step(state: ReActState):
        """ReAct step - reasoning and acting."""
        try:
            query = state["input"]
            session_id = state["session_id"]
            
            # Get conversation context
            conversation_context = get_conversation_context(memory_manager, session_id, max_length=10000)
            
            print(f"ReAct executing with context length: {len(conversation_context)} chars")
            
            # Create prompt with context
            react_prompt = REACT_PROMPT_TEMPLATE.format(
                query=query,
                conversation_context=conversation_context
            )
            
            # Create ReAct agent
            agent_executor = create_react_agent(react_llm, tools, prompt=react_prompt)
            
            config = {
                "metadata": {
                    "step": "react",
                    "session_id": session_id,
                    "query": query
                },
                "tags": [
                    f"session-{session_id}",
                    "react"
                ]
            }
            
            try:
                result = agent_executor.invoke({
                    "messages": [HumanMessage(content=query)]
                }, config=config)
                
                response_content = result["messages"][-1].content
                
                print(f"ReAct completed with response length: {len(response_content)} chars")
                
                return {
                    "response": response_content,
                    "session_id": session_id
                }
                
            except Exception as exec_error:
                error_response = f"Error during research: {str(exec_error)}"
                print(f"ReAct execution error: {exec_error}")
                return {
                    "response": error_response,
                    "session_id": session_id
                }
        
        except Exception as e:
            print(f"ReAct error: {e}")
            return {
                "response": f"Error processing request: {str(e)}",
                "session_id": state["session_id"]
            }
    
    # Build workflow
    workflow = StateGraph(ReActState)
    workflow.add_node("react", react_step)
    workflow.add_edge(START, "react")
    workflow.add_edge("react", END)
    
    return workflow

# =============================================================================
# CLEANED RESEARCH AGENT CLASS
# =============================================================================

class ResearchAgent:
    """Clean ReAct Research Agent."""
    
    def __init__(self, es_client=None, index_name: str = "research-publications-static", 
                 recursion_limit: int = 50, memory_manager=None): 
        self.es_client = es_client
        self.index_name = index_name
        self.recursion_limit = recursion_limit
        self.memory_manager = memory_manager
        self.app = None

    def _compile_agent(self, session_id: str = None):
        """Compile ReAct agent for session."""
        workflow = create_react_workflow(self.es_client, self.index_name, session_id, self.memory_manager)
        self.app = workflow.compile()

    async def stream_query(self, query: str, conversation_history: Optional[List[Dict]] = None, frontend_session_id: str = None):
        """Stream query using ReAct workflow with proper cleanup."""
        
        session_id = frontend_session_id or f"fallback_{str(uuid.uuid4())}"
        
        # Compile if needed
        if self.app is None:
            self._compile_agent(session_id)
        
        initial_state = {
            "input": query,
            "response": None,
            "session_id": session_id
        }
        
        config = {
            "recursion_limit": self.recursion_limit,
            "metadata": {
                "query": query,
                "session_id": session_id,
                "index_name": self.index_name,
                "conversation_turn": len(conversation_history or []) + 1
            },
            "tags": [
                "streaming", 
                f"session-{session_id}",
                f"turn-{len(conversation_history or []) + 1}",
                "react"
            ],
            "run_name": f"ReAct-Query-Turn-{len(conversation_history or []) + 1}"
        }
        
        try:
            # Use astream with proper async context handling
            stream = self.app.astream(initial_state, config=config)
            
            async for event in stream:
                print(f"Streaming event: {list(event.keys())}")
                if 'react' in event and 'response' in event['react']:
                    print(f"Found response in event: {event['react']['response'][:100]}...")
                yield event
                
        except GeneratorExit:
            # Handle generator cleanup gracefully
            print("Stream generator cleanup - this is normal")
            pass
        except Exception as e:
            print(f"Streaming error: {e}")
            yield {"error": {"error": str(e)}}

    async def stream_query_without_recompile(self, query: str, conversation_history: Optional[List[Dict]] = None, frontend_session_id: str = None):
        """Backward compatibility method."""
        async for event in self.stream_query(query, conversation_history, frontend_session_id):
            yield event


if __name__ == "__main__":
    print("Testing cleaned ReAct workflow...")