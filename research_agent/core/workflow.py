"""
 Workflow - FIXED MEMORY AND CONTEXT HANDLING
 Addresses: Truncation, missing context in planner/replanner, poor context usage
"""

from typing import Dict, Any, List, Optional, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_litellm import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
import os
import uuid
import asyncio
import time
from dotenv import load_dotenv

# Import tools
from ..tools import get_all_tools

# Import prompts from external files
from ..prompts import (
    PLANNING_PROMPT_TEMPLATE,
    EXECUTION_PROMPT_TEMPLATE, 
    REPLANNING_PROMPT_TEMPLATE
)

# =============================================================================
# STATE SCHEMA
# =============================================================================

class PlanExecuteState(TypedDict):
    """State schema for plan-execute workflow."""
    input: str
    plan: List[str]
    past_steps: List[tuple[str, str]]
    response: Optional[str]
    session_id: Optional[str]
    conversation_history: Optional[List[Dict[str, Any]]]

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class Plan(BaseModel):
    """Plan to follow for query execution."""
    steps: List[str] = Field(
        description="Different steps to follow, should be in sorted order. Each step should be a clear, actionable task."
    )

class Response(BaseModel):
    """Final response to user query."""
    response: str = Field(
        description="The final comprehensive answer to the user's question"
    )

class Act(BaseModel):
    """Action to perform - either respond or continue planning."""
    action_type: Literal["response", "plan"] = Field(
        description="Type of action: 'response' to respond to user, 'plan' to continue planning"
    )
    response: Optional[str] = Field(
        default=None,
        description="Final response to user (only if action_type is 'response')"
    )
    steps: Optional[List[str]] = Field(
        default=None,
        description="Plan steps to execute (only if action_type is 'plan')"
    )

# =============================================================================
# CONTEXT HELPER FUNCTIONS
# =============================================================================

def get_conversation_context(memory_manager, session_id: str, max_length: int = 2000) -> str:
    """Get conversation context with proper length management."""
    if not session_id or not memory_manager:
        return "No previous conversation context available."
    
    try:
        history = memory_manager.get_conversation_history_for_state(session_id)
        if not history:
            return "No previous conversation context available."
        
        # Build context from most recent messages, staying within max_length
        context_parts = []
        current_length = 0
        
        # Start from most recent and work backwards
        for msg in reversed(history):
            role = msg["role"].title()
            content = msg["content"]
            
            # Estimate the length we'll add
            new_part = f"- {role}: {content}\n"
            new_length = current_length + len(new_part)
            
            # If adding this would exceed max_length, truncate smartly
            if new_length > max_length:
                remaining_space = max_length - current_length - len(f"- {role}: ") - 20  # Leave space for truncation indicator
                if remaining_space > 100:  # Only add if we have meaningful space
                    truncated_content = content[:remaining_space] + "..."
                    context_parts.insert(0, f"- {role}: {truncated_content}")
                break
            
            context_parts.insert(0, new_part.strip())
            current_length = new_length
        
        context = "\n".join(context_parts)
        print(f"🔍 Context built: {len(context)} chars from {len(context_parts)} messages")
        return context
        
    except Exception as e:
        print(f"❌ Error getting conversation context: {e}")
        return "Error retrieving conversation context."

def format_conversation_summary(memory_manager, session_id: str) -> str:
    """Get a concise summary of conversation for planning/replanning."""
    if not session_id or not memory_manager:
        return "No conversation history."
    
    try:
        history = memory_manager.get_conversation_history_for_state(session_id)
        if not history:
            return "No conversation history."
        
        # Get last 2 Q&A pairs for planning context
        recent_messages = history[-4:] if len(history) >= 4 else history
        
        summary_parts = []
        for i in range(0, len(recent_messages), 2):
            if i + 1 < len(recent_messages):
                user_msg = recent_messages[i]["content"]
                assistant_msg = recent_messages[i + 1]["content"]
                summary_parts.append(f"Q: {user_msg}")
                summary_parts.append(f"A: {assistant_msg}")
        
        return "\n".join(summary_parts)
        
    except Exception as e:
        print(f"❌ Error getting conversation summary: {e}")
        return "Error retrieving conversation summary."

# =============================================================================
# LANGSMITH SETUP
# =============================================================================

def setup_langsmith(session_id: str = None):
    """Setup LangSmith for session tracking."""
    load_dotenv()
    
    project_name = "research-agent-conversations"
    session_name = session_id or "default-session"
    
    langsmith_config = {
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "true"),
        "LANGCHAIN_ENDPOINT": os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY"),
        "LANGCHAIN_PROJECT": project_name,
        "LANGCHAIN_SESSION": session_name
    }
    
    # Apply configuration
    for key, value in langsmith_config.items():
        if value:
            os.environ[key] = value
    
    if not os.getenv("LANGCHAIN_API_KEY"):
        return None
    
    return True

# =============================================================================
# MODEL CONFIGURATION SYSTEM
# =============================================================================

def create_llm_with_config(purpose: str, session_id: str = None) -> ChatLiteLLM:
    """Create LLM with purpose-specific model and configuration."""
    
    # Purpose-specific configurations with model names
    configs = {
        "planning": {
            "model": "anthropic/claude-sonnet-3.7",
            "temperature": 0,
            "max_tokens": 2000,
            "description": "Planning model - structured plan generation"
        },
        "execution": {
            "model": "anthropic/claude-haiku-3.5",
            "temperature": 0.1,
            "max_tokens": 4000,
            "description": "Execution model - tool interaction and research"
        },
        "replanning": {
            "model": "anthropic/claude-sonnet-3.7",
            "temperature": 0,
            "max_tokens": 3000,
            "description": "Replanning model - critical decision making"
        }
    }
    
    config = configs.get(purpose, configs["execution"])
    model_name = config["model"]
    
    metadata = {
        "component": f"{purpose}_llm",
        "model_name": model_name,
        "purpose": purpose
    }
    
    if session_id:
        metadata.update({
            "session_id": session_id,
            "session_group": f"research-session-{session_id}"
        })
    
    try:
        llm = ChatLiteLLM(
            model=model_name,
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
# WORKFLOW CREATION
# =============================================================================

def create_workflow(
    es_client=None, 
    index_name: str = "research-publications-static", 
    session_id: str = None,
    memory_manager=None
) -> StateGraph:
    """Create research workflow with configured models."""
    
    # Setup LangSmith
    setup_langsmith(session_id)
    
    # Get tools
    if es_client:
        tools = get_all_tools(es_client=es_client, index_name=index_name)
    else:
        tools = get_all_tools()
    
    # ✅ FIXED: Use provided memory manager
    if memory_manager is None:
        try:
            from .memory_singleton import get_global_memory_manager
            memory_manager = get_global_memory_manager()
            print("✅ Using global memory manager singleton")
        except ImportError:
            from .memory_manager import MemoryManager
            memory_manager = MemoryManager()
            print("⚠️ Warning: Using fallback MemoryManager")
    
    # Create LLMs with configured models
    planning_llm = create_llm_with_config("planning", session_id)
    execution_llm = create_llm_with_config("execution", session_id)
    replanning_llm = create_llm_with_config("replanning", session_id)
    
    # =============================================================================
    # WORKFLOW NODES
    # =============================================================================
    
    def plan_step(state: PlanExecuteState):
        """Planning step with conversation context."""
        try:
            query = state["input"]
            session_id = state.get("session_id", f"fallback_{int(time.time())}")
            
            # ✅ FIXED: Include conversation context in planning
            conversation_summary = format_conversation_summary(memory_manager, session_id)
            
            # Format tool information
            tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
            
            # ✅ UPDATED: Include conversation context in planning prompt
            planning_prompt_text = PLANNING_PROMPT_TEMPLATE.format(
                query=query,
                tool_descriptions=tool_descriptions,
                conversation_context=conversation_summary  # Add this to your planning template
            )
            
            print(f"🔍 PLANNING with context: {conversation_summary[:200]}...")
            
            # Create planner
            planner_prompt = ChatPromptTemplate.from_template(planning_prompt_text)
            planner = planner_prompt | planning_llm.with_structured_output(Plan)
            
            plan = planner.invoke({})
            
            return {
                "plan": plan.steps,
                "session_id": session_id
            }
            
        except Exception as e:
            print(f"❌ Planning error: {e}")
            fallback_plan = [f"Research comprehensive information about: {query}"]
            return {
                "plan": fallback_plan,
                "session_id": state.get("session_id", f"fallback_{int(time.time())}")
            }
    
    def execute_step(state: PlanExecuteState):
        """Execution step with full conversation context."""
        try:
            plan = state["plan"]
            past_steps = state.get("past_steps", [])
            session_id = state.get("session_id")
            
            if not plan:
                return {"past_steps": past_steps}
            
            task = plan[0]
            original_query = state.get("input", "")
            
            # ✅ FIXED: Get full conversation context without truncation issues
            conversation_context = get_conversation_context(memory_manager, session_id, max_length=10000)
            
            print(f"🔍 EXECUTION with context length: {len(conversation_context)} chars")
            print(f"🔍 Context preview: {conversation_context[:300]}...")
            
            # Use external prompt template
            execution_prompt = EXECUTION_PROMPT_TEMPLATE.format(
                original_query=original_query,
                task=task,
                research_context=conversation_context
            )
            
            # Execute
            agent_executor = create_react_agent(execution_llm, tools, prompt=execution_prompt)
            
            config = {
                "metadata": {
                    "step": "execute",
                    "task": task,
                    "session_id": session_id,
                    "step_number": len(past_steps) + 1
                },
                "tags": [
                    f"session-{session_id}",
                    f"step-{len(past_steps) + 1}",
                    "execution"
                ]
            }
            
            try:
                result = agent_executor.invoke({
                    "messages": [HumanMessage(content=f"Execute this research task: {task}")]
                }, config=config)
                
                response_content = result["messages"][-1].content
                
            except Exception as exec_error:
                response_content = f"Error executing task '{task}': {str(exec_error)}"
            
            updated_past_steps = past_steps + [(task, response_content)]
            
            return {"past_steps": updated_past_steps}
        
        except Exception as e:
            print(f"❌ Execution error: {e}")
            error_response = f"Error executing task: {str(e)}"
            
            task = plan[0] if plan else "unknown_task"
            updated_past_steps = state.get("past_steps", []) + [(task, error_response)]
            return {"past_steps": updated_past_steps}
    
    def replan_step(state: PlanExecuteState):
        """Replanning step with conversation context."""
        try:
            session_id = state.get("session_id")
            original_plan = state.get("plan", [])
            past_steps = state.get("past_steps", [])
            
            # ✅ FIXED: Include conversation context in replanning
            conversation_summary = format_conversation_summary(memory_manager, session_id)
            
            # Create research summary from past steps
            research_summary = "No research completed yet."
            if past_steps:
                summary_parts = []
                for i, (task, result) in enumerate(past_steps, 1):
                    # Truncate long results for summary
                    result_preview = result[:500] + "..." if len(result) > 500 else result
                    summary_parts.append(f"Step {i}: {task}\nResult: {result_preview}")
                research_summary = "\n\n".join(summary_parts)
            
            # ✅ UPDATED: Include conversation context in replanning prompt
            replanning_prompt = REPLANNING_PROMPT_TEMPLATE.format(
                original_objective=state["input"],
                original_plan=original_plan,
                research_summary=research_summary,
                conversation_context=conversation_summary  # Add this to your replanning template
            )
            
            print(f"🔍 REPLANNING with context: {conversation_summary[:200]}...")
            
            # Create replanner
            replanner_prompt_obj = ChatPromptTemplate.from_template(replanning_prompt)
            replanner = replanner_prompt_obj | replanning_llm.with_structured_output(Act)
            
            config = {
                "metadata": {
                    "step": "replanning",
                    "session_id": session_id
                },
                "tags": [
                    f"session-{session_id}",
                    "replanning"
                ]
            }
            
            try:
                response = replanner.invoke({}, config=config)
            except Exception as replan_error:
                return {"response": f"Research completed. Error in replanning: {str(replan_error)}"}
            
            if response.action_type == "response":
                return {"response": response.response}
            else:
                return {"plan": response.steps or []}
                
        except Exception as e:
            print(f"❌ Replanning error: {e}")
            # Create fallback response from past steps
            if state.get("past_steps"):
                last_result = state["past_steps"][-1][1]
                fallback_response = f"Research completed. Last result: {last_result[:1000]}"
            else:
                fallback_response = f"Research error during replanning: {str(e)}"
            
            return {"response": fallback_response}
    
    def should_end(state: PlanExecuteState) -> Literal["agent", "__end__"]:
        """Simple decision function for workflow routing."""
        if state.get("response"):
            return "__end__"
        else:
            return "agent"
    
    # =============================================================================
    # WORKFLOW CONSTRUCTION
    # =============================================================================
    
    workflow = StateGraph(PlanExecuteState)
    
    # Add nodes
    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)
    
    # Add edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    workflow.add_edge("agent", "replan")
    workflow.add_conditional_edges(
        "replan",
        should_end,
        ["agent", END]
    )
    
    return workflow

# =============================================================================
# RESEARCH AGENT CLASS
# =============================================================================

class ResearchAgent:
    """Research Agent using LangGraph's session management."""
    
    def __init__(self, es_client=None, index_name: str = "research-publications-static", 
                 recursion_limit: int = 50, memory_manager=None): 
        self.es_client = es_client
        self.index_name = index_name
        self.recursion_limit = recursion_limit
        self.memory_manager = memory_manager
        self.app = None

    def _compile_agent(self, session_id: str = None):
        """Compile agent for session."""
        workflow = create_workflow(self.es_client, self.index_name, session_id, self.memory_manager)
        self.app = workflow.compile()

    async def stream_query_without_recompile(self, query: str, conversation_history: Optional[List[Dict]] = None, frontend_session_id: str = None):
        """Stream query using compiled workflow."""
        
        session_id = frontend_session_id or f"fallback_{str(uuid.uuid4())}"
        
        # Compile if needed
        if self.app is None:
            self._compile_agent(session_id)
        
        # Stream with session config
        async for event in self._stream_with_config(query, conversation_history, session_id):
            yield event
    
    async def _stream_with_config(self, query: str, conversation_history: Optional[List[Dict]], session_id: str):
        """Helper method to stream with consistent config."""
        
        initial_state = {
            "input": query,
            "plan": [],
            "past_steps": [],
            "response": None,
            "session_id": session_id,
            "conversation_history": conversation_history or []
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
                f"turn-{len(conversation_history or []) + 1}"
            ],
            "run_name": f"Research-Query-Turn-{len(conversation_history or []) + 1}"
        }
        
        try:
            async for event in self.app.astream(initial_state, config=config):
                yield event
        except Exception as e:
            yield {"error": {"error": str(e)}}


if __name__ == "__main__":
    print("Testing workflow with fixed memory and context handling...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass