"""
 Workflow - Keeps model configuration flexibility, removes complex session caching
Uses LangGraph's built-in session management instead of custom caching
MINIMAL LOGGING VERSION
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

# Import tools and memory
from ..tools import get_all_tools
from .memory_manager import MemoryManager

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
# MODEL CONFIGURATION SYSTEM (KEPT AS REQUESTED)
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
            "model": "anthropic/claude-sonnet-4",
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
    session_id: str = None
) -> StateGraph:
    """Create research workflow with configured models."""
    
    # Setup LangSmith
    setup_langsmith(session_id)
    
    # Get tools
    if es_client:
        tools = get_all_tools(es_client=es_client, index_name=index_name)
    else:
        tools = get_all_tools()
    
    # Initialize memory
    memory_manager = MemoryManager()
    
    # Create LLMs with configured models
    planning_llm = create_llm_with_config("planning", session_id)
    execution_llm = create_llm_with_config("execution", session_id)
    replanning_llm = create_llm_with_config("replanning", session_id)
    
    # =============================================================================
    # WORKFLOW NODES
    # =============================================================================
    
    def plan_step(state: PlanExecuteState):
        """Planning step using the configured planning model."""
        try:
            query = state["input"]
            conversation_history = state.get("conversation_history", [])
            session_id = state.get("session_id", f"fallback_{int(time.time())}")
            
            # Format tool information
            tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
            
            # Use external prompt template
            planning_prompt_text = PLANNING_PROMPT_TEMPLATE.format(
                query=query,
                tool_descriptions=tool_descriptions
            )
            
            # Create planner
            planner_prompt = ChatPromptTemplate.from_template(planning_prompt_text)
            planner = planner_prompt | planning_llm.with_structured_output(Plan)
            
            plan = planner.invoke({})
            
            return {
                "plan": plan.steps,
                "session_id": session_id
            }
            
        except Exception as e:
            fallback_plan = [f"Research comprehensive information about: {query}"]
            return {
                "plan": fallback_plan,
                "session_id": state.get("session_id", f"fallback_{int(time.time())}")
            }
    
    def execute_step(state: PlanExecuteState):
        """Execution step using the configured execution model."""
        try:
            plan = state["plan"]
            past_steps = state.get("past_steps", [])
            session_id = state.get("session_id")
            
            if not plan:
                return {"past_steps": past_steps}
            
            task = plan[0]
            original_query = state.get("input", "")
            
            # Get conversation context from memory
            conversation_context = ""
            if session_id:
                try:
                    history = memory_manager.get_conversation_history_for_state(session_id)
                    if history:
                        recent_messages = history[-4:]  # Last 2 Q&A pairs
                        context_parts = []
                        for msg in recent_messages:
                            role = msg["role"].title()
                            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
                            context_parts.append(f"- {role}: {content}")
                        conversation_context = "\n".join(context_parts)
                except Exception as e:
                    pass
            
            # Use external prompt template
            execution_prompt = EXECUTION_PROMPT_TEMPLATE.format(
                original_query=original_query,
                task=task,
                research_context=conversation_context or "No previous research context available."
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
            error_response = f"Error executing task: {str(e)}"
            
            task = plan[0] if plan else "unknown_task"
            updated_past_steps = state.get("past_steps", []) + [(task, error_response)]
            return {"past_steps": updated_past_steps}
    
    def replan_step(state: PlanExecuteState):
        """Replanning step using the configured replanning model."""
        try:
            session_id = state.get("session_id")
            original_plan = state.get("plan", [])
            past_steps = state.get("past_steps", [])
            
            # Create research summary from past steps
            research_summary = "No research completed yet."
            if past_steps:
                summary_parts = []
                for i, (task, result) in enumerate(past_steps, 1):
                    # Truncate long results for summary
                    result_preview = result[:500] + "..." if len(result) > 500 else result
                    summary_parts.append(f"Step {i}: {task}\nResult: {result_preview}")
                research_summary = "\n\n".join(summary_parts)
            
            # Use external prompt template
            replanning_prompt = REPLANNING_PROMPT_TEMPLATE.format(
                original_objective=state["input"],
                original_plan=original_plan,
                research_summary=research_summary
            )
            
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
    
    def __init__(self, es_client=None, index_name: str = "research-publications-static", recursion_limit: int = 50):
        self.es_client = es_client
        self.index_name = index_name
        self.recursion_limit = recursion_limit
        self.app = None

    def _compile_agent(self, session_id: str = None):
        """Compile agent for session."""
        workflow = create_workflow(self.es_client, self.index_name, session_id)
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
    print("Testing simplified workflow with configured models...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass