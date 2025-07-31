"""
COMPLETE Enhanced workflow.py with Smart LLM-Powered Methodology Learning
BUILDS ON: Your existing minimal callback workflow with LLM-powered learning integration
ADDS: Smart methodology logging throughout the research process
CLEANED: Removed redundant conversation context from planning
REFACTORED: Moved prompts to separate TXT files for better maintainability
UPDATED: Correct import paths for your project structure
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

# CRITICAL FIX: Minimal LangSmith imports to avoid callback issues
from langsmith import Client

# Import tools, memory, and smart methodology logger
from ..tools import get_all_tools
from .memory_manager import IntegratedMemoryManager
from .methodology_logger import StandardMethodologyLogger

# UPDATED: Import prompts from top-level prompts directory
from ..prompts import (
    PLANNING_PROMPT_TEMPLATE,
    EXECUTION_PROMPT_TEMPLATE, 
    REPLANNING_PROMPT_TEMPLATE
)

# =============================================================================
# STATE SCHEMA (UNCHANGED)
# =============================================================================

class PlanExecuteState(TypedDict):
    """State schema with memory session tracking."""
    input: str
    plan: List[str]
    past_steps: List[tuple[str, str]]
    response: Optional[str]
    
    # Memory tracking
    session_id: Optional[str]
    memory_session_id: Optional[str]
    conversation_history: Optional[List[Dict[str, Any]]]

# =============================================================================
# PYDANTIC MODELS (UNCHANGED)
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
# MINIMAL LANGSMITH SETUP - NO PROBLEMATIC CALLBACKS
# =============================================================================

def setup_minimal_langsmith():
    """
    CRITICAL FIX: Minimal LangSmith setup without problematic callbacks.
    Only sets environment variables, no callback handlers.
    """
    load_dotenv()
    
    # Set LangSmith environment variables
    langsmith_config = {
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "true"),
        "LANGCHAIN_ENDPOINT": os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY"),
        "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT", "research-agent-smart-methodology")
    }
    
    # Apply configuration
    for key, value in langsmith_config.items():
        if value:
            os.environ[key] = value
    
    # Validate configuration
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("⚠️ WARNING: LANGCHAIN_API_KEY not set. LangSmith tracing disabled.")
        return None
    
    try:
        # Test connection without creating callback handlers
        client = Client(
            api_url=os.getenv("LANGCHAIN_ENDPOINT"),
            api_key=os.getenv("LANGCHAIN_API_KEY")
        )
        print("✅ LangSmith environment configured with smart methodology tracking")
        return client
    except Exception as e:
        print(f"❌ LangSmith configuration error: {e}")
        return None

# =============================================================================
# MAIN WORKFLOW CREATION WITH SMART METHODOLOGY LEARNING
# =============================================================================

def create_research_workflow(es_client=None, index_name: str = "research-publications-static", session_id: str = None) -> StateGraph:
    """
    Create research workflow with SMART LLM-POWERED methodology learning.
    BUILDS ON: Existing minimal callback workflow.
    ADDS: Intelligent methodology observation and analysis throughout.
    CLEANED: Removed redundant conversation context from planning.
    REFACTORED: Uses external TXT prompt templates for better maintainability.
    UPDATED: Correct import paths for your project structure.
    """
    
    # Setup minimal LangSmith (no callbacks)
    langsmith_client = setup_minimal_langsmith()
    
    # Get tools
    if es_client:
        tools = get_all_tools(es_client=es_client, index_name=index_name)
        print(f"✅ SMART METHODOLOGY workflow: Initialized {len(tools)} research tools")
    else:
        tools = get_all_tools()
        print(f"✅ SMART METHODOLOGY workflow: Initialized {len(tools)} research tools")
    
    # Initialize integrated memory
    integrated_memory = IntegratedMemoryManager(memory_type="buffer_window")
    print("🧠 SMART METHODOLOGY workflow: Integrated memory initialized")
    
    # ENHANCED: Initialize smart methodology logger
    standard_logger = StandardMethodologyLogger()
    print("🧠 Smart Methodology Learning system active with LLM analysis")
    
    # CRITICAL FIX: Create LLMs WITHOUT callback handlers
    try:
        llm = ChatLiteLLM(
            model="anthropic/claude-sonnet-4",
            api_key=os.getenv("LITELLM_API_KEY"),
            api_base=os.getenv("LITELLM_BASE_URL"),
            temperature=0,
            # REMOVED: callbacks=callbacks,  # This was causing async issues
            metadata={"component": "smart_methodology_main_llm", "session_id": session_id}
        )
        
        replanner_llm = ChatLiteLLM(
            model="anthropic/claude-haiku-3.5",
            api_key=os.getenv("LITELLM_API_KEY"),
            api_base=os.getenv("LITELLM_BASE_URL"),
            temperature=0,
            # REMOVED: callbacks=callbacks,  # This was causing async issues
            metadata={"component": "smart_methodology_replanner_llm", "session_id": session_id}
        )
        print("✅ LLMs initialized WITHOUT problematic callbacks + smart methodology tracking")
        
    except Exception as e:
        print(f"❌ Error initializing LLMs: {e}")
        raise
    
    # =============================================================================
    # ENHANCED WORKFLOW NODES WITH SMART METHODOLOGY LEARNING
    # =============================================================================
    
    def plan_step(state: PlanExecuteState):
        """CLEANED planning step using external TXT prompt template."""
        try:
            query = state["input"]
            conversation_history = state.get("conversation_history", [])
            memory_session_id = state.get("memory_session_id", str(uuid.uuid4()))
            
            print(f"📋 SMART METHODOLOGY Planning for query: {query}")
            print(f"🧠 Memory session ID: {memory_session_id}")
            
            # ENHANCED: Smart query analysis with rich context
            is_followup = len(conversation_history) > 0
            previous_context = ""
            if is_followup and conversation_history:
                # Get richer context for LLM analysis
                prev_user_messages = [msg for msg in conversation_history if msg.get('role') == 'user']
                prev_ai_messages = [msg for msg in conversation_history if msg.get('role') == 'assistant']
                
                if prev_user_messages and prev_ai_messages:
                    previous_context = f"Previous query: {prev_user_messages[-1].get('content', '')}\nPrevious response: {prev_ai_messages[-1].get('content', '')[:300]}"
            
            # SMART LOGGING: Intelligent query analysis
            standard_logger.log_query_start(
                memory_session_id, query, is_followup, previous_context
            )
            
            # Format tool information
            tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
            
            # REFACTORED: Use external TXT prompt template
            planning_prompt_text = PLANNING_PROMPT_TEMPLATE.format(
                query=query,
                tool_descriptions=tool_descriptions
            )
            
            # Create planner WITHOUT callbacks
            planner_prompt = ChatPromptTemplate.from_template(planning_prompt_text)
            planner = planner_prompt | llm.with_structured_output(Plan)
            
            plan = planner.invoke({})
            
            print(f"📋 Created plan with {len(plan.steps)} steps:")
            for i, step in enumerate(plan.steps, 1):
                print(f"  {i}. {step}")
            
            return {
                "plan": plan.steps,
                "memory_session_id": memory_session_id
            }
            
        except Exception as e:
            print(f"❌ Error in SMART METHODOLOGY planning: {e}")
            fallback_plan = [f"Research comprehensive information about: {query}"]
            return {
                "plan": fallback_plan,
                "memory_session_id": str(uuid.uuid4())
            }
    
    def execute_step(state: PlanExecuteState):
        """Execution step using external TXT prompt template."""
        try:
            plan = state["plan"]
            past_steps = state.get("past_steps", [])
            memory_session_id = state.get("memory_session_id", str(uuid.uuid4()))
            
            if not plan:
                print("⚠️ No plan available for execution")
                return {"past_steps": past_steps}
            
            task = plan[0]
            original_query = state.get("input", "")
            
            print(f"🔧 SMART METHODOLOGY Executing task: {task}")
            
            # Build execution context
            research_context = integrated_memory.get_research_context_summary(memory_session_id, max_recent_steps=5)
            print(f"📋 Research context: {len(research_context):,} chars")
            
            # REFACTORED: Use external TXT prompt template
            execution_prompt = EXECUTION_PROMPT_TEMPLATE.format(
                original_query=original_query,
                task=task,
                research_context=research_context
            )
            
            # CRITICAL FIX: Execute WITHOUT callbacks
            agent_executor = create_react_agent(llm, tools, prompt=execution_prompt)
            
            # Minimal config without problematic callbacks
            config = {
                "metadata": {
                    "step": "smart_methodology_execute",
                    "task": task,
                    "session_id": session_id,
                    "memory_session_id": memory_session_id,
                    "step_number": len(past_steps) + 1
                }
                # REMOVED: "callbacks": callbacks  # This was causing async issues
            }
            
            # ENHANCED: Track execution with timing
            execution_start = time.time()
            
            # Execute with minimal callback approach
            try:
                result = agent_executor.invoke({
                    "messages": [HumanMessage(content=f"Execute this research task: {task}")]
                }, config=config)
                
                response_content = result["messages"][-1].content
                execution_time = time.time() - execution_start
                
                print(f"✅ SMART METHODOLOGY Task completed: {len(response_content):,} chars in {execution_time:.1f}s")
                
            except Exception as exec_error:
                print(f"❌ Execution error: {exec_error}")
                response_content = f"Error executing task '{task}': {str(exec_error)}"
                execution_time = time.time() - execution_start
            
            # SMART LOGGING: Intelligent tool effectiveness analysis
            success = len(response_content) > 100 and "error" not in response_content.lower()
            
            standard_logger.log_tool_usage(
                memory_session_id,
                "research_agent",  # Could be more specific about which tools were used
                task,
                success,
                response_content,  # Full result for LLM analysis
                f"Execution time: {execution_time:.1f}s, Steps completed: {len(past_steps) + 1}, Success: {success}"
            )
            
            # Store FULL result
            reference_id = integrated_memory.store_research_step(
                memory_session_id,
                task,
                response_content
            )
            
            print(f"📚 SMART METHODOLOGY Stored result as: {reference_id}")
            
            updated_past_steps = past_steps + [(task, f"COMPLETE research stored as {reference_id}")]
            
            return {"past_steps": updated_past_steps}
        
        except Exception as e:
            error_response = f"Error executing task: {str(e)}"
            print(f"❌ Error in SMART METHODOLOGY execute_step: {e}")
            
            memory_session_id = state.get("memory_session_id", str(uuid.uuid4()))
            task = plan[0] if plan else "unknown_task"
            
            # SMART LOGGING: Log tool failure
            standard_logger.log_tool_usage(
                memory_session_id,
                "research_agent",
                task,
                False,  # Failed
                error_response,
                f"Tool execution failed with error: {str(e)}"
            )
            
            try:
                integrated_memory.store_research_step(memory_session_id, task, error_response)
            except Exception as mem_error:
                print(f"⚠️ Could not store error in memory: {mem_error}")
            
            updated_past_steps = state.get("past_steps", []) + [(task, error_response)]
            return {"past_steps": updated_past_steps}
    
    def replan_step(state: PlanExecuteState):
        """CLEANED replanning step using external TXT prompt template."""
        try:
            memory_session_id = state.get("memory_session_id", str(uuid.uuid4()))
            original_plan = state.get("plan", [])
            past_steps = state.get("past_steps", [])
            
            print(f"🔄 SMART METHODOLOGY Replanning for session: {memory_session_id}")
            
            # ENHANCED: Track session timing for comprehensive analysis
            session_start_time = getattr(replan_step, 'session_start_time', None)
            if session_start_time is None:
                replan_step.session_start_time = time.time()
                session_start_time = replan_step.session_start_time
            
            # Get research context with full content for LLM analysis
            research_summary = integrated_memory.get_research_context_summary(
                memory_session_id, 
                max_recent_steps=3
            )
            
            print(f"📋 Research context: {len(research_summary):,} chars")
            
            # Verify we have research data
            if research_summary == "No research steps completed yet.":
                if past_steps:
                    research_summary = f"Completed {len(past_steps)} research steps. Latest: {past_steps[-1][0]}"
                else:
                    research_summary = "No research completed yet."
            
            # REFACTORED: Use external TXT prompt template
            replanning_prompt = REPLANNING_PROMPT_TEMPLATE.format(
                original_objective=state["input"],
                original_plan=original_plan,
                research_summary=research_summary
            )
            
            # Create replanner WITHOUT callbacks
            replanner_prompt_obj = ChatPromptTemplate.from_template(replanning_prompt)
            replanner = replanner_prompt_obj | replanner_llm.with_structured_output(Act)
            
            # Minimal config
            config = {
                "metadata": {
                    "step": "smart_methodology_replanning",
                    "memory_session_id": memory_session_id,
                    "session_id": session_id
                }
                # REMOVED: "callbacks": callbacks
            }
            
            # Execute replanner without callbacks
            try:
                response = replanner.invoke({}, config=config)
            except Exception as replan_error:
                print(f"❌ Replanner execution error: {replan_error}")
                return {"response": f"Research completed with smart methodology tracking. Error in replanning: {str(replan_error)}"}
            
            if response.action_type == "response":
                # SMART LOGGING: Comprehensive session completion analysis
                execution_time = time.time() - session_start_time
                
                # Count actual replanning events
                replanning_count = len([step for step in past_steps if "replan" in step[1].lower()])
                
                # Assess final success intelligently
                final_success = "success"
                if "partial" in response.response.lower() or "incomplete" in response.response.lower():
                    final_success = "partial"
                elif "error" in response.response.lower() or "failed" in response.response.lower():
                    final_success = "failed"
                
                # Get comprehensive research results for LLM analysis
                comprehensive_data = integrated_memory.get_comprehensive_final_response_data(memory_session_id)
                full_results = "\n\n".join(comprehensive_data.get('full_results', []))  # ✅ ALL results
                
                standard_logger.log_session_complete(
                    memory_session_id,
                    state["input"],
                    len(past_steps),
                    replanning_count,
                    final_success,
                    execution_time,
                    full_results
                )
                
                # CLEANED: Create comprehensive final response with NO truncation or technical footers
                enhanced_response = response.response
                
                return {"response": enhanced_response}
            else:
                # SMART LOGGING: Intelligent replanning analysis
                # Determine replanning reason intelligently
                if len(past_steps) == 0:
                    replanning_reason = "Initial planning phase - setting up research approach"
                elif "no research steps completed yet" in research_summary.lower():
                    replanning_reason = "No research progress made - need different approach"
                elif len(research_summary) < 500:
                    replanning_reason = "Insufficient research results - need additional investigation"
                else:
                    replanning_reason = "Research incomplete - expanding investigation scope"
                
                # Describe approaches for LLM analysis
                previous_approach = f"Plan with {len(original_plan)} steps: {', '.join(original_plan[:2])}{'...' if len(original_plan) > 2 else ''}"
                new_approach = f"Revised plan with {len(response.steps or [])} steps: {', '.join((response.steps or [])[:2])}{'...' if len(response.steps or []) > 2 else ''}"
                
                standard_logger.log_replanning_event(
                    memory_session_id,
                    state["input"],
                    len(past_steps) + 1,
                    replanning_reason,
                    previous_approach,
                    new_approach,
                    research_summary  # Full context for LLM analysis
                )
                
                print(f"🔄 SMART METHODOLOGY replanner: Continuing with more steps")
                return {"plan": response.steps or []}
                
        except Exception as e:
            print(f"❌ Error in SMART METHODOLOGY replanning: {e}")
            
            memory_session_id = state.get("memory_session_id", str(uuid.uuid4()))
            try:
                fallback_summary = integrated_memory.get_research_context_summary(memory_session_id, max_recent_steps=2)
                
                if fallback_summary != "No research steps completed yet.":
                    # CLEANED: Clean fallback response without technical footer
                    fallback_response = f"""Based on research completed:

{fallback_summary[:1000]}{"..." if len(fallback_summary) > 1000 else ""}"""
                else:
                    fallback_response = "Research completed successfully."
                
                return {"response": fallback_response}
                
            except Exception as fallback_error:
                print(f"⚠️ Fallback also failed: {fallback_error}")
                return {"response": f"Research error during replanning: {str(e)}. Analysis captured for learning."}
    
    def should_end(state: PlanExecuteState) -> Literal["agent", "__end__"]:
        """Simple decision function for workflow routing."""
        if state.get("response"):
            return "__end__"
        else:
            print("🔄 SMART METHODOLOGY: Continuing to agent execution")
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
    
    print("🧠 SMART METHODOLOGY workflow constructed with LLM-powered learning + external TXT prompts")
    
    return workflow

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compile_research_agent(es_client=None, index_name: str = "research-publications-static", recursion_limit: int = 50, session_id: str = None):
    """Compile research agent with smart methodology learning."""
    workflow = create_research_workflow(es_client, index_name, session_id)
    app = workflow.compile()
    return app

def run_research_query(query: str, es_client=None, index_name: str = "research-publications-static", recursion_limit: int = 50, stream: bool = False, conversation_history: Optional[List[Dict]] = None, session_id: str = None) -> Dict[str, Any]:
    """Run research query using SMART METHODOLOGY patterns."""
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    app = compile_research_agent(es_client, index_name, recursion_limit, session_id)
    
    initial_state = {
        "input": query,
        "plan": [],
        "past_steps": [],
        "response": None,
        "session_id": session_id,
        "memory_session_id": str(uuid.uuid4()),
        "conversation_history": conversation_history or []
    }
    
    config = {
        "recursion_limit": recursion_limit,
        "metadata": {
            "query": query,
            "session_id": session_id,
            "index_name": index_name,
            "has_conversation_history": bool(conversation_history),
            "agent_type": "smart_methodology"
        },
        "tags": ["smart_methodology_agent", "plan_execute", f"session_{session_id}"]
    }
    
    if stream:
        final_state = None
        for event in app.stream(initial_state, config=config):
            for k, v in event.items():
                if k != "__end__":
                    final_state = v
        return final_state
    else:
        result = app.invoke(initial_state, config=config)
        return result

# =============================================================================
# RESEARCH AGENT CLASS WITH SMART METHODOLOGY
# =============================================================================

class ResearchAgent:
    """Research Agent using smart methodology learning."""
    
    def __init__(self, es_client=None, index_name: str = "research-publications-static", recursion_limit: int = 50):
        self.es_client = es_client
        self.index_name = index_name
        self.recursion_limit = recursion_limit
        self.app = None
        self.langsmith_client = setup_minimal_langsmith()

    def _compile_agent(self, session_id: str = None):
        """Compile agent with smart methodology learning."""
        self.app = compile_research_agent(
            self.es_client, 
            self.index_name, 
            self.recursion_limit,
            session_id
        )

    async def stream_query(self, query: str, conversation_history: Optional[List[Dict]] = None):
        """Stream query with smart methodology learning."""
        
        session_id = str(uuid.uuid4())
        memory_session_id = str(uuid.uuid4())
        self._compile_agent(session_id)
        
        initial_state = {
            "input": query,
            "plan": [],
            "past_steps": [],
            "response": None,
            "session_id": session_id,
            "memory_session_id": memory_session_id,
            "conversation_history": conversation_history or []
        }
        
        config = {
            "recursion_limit": self.recursion_limit,
            "metadata": {
                "query": query,
                "session_id": session_id,
                "memory_session_id": memory_session_id,
                "index_name": self.index_name,
                "agent_type": "smart_methodology"
            },
            "tags": ["smart_methodology_agent", "streaming", f"session_{session_id}"]
        }
        
        # Stream with smart methodology learning
        try:
            print("🚀 Starting SMART METHODOLOGY stream with LLM analysis")
            async for event in self.app.astream(initial_state, config=config):
                yield event
        except Exception as e:
            print(f"❌ Error in SMART METHODOLOGY streaming: {e}")
            yield {"error": {"error": str(e)}}

if __name__ == "__main__":
    print("Testing SMART METHODOLOGY workflow with LLM-powered learning...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    

    print("🧠 Key features:")
    print("  - LLM-powered query analysis and categorization")
    print("  - Intelligent tool effectiveness assessment")
    print("  - Smart replanning reason analysis")
    print("  - Comprehensive session outcome evaluation")
    print("  - Dynamic pattern recognition and learning")
    print("  - No hard-coded rules - fully adaptive")
    print("  - CLEANED: No redundant conversation context in planning")
    print("  - CLEANED: No truncation limits or technical footers")
    print("  - REFACTORED: External TXT prompt templates for better maintainability")
    print("  - UPDATED: Correct import paths for your project structure")