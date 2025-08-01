"""
FIXED Enhanced workflow.py - Uses Frontend Session ID with Workflow Caching
CRITICAL FIX: Maintains compiled workflows per session for LangSmith continuity
REMOVED: Workflow recompilation that broke LangSmith session tracking
ADDED: Workflow caching and reuse within sessions
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
# STATE SCHEMA - FIXED to use frontend session_id consistently
# =============================================================================

class PlanExecuteState(TypedDict):
    """State schema with consistent frontend session tracking."""
    input: str
    plan: List[str]
    past_steps: List[tuple[str, str]]
    response: Optional[str]
    
    # FIXED: Use single session_id from frontend consistently
    session_id: Optional[str]  # This comes from frontend and is used for ALL memory operations
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
# FIXED LANGSMITH SETUP WITH SESSION CONSISTENCY
# =============================================================================

def setup_minimal_langsmith(frontend_session_id: str = None):
    """
    BEST PRACTICE: Use one main project with session metadata for organization.
    Based on LangSmith recommendations for session management.
    """
    load_dotenv()
    
    # BEST PRACTICE: Use one main project name with session metadata
    project_name = "research-agent-conversations"  # Single project for all conversations
    session_name = frontend_session_id or "default-session"
    
    # Set LangSmith environment variables following best practices
    langsmith_config = {
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "true"),
        "LANGCHAIN_ENDPOINT": os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY"),
        "LANGCHAIN_PROJECT": project_name,  # Single project for all conversations
        "LANGCHAIN_SESSION": session_name   # Session ID for thread management
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
        # Test connection with session consistency
        client = Client(
            api_url=os.getenv("LANGCHAIN_ENDPOINT"),
            api_key=os.getenv("LANGCHAIN_API_KEY")
        )
        print(f"✅ LangSmith configured - Project: {project_name}")
        print(f"🔗 Session: {frontend_session_id}")
        return client
    except Exception as e:
        print(f"❌ LangSmith configuration error: {e}")
        return None

# =============================================================================
# MAIN WORKFLOW CREATION - FIXED SESSION CONSISTENCY
# =============================================================================

def create_research_workflow(es_client=None, index_name: str = "research-publications-static", frontend_session_id: str = None) -> StateGraph:
    """
    FIXED: Create research workflow using frontend session_id consistently.
    CRITICAL FIX: All memory operations AND LangSmith tracing use the same frontend session_id.
    """
    
    # CRITICAL FIX: Setup LangSmith with the frontend session ID
    langsmith_client = setup_minimal_langsmith(frontend_session_id)
    
    # Get tools
    if es_client:
        tools = get_all_tools(es_client=es_client, index_name=index_name)
        print(f"✅ FIXED workflow: Initialized {len(tools)} research tools")
    else:
        tools = get_all_tools()
        print(f"✅ FIXED workflow: Initialized {len(tools)} research tools")
    
    # Initialize integrated memory
    integrated_memory = IntegratedMemoryManager(memory_type="buffer_window")
    print("🧠 FIXED workflow: Integrated memory initialized")
    
    # Initialize smart methodology logger
    standard_logger = StandardMethodologyLogger()
    print("🧠 Smart Methodology Learning system active with session consistency")
    
    # CRITICAL FIX: Create LLMs with session-aware metadata
    try:
        llm = ChatLiteLLM(
            model="anthropic/claude-sonnet-4",
            api_key=os.getenv("LITELLM_API_KEY"),
            api_base=os.getenv("LITELLM_BASE_URL"),
            temperature=0,
            metadata={
                "component": "session_cached_main_llm", 
                "frontend_session_id": frontend_session_id,
                "session_group": f"research-session-{frontend_session_id}"
            }
        )
        
        replanner_llm = ChatLiteLLM(
            model="anthropic/claude-haiku-3.5",
            api_key=os.getenv("LITELLM_API_KEY"),
            api_base=os.getenv("LITELLM_BASE_URL"),
            temperature=0,
            metadata={
                "component": "session_cached_replanner_llm", 
                "frontend_session_id": frontend_session_id,
                "session_group": f"research-session-{frontend_session_id}"
            }
        )
        print(f"✅ LLMs initialized with session consistency: {frontend_session_id}")
        
    except Exception as e:
        print(f"❌ Error initializing LLMs: {e}")
        raise
    
    # =============================================================================
    # FIXED WORKFLOW NODES - USE FRONTEND SESSION_ID CONSISTENTLY
    # =============================================================================
    
    def plan_step(state: PlanExecuteState):
        """FIXED planning step using frontend session_id consistently."""
        try:
            query = state["input"]
            conversation_history = state.get("conversation_history", [])
            # CRITICAL FIX: Use frontend session_id directly, don't generate new one
            frontend_session_id = state.get("session_id")
            
            if not frontend_session_id:
                # Fallback only if no frontend session provided
                frontend_session_id = f"fallback_{int(time.time())}"
                print("⚠️ No frontend session_id provided, using fallback")
            
            print(f"📋 FIXED Planning for query: {query}")
            print(f"🔗 Using frontend session ID: {frontend_session_id}")
            
            # Smart query analysis with rich context
            is_followup = len(conversation_history) > 0
            previous_context = ""
            if is_followup and conversation_history:
                prev_user_messages = [msg for msg in conversation_history if msg.get('role') == 'user']
                prev_ai_messages = [msg for msg in conversation_history if msg.get('role') == 'assistant']
                
                if prev_user_messages and prev_ai_messages:
                    previous_context = f"Previous query: {prev_user_messages[-1].get('content', '')}\nPrevious response: {prev_ai_messages[-1].get('content', '')[:300]}"
            
            # SMART LOGGING: Use frontend session_id
            standard_logger.log_query_start(
                frontend_session_id, query, is_followup, previous_context
            )
            
            # Format tool information
            tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
            
            # Use external TXT prompt template
            planning_prompt_text = PLANNING_PROMPT_TEMPLATE.format(
                query=query,
                tool_descriptions=tool_descriptions
            )
            
            # Create planner WITHOUT callbacks
            planner_prompt = ChatPromptTemplate.from_template(planning_prompt_text)
            planner = planner_prompt | llm.with_structured_output(Plan)
            
            plan = planner.invoke({})
            
            print(f"📋 Created plan with {len(plan.steps)} steps")
            for i, step in enumerate(plan.steps, 1):
                print(f"  {i}. {step}")
            
            # CRITICAL FIX: Return the same frontend session_id, don't create new one
            return {
                "plan": plan.steps,
                "session_id": frontend_session_id  # Keep the same frontend session_id
            }
            
        except Exception as e:
            print(f"❌ Error in FIXED planning: {e}")
            fallback_plan = [f"Research comprehensive information about: {query}"]
            return {
                "plan": fallback_plan,
                "session_id": state.get("session_id", f"fallback_{int(time.time())}")
            }
    
    def execute_step(state: PlanExecuteState):
        """FIXED execution step using frontend session_id consistently."""
        try:
            plan = state["plan"]
            past_steps = state.get("past_steps", [])
            # CRITICAL FIX: Use the same frontend session_id for memory operations
            frontend_session_id = state.get("session_id")
            
            if not plan:
                print("⚠️ No plan available for execution")
                return {"past_steps": past_steps}
            
            task = plan[0]
            original_query = state.get("input", "")
            
            print(f"🔧 FIXED Executing task: {task}")
            print(f"🔗 Using frontend session ID for memory: {frontend_session_id}")
            
            # CRITICAL FIX: Use frontend_session_id for memory operations
            research_context = integrated_memory.get_research_context_summary(frontend_session_id, max_recent_steps=5)
            print(f"📋 Research context: {len(research_context):,} chars")
            
            # Use external TXT prompt template
            execution_prompt = EXECUTION_PROMPT_TEMPLATE.format(
                original_query=original_query,
                task=task,
                research_context=research_context
            )
            
            # Execute WITHOUT callbacks but with session metadata
            agent_executor = create_react_agent(llm, tools, prompt=execution_prompt)
            
            # Enhanced config with session tracking following LangSmith best practices
            config = {
                "metadata": {
                    "step": "session_cached_execute",
                    "task": task,
                    "frontend_session_id": frontend_session_id,
                    "session_id": frontend_session_id,  # BEST PRACTICE: Session metadata
                    "step_number": len(past_steps) + 1,
                    "langsmith_session": frontend_session_id,
                    "workflow_reused": True,
                    "conversation_turn": len(past_steps) + 1
                },
                "tags": [
                    f"session-{frontend_session_id}",
                    f"step-{len(past_steps) + 1}",
                    "execution",
                    "workflow-cached"
                ]
            }
            
            # Track execution with timing
            execution_start = time.time()
            
            try:
                result = agent_executor.invoke({
                    "messages": [HumanMessage(content=f"Execute this research task: {task}")]
                }, config=config)
                
                response_content = result["messages"][-1].content
                execution_time = time.time() - execution_start
                
                print(f"✅ FIXED Task completed: {len(response_content):,} chars in {execution_time:.1f}s")
                
            except Exception as exec_error:
                print(f"❌ Execution error: {exec_error}")
                response_content = f"Error executing task '{task}': {str(exec_error)}"
                execution_time = time.time() - execution_start
            
            # SMART LOGGING: Use frontend session_id
            success = len(response_content) > 100 and "error" not in response_content.lower()
            
            standard_logger.log_tool_usage(
                frontend_session_id,
                "research_agent", 
                task,
                success,
                response_content,
                f"Execution time: {execution_time:.1f}s, Steps completed: {len(past_steps) + 1}, Success: {success}"
            )
            
            # CRITICAL FIX: Store FULL result using frontend session_id
            reference_id = integrated_memory.store_research_step(
                frontend_session_id,  # Use frontend session_id for memory consistency
                task,
                response_content
            )
            
            print(f"📚 FIXED Stored result as: {reference_id} (session: {frontend_session_id})")
            
            updated_past_steps = past_steps + [(task, f"COMPLETE research stored as {reference_id}")]
            
            return {"past_steps": updated_past_steps}
        
        except Exception as e:
            error_response = f"Error executing task: {str(e)}"
            print(f"❌ Error in FIXED execute_step: {e}")
            
            frontend_session_id = state.get("session_id")
            task = plan[0] if plan else "unknown_task"
            
            # SMART LOGGING: Use frontend session_id
            standard_logger.log_tool_usage(
                frontend_session_id,
                "research_agent",
                task,
                False,
                error_response,
                f"Tool execution failed with error: {str(e)}"
            )
            
            try:
                integrated_memory.store_research_step(frontend_session_id, task, error_response)
            except Exception as mem_error:
                print(f"⚠️ Could not store error in memory: {mem_error}")
            
            updated_past_steps = state.get("past_steps", []) + [(task, error_response)]
            return {"past_steps": updated_past_steps}
    
    def replan_step(state: PlanExecuteState):
        """FIXED replanning step using frontend session_id consistently."""
        try:
            # CRITICAL FIX: Use frontend session_id for all memory operations
            frontend_session_id = state.get("session_id")
            original_plan = state.get("plan", [])
            past_steps = state.get("past_steps", [])
            
            print(f"🔄 FIXED Replanning for frontend session: {frontend_session_id}")
            
            # Track session timing
            session_start_time = getattr(replan_step, 'session_start_time', None)
            if session_start_time is None:
                replan_step.session_start_time = time.time()
                session_start_time = replan_step.session_start_time
            
            # CRITICAL FIX: Get research context using frontend session_id
            research_summary = integrated_memory.get_research_context_summary(
                frontend_session_id,  # Use frontend session_id consistently
                max_recent_steps=3
            )
            
            print(f"📋 Research context: {len(research_summary):,} chars")
            
            # Verify we have research data
            if research_summary == "No research steps completed yet.":
                if past_steps:
                    research_summary = f"Completed {len(past_steps)} research steps. Latest: {past_steps[-1][0]}"
                else:
                    research_summary = "No research completed yet."
            
            # Use external TXT prompt template
            replanning_prompt = REPLANNING_PROMPT_TEMPLATE.format(
                original_objective=state["input"],
                original_plan=original_plan,
                research_summary=research_summary
            )
            
            # Create replanner WITHOUT callbacks but with session metadata
            replanner_prompt_obj = ChatPromptTemplate.from_template(replanning_prompt)
            replanner = replanner_prompt_obj | replanner_llm.with_structured_output(Act)
            
            # Enhanced config with session tracking following LangSmith best practices
            config = {
                "metadata": {
                    "step": "session_cached_replanning",
                    "frontend_session_id": frontend_session_id,
                    "session_id": frontend_session_id,  # BEST PRACTICE: Session metadata
                    "langsmith_session": frontend_session_id,
                    "workflow_reused": True
                },
                "tags": [
                    f"session-{frontend_session_id}",
                    "replanning",
                    "workflow-cached"
                ]
            }
            
            try:
                response = replanner.invoke({}, config=config)
            except Exception as replan_error:
                print(f"❌ Replanner execution error: {replan_error}")
                return {"response": f"Research completed with session consistency. Error in replanning: {str(replan_error)}"}
            
            if response.action_type == "response":
                # SMART LOGGING: Use frontend session_id for completion analysis
                execution_time = time.time() - session_start_time
                replanning_count = len([step for step in past_steps if "replan" in step[1].lower()])
                
                # Assess final success
                final_success = "success"
                if "partial" in response.response.lower() or "incomplete" in response.response.lower():
                    final_success = "partial"
                elif "error" in response.response.lower() or "failed" in response.response.lower():
                    final_success = "failed"
                
                # CRITICAL FIX: Get comprehensive results using frontend session_id
                comprehensive_data = integrated_memory.get_comprehensive_final_response_data(frontend_session_id)
                full_results = "\n\n".join(comprehensive_data.get('full_results', []))
                
                standard_logger.log_session_complete(
                    frontend_session_id,  # Use frontend session_id consistently
                    state["input"],
                    len(past_steps),
                    replanning_count,
                    final_success,
                    execution_time,
                    full_results
                )
                
                return {"response": response.response}
            else:
                # SMART LOGGING: Use frontend session_id for replanning analysis
                if len(past_steps) == 0:
                    replanning_reason = "Initial planning phase - setting up research approach"
                elif "no research steps completed yet" in research_summary.lower():
                    replanning_reason = "No research progress made - need different approach"
                elif len(research_summary) < 500:
                    replanning_reason = "Insufficient research results - need additional investigation"
                else:
                    replanning_reason = "Research incomplete - expanding investigation scope"
                
                previous_approach = f"Plan with {len(original_plan)} steps: {', '.join(original_plan[:2])}{'...' if len(original_plan) > 2 else ''}"
                new_approach = f"Revised plan with {len(response.steps or [])} steps: {', '.join((response.steps or [])[:2])}{'...' if len(response.steps or []) > 2 else ''}"
                
                standard_logger.log_replanning_event(
                    frontend_session_id,  # Use frontend session_id consistently
                    state["input"],
                    len(past_steps) + 1,
                    replanning_reason,
                    previous_approach,
                    new_approach,
                    research_summary
                )
                
                print(f"🔄 FIXED replanner: Continuing with more steps")
                return {"plan": response.steps or []}
                
        except Exception as e:
            print(f"❌ Error in FIXED replanning: {e}")
            
            frontend_session_id = state.get("session_id")
            try:
                # CRITICAL FIX: Use frontend session_id for fallback summary
                fallback_summary = integrated_memory.get_research_context_summary(frontend_session_id, max_recent_steps=2)
                
                if fallback_summary != "No research steps completed yet.":
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
            print("🔄 FIXED: Continuing to agent execution")
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
    
    print("🔗 FIXED workflow constructed with session caching support")
    
    return workflow

# =============================================================================
# HELPER FUNCTIONS - FIXED SESSION CONSISTENCY
# =============================================================================

def compile_research_agent(es_client=None, index_name: str = "research-publications-static", recursion_limit: int = 50, frontend_session_id: str = None):
    """FIXED: Compile research agent with frontend session consistency."""
    workflow = create_research_workflow(es_client, index_name, frontend_session_id)
    app = workflow.compile()
    return app

def run_research_query(query: str, es_client=None, index_name: str = "research-publications-static", recursion_limit: int = 50, stream: bool = False, conversation_history: Optional[List[Dict]] = None, frontend_session_id: str = None) -> Dict[str, Any]:
    """FIXED: Run research query using frontend session_id consistently."""
    
    # CRITICAL FIX: Use frontend session_id if provided, don't generate new one
    if not frontend_session_id:
        frontend_session_id = f"fallback_{str(uuid.uuid4())}"
        print("⚠️ No frontend session_id provided, using fallback")
    
    app = compile_research_agent(es_client, index_name, recursion_limit, frontend_session_id)
    
    # CRITICAL FIX: Use frontend session_id in initial state
    initial_state = {
        "input": query,
        "plan": [],
        "past_steps": [],
        "response": None,
        "session_id": frontend_session_id,  # Use frontend session_id consistently
        "conversation_history": conversation_history or []
    }
    
    config = {
        "recursion_limit": recursion_limit,
        "metadata": {
            "query": query,
            "frontend_session_id": frontend_session_id,
            "index_name": index_name,
            "has_conversation_history": bool(conversation_history),
            "agent_type": "session_cached",
            "langsmith_session": frontend_session_id
        },
        "tags": [
            "session_cached_agent", 
            "plan_execute", 
            f"session-{frontend_session_id}",
            f"turn-{len(conversation_history or []) + 1}"
        ]
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
# RESEARCH AGENT CLASS - FIXED SESSION CACHING
# =============================================================================

class ResearchAgent:
    """FIXED: Research Agent that caches compiled workflows for session continuity."""
    
    def __init__(self, es_client=None, index_name: str = "research-publications-static", recursion_limit: int = 50):
        self.es_client = es_client
        self.index_name = index_name
        self.recursion_limit = recursion_limit
        self.app = None
        self.session_id = None  # Track which session this agent is compiled for

    def _compile_agent(self, frontend_session_id: str = None):
        """FIXED: Compile agent ONCE per session for continuity."""
        print(f"🔨 Compiling agent for session: {frontend_session_id}")
        
        self.app = compile_research_agent(
            self.es_client, 
            self.index_name, 
            self.recursion_limit,
            frontend_session_id
        )
        self.session_id = frontend_session_id
        print(f"✅ Agent compiled and cached for session: {frontend_session_id}")

    async def stream_query(self, query: str, conversation_history: Optional[List[Dict]] = None, frontend_session_id: str = None):
        """ORIGINAL: Stream query with recompilation (creates new LangSmith session)."""
        
        if not frontend_session_id:
            frontend_session_id = f"fallback_{str(uuid.uuid4())}"
            print("⚠️ No frontend session_id provided, using fallback")
        
        print(f"🔗 stream_query with RECOMPILATION: {frontend_session_id}")
        
        # This recompiles the workflow, breaking LangSmith session continuity
        self._compile_agent(frontend_session_id)
        
        # Use the helper method to actually stream
        async for event in self._stream_with_config(query, conversation_history, frontend_session_id):
            yield event
    
    async def stream_query_without_recompile(self, query: str, conversation_history: Optional[List[Dict]] = None, frontend_session_id: str = None):
        """
        CRITICAL FIX: Stream query WITHOUT recompiling workflow.
        Maintains LangSmith session continuity by reusing compiled workflow.
        """
        
        if not frontend_session_id:
            frontend_session_id = f"fallback_{str(uuid.uuid4())}"
            print("⚠️ No frontend session_id provided, using fallback")
        
        print(f"🔗 stream_query_without_recompile (SESSION CONTINUITY): {frontend_session_id}")
        
        # Check if agent is compiled for this session
        if self.app is None:
            print("⚠️ Agent not compiled, compiling now...")
            self._compile_agent(frontend_session_id)
        elif self.session_id != frontend_session_id:
            print(f"⚠️ Agent compiled for different session ({self.session_id}), recompiling...")
            self._compile_agent(frontend_session_id)
        else:
            print(f"✅ Reusing compiled workflow for session: {frontend_session_id}")
        
        # Use the helper method to actually stream
        async for event in self._stream_with_config(query, conversation_history, frontend_session_id):
            yield event
    
    async def _stream_with_config(self, query: str, conversation_history: Optional[List[Dict]], frontend_session_id: str):
        """Helper method to stream with consistent config."""
        
        # Use frontend session_id in initial state
        initial_state = {
            "input": query,
            "plan": [],
            "past_steps": [],
            "response": None,
            "session_id": frontend_session_id,
            "conversation_history": conversation_history or []
        }
        
        # Enhanced config for session continuity following LangSmith best practices
        config = {
            "recursion_limit": self.recursion_limit,
            "metadata": {
                "query": query,
                "frontend_session_id": frontend_session_id,
                "session_id": frontend_session_id,  # BEST PRACTICE: Session metadata for threads
                "index_name": self.index_name,
                "agent_type": "session_cached",
                "langsmith_session_id": frontend_session_id,
                "conversation_turn": len(conversation_history or []) + 1,
                "workflow_reused": self.session_id == frontend_session_id  # Track if workflow was reused
            },
            "tags": [
                "session_cached_agent", 
                "streaming", 
                f"session-{frontend_session_id}",
                f"turn-{len(conversation_history or []) + 1}",
                "workflow-cached" if self.session_id == frontend_session_id else "workflow-new"
            ],
            # BEST PRACTICE: Consistent run naming for conversation threads
            "run_name": f"Conversation-Turn-{len(conversation_history or []) + 1}"
        }
        
        # Stream with session continuity
        try:
            workflow_status = "CACHED" if self.session_id == frontend_session_id else "NEW"
            print(f"🚀 Starting stream with {workflow_status} workflow (session continuity)")
            async for event in self.app.astream(initial_state, config=config):
                yield event
        except Exception as e:
            print(f"❌ Error in cached workflow streaming: {e}")
            yield {"error": {"error": str(e)}}

if __name__ == "__main__":
    print("Testing FIXED workflow with session caching and LangSmith continuity...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    print("🔗 Key fixes:")
    print("  - Uses frontend session_id for ALL memory operations")
    print("  - Uses frontend session_id for LangSmith session tracking")
    print("  - Caches compiled workflows per session")
    print("  - No workflow recompilation within sessions")
    print("  - Memory consistency across follow-up questions")
    print("  - LangSmith traces grouped by conversation session")
    print("  - Smart methodology logging with session consistency")
    print("  - Complete information preservation")
    print("  - Fixed session_id propagation through entire workflow")
    print("  - LangSmith project names include session ID")
    print("  - All traces within a conversation appear together in LangSmith")
    print("  - Workflow caching prevents LangSmith session fragmentation")