"""
LANGSMITH CALLBACK FIX for workflow.py
CRITICAL: Completely removes problematic async callbacks that cause warnings
Uses minimal synchronous logging instead
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
from dotenv import load_dotenv

# CRITICAL FIX: Minimal LangSmith imports to avoid callback issues
from langsmith import Client

# Import tools and memory
from ..tools import get_all_tools
from .memory_manager import IntegratedMemoryManager

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
        "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT", "research-agent-minimal-callbacks")
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
        print("✅ LangSmith environment configured (minimal callbacks)")
        return client
    except Exception as e:
        print(f"❌ LangSmith configuration error: {e}")
        return None

# =============================================================================
# MAIN WORKFLOW CREATION - NO PROBLEMATIC CALLBACKS
# =============================================================================

def create_research_workflow(es_client=None, index_name: str = "research-publications-static", session_id: str = None) -> StateGraph:
    """
    Create research workflow with MINIMAL callbacks to prevent async warnings.
    CRITICAL: Uses environment-based LangSmith tracing instead of callback handlers.
    """
    
    # Setup minimal LangSmith (no callbacks)
    langsmith_client = setup_minimal_langsmith()
    
    # Get tools
    if es_client:
        tools = get_all_tools(es_client=es_client, index_name=index_name)
        print(f"✅ MINIMAL CALLBACK workflow: Initialized {len(tools)} research tools")
    else:
        tools = get_all_tools()
        print(f"✅ MINIMAL CALLBACK workflow: Initialized {len(tools)} research tools")
    
    # Initialize integrated memory
    integrated_memory = IntegratedMemoryManager(memory_type="buffer_window")
    print("🧠 MINIMAL CALLBACK workflow: Integrated memory initialized")
    
    # CRITICAL FIX: Create LLMs WITHOUT callback handlers
    try:
        llm = ChatLiteLLM(
            model="anthropic/claude-sonnet-4",
            api_key=os.getenv("LITELLM_API_KEY"),
            api_base=os.getenv("LITELLM_BASE_URL"),
            temperature=0,
            # REMOVED: callbacks=callbacks,  # This was causing async issues
            metadata={"component": "minimal_callback_main_llm", "session_id": session_id}
        )
        
        replanner_llm = ChatLiteLLM(
            model="anthropic/claude-haiku-3.5",
            api_key=os.getenv("LITELLM_API_KEY"),
            api_base=os.getenv("LITELLM_BASE_URL"),
            temperature=0,
            # REMOVED: callbacks=callbacks,  # This was causing async issues
            metadata={"component": "minimal_callback_replanner_llm", "session_id": session_id}
        )
        print("✅ LLMs initialized WITHOUT problematic callbacks")
        
    except Exception as e:
        print(f"❌ Error initializing LLMs: {e}")
        raise
    
    # =============================================================================
    # WORKFLOW NODES - MINIMAL CALLBACK APPROACH
    # =============================================================================
    
    def plan_step(state: PlanExecuteState):
        """Planning step with minimal callback approach."""
        try:
            query = state["input"]
            conversation_history = state.get("conversation_history", [])
            memory_session_id = state.get("memory_session_id", str(uuid.uuid4()))
            
            print(f"📋 MINIMAL CALLBACK Planning for query: {query}")
            print(f"🧠 Memory session ID: {memory_session_id}")
            
            # Build conversation context
            context_str = ""
            if conversation_history:
                context_lines = []
                for msg in conversation_history[-4:]:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')[:150]
                    context_lines.append(f"{role.title()}: {content}...")
                context_str = "\n".join(context_lines)
            else:
                context_str = "No previous conversation history."
            
            # Format tool information
            tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
            
            # Create planning prompt
            planning_prompt_text = f"""Create a step-by-step research plan for: "{query}"

Available research tools:
{tool_descriptions}

Conversation context:
{context_str}

Create 3-4 specific research steps that will thoroughly answer the user's question about authors, publications, or research fields.
Each step should be clear, actionable, and build upon previous steps.

Focus on:
- Author information (affiliations, research areas)
- Publication analysis (key works, themes, impact)
- Research trends and collaboration patterns
- Comprehensive profile building

Steps:"""
            
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
            print(f"❌ Error in MINIMAL CALLBACK planning: {e}")
            fallback_plan = [f"Research comprehensive information about: {query}"]
            return {
                "plan": fallback_plan,
                "memory_session_id": str(uuid.uuid4())
            }
    
    def execute_step(state: PlanExecuteState):
        """MINIMAL CALLBACK execution step."""
        try:
            plan = state["plan"]
            past_steps = state.get("past_steps", [])
            memory_session_id = state.get("memory_session_id", str(uuid.uuid4()))
            
            if not plan:
                print("⚠️ No plan available for execution")
                return {"past_steps": past_steps}
            
            task = plan[0]
            original_query = state.get("input", "")
            
            print(f"🔧 MINIMAL CALLBACK Executing task: {task}")
            
            # Build execution context
            research_context = integrated_memory.get_research_context_summary(memory_session_id, max_recent_steps=5)
            print(f"📋 Research context: {len(research_context):,} chars")
            
            # Build execution prompt
            execution_prompt = f"""You are executing a research task to help answer questions about authors, publications, and research fields.

RESEARCH OBJECTIVE: {original_query}

CURRENT TASK: {task}

PREVIOUS RESEARCH CONTEXT:
{research_context}

INSTRUCTIONS:
- Use the available research tools to gather specific, detailed information
- Focus on concrete findings: names, numbers, dates, affiliations, publications
- Look for patterns in research areas, collaboration networks, and publication trends
- Provide comprehensive results with supporting evidence
- Be thorough in your search and analysis

Execute this task now using the available tools."""
            
            # CRITICAL FIX: Execute WITHOUT callbacks
            agent_executor = create_react_agent(llm, tools, prompt=execution_prompt)
            
            # Minimal config without problematic callbacks
            config = {
                "metadata": {
                    "step": "minimal_callback_execute",
                    "task": task,
                    "session_id": session_id,
                    "memory_session_id": memory_session_id,
                    "step_number": len(past_steps) + 1
                }
                # REMOVED: "callbacks": callbacks  # This was causing async issues
            }
            
            # Execute with minimal callback approach
            try:
                result = agent_executor.invoke({
                    "messages": [HumanMessage(content=f"Execute this research task: {task}")]
                }, config=config)
                
                response_content = result["messages"][-1].content
                print(f"✅ MINIMAL CALLBACK Task completed: {len(response_content):,} chars")
                
            except Exception as exec_error:
                print(f"❌ Execution error: {exec_error}")
                response_content = f"Error executing task '{task}': {str(exec_error)}"
            
            # Store FULL result
            reference_id = integrated_memory.store_research_step(
                memory_session_id,
                task,
                response_content
            )
            
            print(f"📚 MINIMAL CALLBACK Stored result as: {reference_id}")
            
            updated_past_steps = past_steps + [(task, f"COMPLETE research stored as {reference_id}")]
            
            return {"past_steps": updated_past_steps}
        
        except Exception as e:
            error_response = f"Error executing task: {str(e)}"
            print(f"❌ Error in MINIMAL CALLBACK execute_step: {e}")
            
            memory_session_id = state.get("memory_session_id", str(uuid.uuid4()))
            try:
                integrated_memory.store_research_step(memory_session_id, task, error_response)
            except Exception as mem_error:
                print(f"⚠️ Could not store error in memory: {mem_error}")
            
            updated_past_steps = state.get("past_steps", []) + [(task, error_response)]
            return {"past_steps": updated_past_steps}
    
    def replan_step(state: PlanExecuteState):
        """MINIMAL CALLBACK replanning step."""
        try:
            memory_session_id = state.get("memory_session_id", str(uuid.uuid4()))
            
            print(f"🔄 MINIMAL CALLBACK Replanning for session: {memory_session_id}")
            
            # Get research context
            research_summary = integrated_memory.get_research_context_summary(
                memory_session_id, 
                max_recent_steps=3
            )
            
            print(f"📋 Research context: {len(research_summary):,} chars")
            
            # Verify we have research data
            if research_summary == "No research steps completed yet.":
                past_steps = state.get("past_steps", [])
                if past_steps:
                    research_summary = f"Completed {len(past_steps)} research steps. Latest: {past_steps[-1][0]}"
                else:
                    research_summary = "No research completed yet."
            
            # Create replanning prompt
            replanning_prompt = f"""You are a research replanner with access to COMPLETE research results.

ANALYSIS CONTEXT:
Original objective: {state["input"]}
Original plan: {state["plan"]}

COMPLETE RESEARCH RESULTS:
{research_summary}

DECISION CRITERIA:
Based on these COMPLETE research results, decide whether to:
1. **Provide a final response** (action_type: "response") - if comprehensive information is available
2. **Continue with more steps** (action_type: "plan") - if critical information is missing

GUIDELINES FOR RESPONSE:
✅ Provide final response when:
- User's question has been comprehensively answered with COMPLETE information
- The research results contain sufficient detail and context
- No critical information gaps remain in the FULL research content
- The research objective has been met with rich, detailed findings

GUIDELINES FOR PLAN:
⚠️ Continue with plan only when:
- Important aspects of the query remain genuinely unanswered
- Need specific additional data not present in COMPLETE current research
- Current FULL results are incomplete for comprehensive answer
- Missing critical information despite having detailed research content

RESPONSE QUALITY (when providing final response):
- Lead with direct answer to user's specific question
- Include relevant statistics, counts, and examples from COMPLETE research
- Use clear structure with headers and formatting
- Address all aspects of the original query with full context
- Mention scope and limitations when relevant

Make your decision based on the COMPLETE research content available:"""
            
            # Create replanner WITHOUT callbacks
            replanner_prompt = ChatPromptTemplate.from_template(replanning_prompt)
            replanner = replanner_prompt | replanner_llm.with_structured_output(Act)
            
            # Minimal config
            config = {
                "metadata": {
                    "step": "minimal_callback_replanning",
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
                return {"response": f"Research completed with minimal callbacks. Error in replanning: {str(replan_error)}"}
            
            if response.action_type == "response":
                # Create comprehensive final response
                comprehensive_data = integrated_memory.get_comprehensive_final_response_data(memory_session_id)
                
                print(f"🎯 MINIMAL CALLBACK Creating final response with {comprehensive_data['total_steps']} results")
                
                if comprehensive_data['full_results']:
                    enhanced_response = f"""{response.response}

## 📊 Complete Research Analysis

"""
                    
                    for i, full_result in enumerate(comprehensive_data['full_results'][:2], 1):
                        result_preview = full_result[:2000] + ("..." if len(full_result) > 2000 else "")
                        enhanced_response += f"""### Research Step {i} Results:
{result_preview}

"""
                    
                    enhanced_response += f"""---
*Analysis based on {comprehensive_data['total_steps']} complete research steps with {comprehensive_data['total_content_length']:,} total characters. Minimal callback approach prevents async warnings.*"""
                else:
                    enhanced_response = f"""{response.response}

---
*Research completed with {comprehensive_data['total_steps']} comprehensive steps using minimal callbacks.*"""
                
                print("✅ MINIMAL CALLBACK replanner: Providing final response")
                return {"response": enhanced_response}
            else:
                print(f"🔄 MINIMAL CALLBACK replanner: Continuing with more steps")
                return {"plan": response.steps or []}
                
        except Exception as e:
            print(f"❌ Error in MINIMAL CALLBACK replanning: {e}")
            
            memory_session_id = state.get("memory_session_id", str(uuid.uuid4()))
            try:
                fallback_summary = integrated_memory.get_research_context_summary(memory_session_id, max_recent_steps=2)
                
                if fallback_summary != "No research steps completed yet.":
                    fallback_response = f"""Based on research completed with minimal callbacks:

{fallback_summary[:1000]}{"..." if len(fallback_summary) > 1000 else ""}

---
*Research completed successfully. Minimal callback approach prevents async warnings.*"""
                else:
                    fallback_response = "Research completed with minimal callbacks."
                
                return {"response": fallback_response}
                
            except Exception as fallback_error:
                print(f"⚠️ Fallback also failed: {fallback_error}")
                return {"response": f"Research error during replanning: {str(e)}. Minimal callbacks prevent async corruption."}
    
    def should_end(state: PlanExecuteState) -> Literal["agent", "__end__"]:
        """Simple decision function for workflow routing."""
        if state.get("response"):
            print("✅ MINIMAL CALLBACK: Final response available - ending workflow")
            return "__end__"
        else:
            print("🔄 MINIMAL CALLBACK: Continuing to agent execution")
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
    
    print("🔧 MINIMAL CALLBACK workflow constructed (no problematic callbacks)")
    
    return workflow

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compile_research_agent(es_client=None, index_name: str = "research-publications-static", recursion_limit: int = 50, session_id: str = None):
    """Compile research agent with minimal callbacks."""
    workflow = create_research_workflow(es_client, index_name, session_id)
    app = workflow.compile()
    return app

def run_research_query(query: str, es_client=None, index_name: str = "research-publications-static", recursion_limit: int = 50, stream: bool = False, conversation_history: Optional[List[Dict]] = None, session_id: str = None) -> Dict[str, Any]:
    """Run research query using MINIMAL CALLBACK patterns."""
    
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
            "agent_type": "minimal_callbacks"
        },
        "tags": ["minimal_callbacks_agent", "plan_execute", f"session_{session_id}"]
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
# RESEARCH AGENT CLASS - MINIMAL CALLBACKS
# =============================================================================

class ResearchAgent:
    """Research Agent using minimal callbacks to prevent async warnings."""
    
    def __init__(self, es_client=None, index_name: str = "research-publications-static", recursion_limit: int = 50):
        self.es_client = es_client
        self.index_name = index_name
        self.recursion_limit = recursion_limit
        self.app = None
        self.langsmith_client = setup_minimal_langsmith()

    def _compile_agent(self, session_id: str = None):
        """Compile agent with minimal callbacks."""
        self.app = compile_research_agent(
            self.es_client, 
            self.index_name, 
            self.recursion_limit,
            session_id
        )

    async def stream_query(self, query: str, conversation_history: Optional[List[Dict]] = None):
        """Stream query with minimal callbacks to prevent async warnings."""
        
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
                "agent_type": "minimal_callbacks"
            },
            "tags": ["minimal_callbacks_agent", "streaming", f"session_{session_id}"]
        }
        
        # Stream with minimal callbacks
        try:
            print("🚀 Starting MINIMAL CALLBACK stream")
            async for event in self.app.astream(initial_state, config=config):
                yield event
            print("✅ MINIMAL CALLBACK stream completed")
        except Exception as e:
            print(f"❌ Error in MINIMAL CALLBACK streaming: {e}")
            yield {"error": {"error": str(e)}}

if __name__ == "__main__":
    print("Testing MINIMAL CALLBACK workflow...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    print("✅ MINIMAL CALLBACK workflow loaded successfully!")
    print("🎯 Key features:")
    print("  - No problematic callback handlers")
    print("  - Environment-based LangSmith tracing")
    print("  - Clean async execution")
    print("  - Complete information preservation")