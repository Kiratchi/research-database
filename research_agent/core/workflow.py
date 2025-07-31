"""
COMBINED Enhanced workflow.py - Best of Both Worlds
COMBINES: Smart methodology learning + Plan-Execute architecture + LangChain automatic memory
BUILDS ON: Original plan-execute workflow with LangChain memory integration
FILENAME: workflow.py
"""

from typing import Dict, Any, List, Optional, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_react_agent, AgentExecutor 
from langchain_litellm import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
import os
import uuid
import asyncio
import time
from dotenv import load_dotenv


# CRITICAL FIX: Minimal LangSmith imports to avoid callback issues
from langsmith import Client

# Import LangChain memory and tools
from .memory_manager import SessionMemoryManager
from .methodology_logger import SmartMethodologyLogger

# Import tools
try:
    from ..tools import get_all_tools
    TOOLS_AVAILABLE = True
except ImportError:
    print("⚠️ Tools not available - falling back to basic functionality")
    TOOLS_AVAILABLE = False

# =============================================================================
# STATE SCHEMA - Enhanced with LangChain Memory
# =============================================================================

class PlanExecuteState(TypedDict):
    """State schema with LangChain memory integration."""
    input: str
    plan: List[str]
    past_steps: List[tuple[str, str]]
    response: Optional[str]
    
    # LangChain memory integration (session_id is REQUIRED)
    session_id: str  # Required for memory continuity - no more auto-generation!
    chat_history: Optional[List[Any]]  # LangChain injects this automatically

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
        "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT", "combined-research-agent")
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
        print("✅ LangSmith environment configured with combined methodology tracking")
        return client
    except Exception as e:
        print(f"❌ LangSmith configuration error: {e}")
        return None

# =============================================================================
# COMBINED WORKFLOW CREATION WITH SMART METHODOLOGY + LANGCHAIN MEMORY
# =============================================================================

def create_combined_research_workflow(
    es_client=None, 
    index_name: str = "research-publications-static", 
    session_id: str = None,
    memory_manager: SessionMemoryManager = None
) -> StateGraph:
    """
    Create COMBINED research workflow: Plan-Execute + LangChain Memory + Smart Methodology Learning.
    COMBINES: Best features from both approaches with automatic conversation context.
    """
    
    # Setup minimal LangSmith (no callbacks)
    langsmith_client = setup_minimal_langsmith()
    
    # Get tools
    if TOOLS_AVAILABLE:
        if es_client:
            tools = get_all_tools(es_client=es_client, index_name=index_name)
            print(f"✅ COMBINED workflow: Initialized {len(tools)} research tools")
        else:
            tools = get_all_tools()
            print(f"✅ COMBINED workflow: Initialized {len(tools)} research tools")
    else:
        tools = []
        print("⚠️ COMBINED workflow: No tools available")
    
    # Use provided memory manager or create new one
    if memory_manager is None:
        memory_manager = SessionMemoryManager(default_memory_type="buffer_window")
        print("🧠 COMBINED workflow: Created new LangChain memory manager")
    else:
        print("🧠 COMBINED workflow: Using provided LangChain memory manager")
    
    # Initialize smart methodology logger
    try:
        smart_logger = SmartMethodologyLogger()
        print("🧠 Smart Methodology Learning system active with LLM analysis")
    except Exception as e:
        print(f"⚠️ Smart Methodology Logger initialization failed: {e}")
        smart_logger = None
    
    # CRITICAL FIX: Create LLMs WITHOUT callback handlers
    try:
        llm = ChatLiteLLM(
            model="anthropic/claude-sonnet-4",
            api_key=os.getenv("LITELLM_API_KEY"),
            api_base=os.getenv("LITELLM_BASE_URL"),
            temperature=0,
            metadata={"component": "combined_main_llm", "session_id": session_id}
        )
        
        replanner_llm = ChatLiteLLM(
            model="anthropic/claude-haiku-3.5",
            api_key=os.getenv("LITELLM_API_KEY"),
            api_base=os.getenv("LITELLM_BASE_URL"),
            temperature=0,
            metadata={"component": "combined_replanner_llm", "session_id": session_id}
        )
        print("✅ LLMs initialized WITHOUT problematic callbacks + combined methodology tracking")
        
    except Exception as e:
        print(f"❌ Error initializing LLMs: {e}")
        raise
    
    # =============================================================================
    # COMBINED WORKFLOW NODES WITH SMART METHODOLOGY + LANGCHAIN MEMORY
    # =============================================================================
    
    def plan_step(state: PlanExecuteState):
        """Planning step with LangChain memory context and smart analysis."""
        try:
            query = state["input"]
            session_id = state["session_id"]
            
            if not session_id:
                raise ValueError("session_id is required for memory continuity")
            
            print(f"📋 COMBINED Planning for query: {query}")
            print(f"🧠 Session ID: {session_id}")
            
            # Get LangChain conversation history
            conversation_history = memory_manager.get_conversation_history(session_id)
            
            # SMART LOGGING: Intelligent query analysis with conversation context
            is_followup = len(conversation_history) > 0
            previous_context = ""
            if is_followup and conversation_history:
                prev_user_messages = [msg for msg in conversation_history if msg.get('role') == 'user']
                prev_ai_messages = [msg for msg in conversation_history if msg.get('role') == 'assistant']
                
                if prev_user_messages and prev_ai_messages:
                    previous_context = f"Previous query: {prev_user_messages[-1].get('content', '')}\nPrevious response: {prev_ai_messages[-1].get('content', '')[:300]}"
            
            if smart_logger:
                smart_logger.log_query_start(
                    session_id, query, is_followup, previous_context
                )
            
            # Format tool information
            tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools]) if tools else "No tools available"
            
            # COMBINED: Use LangChain memory for automatic context injection
            planning_prompt_text = """Create a step-by-step research plan for: "{input}"

Previous conversation:
{chat_history}

Available research tools:
{tool_descriptions}

Create 3-4 specific research steps that will thoroughly answer the user's question about authors, publications, or research fields.
Each step should be clear, actionable, and build upon previous steps.

If this is a follow-up question based on our previous conversation:
- Reference previously discussed authors, topics, or findings
- Build upon earlier research results without repeating identical searches
- Use pronouns and context naturally ("his publications", "her research areas")
- Avoid redundant searches for information we've already gathered

Focus on:
- Author information (affiliations, research areas)
- Publication analysis (key works, themes, impact)
- Research trends and collaboration patterns
- Comprehensive profile building

Steps:"""
            
            # Create planner with LangChain memory context
            planner_prompt = ChatPromptTemplate.from_template(planning_prompt_text)
            
            # Get memory for automatic context injection
            memory = memory_manager.get_memory_for_session(session_id)
            memory_vars = memory.load_memory_variables({})
            
            planner = planner_prompt | llm.with_structured_output(Plan)
            
            plan = planner.invoke({
                "input": query,
                "chat_history": memory_vars.get("chat_history", []),
                "tool_descriptions": tool_descriptions
            })
            
            print(f"📋 Created plan with {len(plan.steps)} steps:")
            for i, step in enumerate(plan.steps, 1):
                print(f"  {i}. {step}")
            
            return {
                "plan": plan.steps,
                "chat_history": memory_vars.get("chat_history", [])
            }
            
        except Exception as e:
            print(f"❌ Error in COMBINED planning: {e}")
            fallback_plan = [f"Research comprehensive information about: {query}"]
            return {
                "plan": fallback_plan,
                "chat_history": []
            }
    
    def execute_step(state: PlanExecuteState):
        """Execution step with Claude-compatible LangChain ReAct agent and tool effectiveness logging."""
        try:
            plan = state["plan"]
            past_steps = state.get("past_steps", [])
            session_id = state["session_id"]
            
            if not plan:
                print("⚠️ No plan available for execution")
                return {"past_steps": past_steps}
            
            task = plan[0]
            original_query = state.get("input", "")
            print(f"🔧 COMBINED Executing task: {task}")

            if tools:
                execution_start = time.time()

                try:
                    # Create a Claude-friendly task prompt
                    context_aware_input = f"""Research Objective: {original_query}

Current Task: {task}

Instructions:
- Use the available research tools to gather specific, detailed information
- This is part of a multi-step research plan to answer: "{original_query}"
- Focus on concrete findings: names, numbers, dates, affiliations, publications
- Look for patterns in research areas, collaboration networks, and publication trends
- Provide comprehensive results with supporting evidence

Execute this research task now."""

                    # Define Claude-compatible prompt
                    prompt = ChatPromptTemplate.from_messages([
                        ("system",
                        "You are a helpful research assistant that uses tools to answer questions.\n\n"
                        "Available tools:\n{tools}\n\n"
                        "Tool names: {tool_names}\n\n"
                        "When you use a tool, format your response as:\n"
                        "Thought: I need to use a tool\n"
                        "Action: <tool name>\n"
                        "Action Input: <input>\n\n"
                        "Then wait for the result.\n"
                        "When you have enough information, respond with:\n"
                        "Final Answer: <answer>"
                        ),
                        ("user", "{input}"),
                        ("assistant", "{agent_scratchpad}")
                    ])

                    # Create agent using Claude and prompt
                    agent_runnable: Runnable = create_react_agent(
                        llm=llm,
                        tools=tools,
                        prompt=prompt
                    )

                    agent_executor = AgentExecutor(
                        agent=agent_runnable,
                        tools=tools,
                        verbose=True,
                        handle_parsing_errors=True,
                        max_iterations=10,
                        early_stopping_method="force",
                        return_intermediate_steps=False
                    )

                    # Prepare tool metadata for prompt injection
                    tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
                    tool_names = [tool.name for tool in tools]

                    # Invoke agent
                    result = agent_executor.invoke({
                        "input": context_aware_input,
                        "tools": tool_descriptions,
                        "tool_names": tool_names
                    })

                    response_content = result.get("output", "No response generated")
                    execution_time = time.time() - execution_start
                    print(f"✅ COMBINED Task completed: {len(response_content):,} chars in {execution_time:.1f}s")

                except Exception as exec_error:
                    print(f"❌ Execution error: {exec_error}")
                    response_content = f"Error executing task '{task}': {str(exec_error)}"
                    execution_time = time.time() - execution_start
            else:
                response_content = f"No tools available to execute task: {task}"
                execution_time = 0

            # SMART LOGGING
            success = len(response_content) > 100 and "error" not in response_content.lower()

            if smart_logger:
                smart_logger.log_tool_usage(
                    session_id,
                    "research_agent",
                    task,
                    success,
                    response_content,
                    f"Execution time: {execution_time:.1f}s, Steps completed: {len(past_steps) + 1}, Success: {success}"
                )

            updated_past_steps = past_steps + [(task, response_content)]
            return {"past_steps": updated_past_steps}

        except Exception as e:
            error_response = f"Error executing task: {str(e)}"
            print(f"❌ Error in COMBINED execute_step: {e}")

            session_id = state.get("session_id", "unknown")
            task = plan[0] if plan else "unknown_task"

            if smart_logger:
                smart_logger.log_tool_usage(
                    session_id,
                    "research_agent",
                    task,
                    False,
                    error_response,
                    f"Tool execution failed with error: {str(e)}"
                )

            updated_past_steps = state.get("past_steps", []) + [(task, error_response)]
            return {"past_steps": updated_past_steps}


    def replan_step(state: PlanExecuteState):
        """Replanning step with LangChain memory context and intelligent session analysis."""
        try:
            session_id = state.get("session_id", "unknown")
            original_plan = state.get("plan", [])
            past_steps = state.get("past_steps", [])
            
            print(f"🔄 COMBINED Replanning for session: {session_id}")
            
            # Track session timing for comprehensive analysis
            session_start_time = getattr(replan_step, 'session_start_time', None)
            if session_start_time is None:
                replan_step.session_start_time = time.time()
                session_start_time = replan_step.session_start_time
            
            # Get memory context
            memory = memory_manager.get_memory_for_session(session_id)
            memory_vars = memory.load_memory_variables({})
            
            # Build research summary from past steps
            research_summary = ""
            if past_steps:
                research_parts = []
                for step_task, step_result in past_steps[-3:]:  # Last 3 steps
                    preview = step_result[:1000] + ("..." if len(step_result) > 1000 else "")
                    research_parts.append(f"Task: {step_task}\nResult: {preview}")
                research_summary = "\n\n".join(research_parts)
            else:
                research_summary = "No research steps completed yet."
            
            print(f"📋 Research summary: {len(research_summary):,} chars")
            
            # COMBINED: Create replanning prompt with LangChain memory and research context
            replanning_prompt_text = """You are a research replanner with access to conversation history and research results.

Previous conversation:
{chat_history}

ANALYSIS CONTEXT:
Original objective: {input}
Original plan: {original_plan}

COMPLETE RESEARCH RESULTS:
{research_summary}

DECISION CRITERIA:
Based on the conversation history and complete research results, decide whether to:
1. **Provide a final response** (action_type: "response") - if comprehensive information is available
2. **Continue with more steps** (action_type: "plan") - if critical information is missing

GUIDELINES FOR RESPONSE:
✅ Provide final response when:
- User's question has been comprehensively answered with detailed information
- The research results contain sufficient detail and context
- No critical information gaps remain in the complete research content
- Previous conversation shows the research objective has been met with rich findings

GUIDELINES FOR PLAN:
⚠️ Continue with plan only when:
- Important aspects of the query remain genuinely unanswered
- Need specific additional data not present in complete current research
- Current full results are incomplete for comprehensive answer
- Missing critical information despite having detailed research content

RESPONSE QUALITY (when providing final response):
- Reference our conversation context naturally when relevant
- Include relevant statistics, counts, and examples from complete research
- Use clear structure with headers and formatting
- Address all aspects of the original query with full context
- Build upon any previous discussion points from our conversation
- Mention scope and limitations when relevant

Make your decision based on the complete context and research results:"""
            
            # Create replanner with memory and research context
            replanner_prompt = ChatPromptTemplate.from_template(replanning_prompt_text)
            replanner = replanner_prompt | replanner_llm.with_structured_output(Act)
            
            # Execute replanner with full context
            response = replanner.invoke({
                "input": state["input"],
                "chat_history": memory_vars.get("chat_history", []),
                "original_plan": original_plan,
                "research_summary": research_summary
            })
            
            if response.action_type == "response":
                # SMART LOGGING: Comprehensive session completion analysis
                execution_time = time.time() - session_start_time
                replanning_count = len([step for step in past_steps if "replan" in step[1].lower()])
                
                # Assess final success intelligently
                final_success = "success"
                if "partial" in response.response.lower() or "incomplete" in response.response.lower():
                    final_success = "partial"
                elif "error" in response.response.lower() or "failed" in response.response.lower():
                    final_success = "failed"
                
                # Get comprehensive research results for LLM analysis
                full_results = "\n\n".join([step[1] for step in past_steps[-2:]])  # Top 2 results
                
                if smart_logger:
                    smart_logger.log_session_complete(
                        session_id,
                        state["input"],
                        len(past_steps),
                        replanning_count,
                        final_success,
                        execution_time,
                        full_results
                    )
                
                # CRITICAL: Save final interaction to LangChain memory for continuity
                memory = memory_manager.get_memory_for_session(session_id)
                memory.save_context(
                    {"input": state["input"]},
                    {"output": response.response}
                )
                
                print("✅ COMBINED replanner: Providing final response with memory saved")
                return {"response": response.response}
            else:
                # SMART LOGGING: Intelligent replanning analysis
                replanning_reason = "Research incomplete - expanding investigation scope"
                if len(past_steps) == 0:
                    replanning_reason = "Initial planning phase - setting up research approach"
                elif "no research steps completed yet" in research_summary.lower():
                    replanning_reason = "No research progress made - need different approach"
                elif len(research_summary) < 500:
                    replanning_reason = "Insufficient research results - need additional investigation"
                
                previous_approach = f"Plan with {len(original_plan)} steps: {', '.join(original_plan[:2])}{'...' if len(original_plan) > 2 else ''}"
                new_approach = f"Revised plan with {len(response.steps or [])} steps: {', '.join((response.steps or [])[:2])}{'...' if len(response.steps or []) > 2 else ''}"
                
                if smart_logger:
                    smart_logger.log_replanning_event(
                        session_id,
                        state["input"],
                        len(past_steps) + 1,
                        replanning_reason,
                        previous_approach,
                        new_approach,
                        research_summary
                    )
                
                print(f"🔄 COMBINED replanner: Continuing with more steps")
                return {"plan": response.steps or []}
                
        except Exception as e:
            print(f"❌ Error in COMBINED replanning: {e}")
            
            # Try to save something to memory even on error
            session_id = state.get("session_id", "unknown")
            try:
                memory = memory_manager.get_memory_for_session(session_id)
                fallback_response = f"Research completed with combined methodology. Error in replanning: {str(e)}"
                memory.save_context(
                    {"input": state["input"]},
                    {"output": fallback_response}
                )
                return {"response": fallback_response}
            except:
                return {"response": f"Research error during replanning: {str(e)}. Combined methodology analysis captured for learning."}
    
    def should_end(state: PlanExecuteState) -> Literal["agent", "__end__"]:
        """Simple decision function for workflow routing."""
        if state.get("response"):
            print("✅ COMBINED: Final response available - ending workflow")
            return "__end__"
        else:
            print("🔄 COMBINED: Continuing to agent execution")
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
    
    print("🧠 COMBINED workflow constructed with Plan-Execute + LangChain Memory + Smart Methodology")
    
    return workflow

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compile_combined_research_agent(
    es_client=None, 
    index_name: str = "research-publications-static", 
    recursion_limit: int = 50, 
    session_id: str = None,
    memory_manager: SessionMemoryManager = None
):
    """Compile combined research agent with plan-execute + LangChain memory."""
    workflow = create_combined_research_workflow(es_client, index_name, session_id, memory_manager)
    app = workflow.compile()
    return app

def run_combined_research_query(
    query: str, 
    session_id: str,
    es_client=None, 
    index_name: str = "research-publications-static", 
    recursion_limit: int = 50,
    memory_manager: SessionMemoryManager = None
) -> Dict[str, Any]:
    """Run research query using COMBINED methodology with session continuity."""
    
    if not session_id:
        raise ValueError("session_id is REQUIRED for combined methodology with memory continuity")
    
    app = compile_combined_research_agent(es_client, index_name, recursion_limit, session_id, memory_manager)
    
    initial_state = {
        "input": query,
        "plan": [],
        "past_steps": [],
        "response": None,
        "session_id": session_id,
        "chat_history": []  # Will be populated by LangChain memory
    }
    
    config = {
        "recursion_limit": recursion_limit,
        "metadata": {
            "query": query,
            "session_id": session_id,
            "index_name": index_name,
            "agent_type": "combined_plan_execute_langchain_memory"
        },
        "tags": ["combined_methodology_agent", "plan_execute", "langchain_memory", f"session_{session_id}"]
    }
    
    result = app.invoke(initial_state, config=config)
    return result

# =============================================================================
# COMBINED RESEARCH AGENT CLASS
# =============================================================================

class CombinedResearchAgent:
    """Combined Research Agent using Plan-Execute + LangChain Memory + Smart Methodology."""
    
    def __init__(self, es_client=None, index_name: str = "research-publications-static", recursion_limit: int = 50):
        self.es_client = es_client
        self.index_name = index_name
        self.recursion_limit = recursion_limit
        self.app = None
        
        # Initialize LangChain memory manager
        self.memory_manager = SessionMemoryManager(default_memory_type="buffer_window")
        print("🧠 CombinedResearchAgent: LangChain memory manager initialized")

    def _compile_agent(self, session_id: str):
        """Compile agent with combined workflow."""
        if not session_id:
            raise ValueError("session_id is required for combined agent compilation")
        
        self.app = compile_combined_research_agent(
            self.es_client, 
            self.index_name, 
            self.recursion_limit,
            session_id,
            self.memory_manager
        )

    def execute_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """Execute query with combined workflow and LangChain memory."""
        
        if not session_id:
            return {
                "success": False,
                "error": "session_id is REQUIRED for combined methodology with memory continuity",
                "agent_type": "combined_plan_execute_langchain_memory"
            }
        
        try:
            self._compile_agent(session_id)
            
            result = run_combined_research_query(
                query, 
                session_id, 
                self.es_client, 
                self.index_name, 
                self.recursion_limit,
                self.memory_manager
            )
            
            return {
                "success": True,
                "response": result.get("response", "No response generated"),
                "session_id": session_id,
                "agent_type": "combined_plan_execute_langchain_memory",
                "memory_type": "langchain_automatic",
                "architecture": "plan_execute_with_smart_methodology"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Combined research execution failed: {str(e)}",
                "session_id": session_id,
                "agent_type": "combined_plan_execute_langchain_memory"
            }
    
    def get_conversation_context(self, session_id: str) -> str:
        """Get formatted conversation context for debugging."""
        try:
            conversation_history = self.memory_manager.get_conversation_history(session_id)
            
            if not conversation_history:
                return "No conversation history"
            
            context_parts = []
            for msg in conversation_history:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:200] + ("..." if len(msg.get('content', '')) > 200 else "")
                context_parts.append(f"{role.title()}: {content}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            return f"Error getting context: {str(e)}"
    
    def clear_session(self, session_id: str):
        """Clear memory for specific session."""
        if session_id:
            self.memory_manager.clear_session_memory(session_id)
            print(f"🗑️ Cleared combined session: {session_id}")
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for specific session."""
        return self.memory_manager.get_session_info(session_id)
    
    def get_all_sessions_stats(self) -> Dict[str, Any]:
        """Get statistics for all sessions."""
        stats = self.memory_manager.get_memory_stats()
        stats.update({
            "agent_type": "combined_plan_execute_langchain_memory",
            "architecture": "plan_execute_with_smart_methodology",
            "memory_system": "langchain_automatic"
        })
        return stats

def create_combined_research_agent(es_client=None, index_name: str = "research-publications-static") -> CombinedResearchAgent:
    """Factory function to create combined research agent."""
    return CombinedResearchAgent(es_client, index_name)

if __name__ == "__main__":
    print("Testing Combined Research Agent with Plan-Execute + LangChain Memory...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    print("✅ Combined workflow loaded successfully!")
    print("🧠 Key features:")
    print("  - Plan-and-execute architecture with intelligent replanning")
    print("  - LangChain automatic conversation memory injection via {chat_history}")
    print("  - Smart methodology learning with LLM-powered analysis")
    print("  - Tool effectiveness tracking and session analytics")
    print("  - Session continuity across multiple queries")
    print("  - No manual context building - all handled by LangChain")
    print("  - Best of both worlds: sophisticated workflow + proven memory")