"""
Main LangGraph workflow for the research publications agent - WITH LANGSMITH TRACING

Following LangChain's official plan-and-execute pattern with proper response flow
"""

from typing import Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_litellm import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Union, List
import os
from dotenv import load_dotenv
import traceback

# LangSmith imports
from langsmith import Client
from langchain_core.tracers import LangChainTracer
from langchain_core.callbacks import BaseCallbackHandler
import uuid

from .state import PlanExecuteState
from ..tools.elasticsearch_tools import (
    initialize_elasticsearch_tools, 
    create_elasticsearch_tools,
    get_tool_descriptions_for_planning,
    get_tool_descriptions_for_execution
)

from .models import Plan, Response, Act
from .prompt_loader import (
    get_executor_prompt,
    get_planner_system_prompt, 
    get_replanner_prompt,
    get_task_format_template,
    get_context_aware_prompt,
    get_standard_planning_prompt
)

# Custom callback for additional LangSmith logging
class ResearchAgentTracer(BaseCallbackHandler):
    """Custom callback handler for research agent tracing."""
    
    def __init__(self, session_id: str = None):
        super().__init__()
        self.session_id = session_id or str(uuid.uuid4())
        self.step_count = 0
        
    def on_chain_start(self, serialized, inputs, **kwargs):
        """Log when a chain starts."""
        try:
            run_id = kwargs.get('run_id')
            tags = kwargs.get('tags', [])
            
            # Add custom metadata
            metadata = {
                "session_id": self.session_id,
                "component": "research_agent",
                "step_number": self.step_count
            }
            
            # Safe access to serialized data
            chain_name = 'unknown'
            if serialized and isinstance(serialized, dict):
                chain_name = serialized.get('name', 'unknown')
            elif hasattr(serialized, 'get'):
                chain_name = serialized.get('name', 'unknown')
            
            # You can add custom logging here
            print(f"🔍 Starting chain: {chain_name} - Run ID: {run_id}")
        except Exception as e:
            # Silently handle errors in callback to avoid disrupting main flow
            pass
        
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Log when a tool starts."""
        try:
            tool_name = 'unknown_tool'
            if serialized and isinstance(serialized, dict):
                tool_name = serialized.get('name', 'unknown_tool')
            elif hasattr(serialized, 'get'):
                tool_name = serialized.get('name', 'unknown_tool')
            
            print(f"🔧 Using tool: {tool_name}")
        except Exception as e:
            # Silently handle errors in callback
            pass
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Log when an LLM call starts."""
        try:
            model_name = 'unknown_model'
            if serialized and isinstance(serialized, dict):
                model_name = serialized.get('name', 'unknown_model')
            elif hasattr(serialized, 'get'):
                model_name = serialized.get('name', 'unknown_model')
            
            print(f"🤖 LLM call: {model_name}")
        except Exception as e:
            # Silently handle errors in callback
            pass

def setup_langsmith_tracing():
    """Setup LangSmith tracing with environment variables."""
    load_dotenv()
    
    # Ensure LangSmith environment variables are set
    langsmith_config = {
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "true"),
        "LANGCHAIN_ENDPOINT": os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY"),
        "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT", "research-publications-agent")
    }
    
    # Set environment variables if not already set
    for key, value in langsmith_config.items():
        if value:
            os.environ[key] = value
    
    # Verify LangSmith is configured
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("⚠️ WARNING: LANGCHAIN_API_KEY not set. LangSmith tracing will be disabled.")
        return None
    
    try:
        # Initialize LangSmith client
        client = Client(
            api_url=os.getenv("LANGCHAIN_ENDPOINT"),
            api_key=os.getenv("LANGCHAIN_API_KEY")
        )
        
        # Test connection
        client.list_runs(limit=1)
        print("✅ LangSmith tracing initialized successfully")
        return client
        
    except Exception as e:
        print(f"❌ Failed to initialize LangSmith: {e}")
        return None

def create_research_workflow(es_client=None, index_name: str = "research-publications-static", session_id: str = None) -> StateGraph:
    """Create the research workflow with LangSmith tracing enabled."""
    
    # Setup LangSmith tracing
    langsmith_client = setup_langsmith_tracing()
    
    if es_client:
        initialize_elasticsearch_tools(es_client, index_name)

    tools = create_elasticsearch_tools()
    planning_tool_descriptions = get_tool_descriptions_for_planning()
    execution_tool_descriptions = get_tool_descriptions_for_execution()

    # Create LLM with tracing callbacks
    callbacks = []
    if langsmith_client:
        callbacks.append(ResearchAgentTracer(session_id))

    llm = ChatLiteLLM(
        model="anthropic/claude-sonnet-4", 
        api_key=os.getenv("LITELLM_API_KEY"),
        api_base=os.getenv("LITELLM_BASE_URL"),
        temperature=0,
        callbacks=callbacks,
        # Add metadata for LangSmith
        metadata={
            "component": "main_llm",
            "session_id": session_id
        }
    )

    executor_prompt = get_executor_prompt(execution_tool_descriptions)
    agent_executor = create_react_agent(
        llm, 
        tools, 
        prompt=executor_prompt
    )

    planner_system_prompt = get_planner_system_prompt(planning_tool_descriptions)
    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", planner_system_prompt),
        ("placeholder", "{messages}"),
    ])
    planner = planner_prompt | llm.with_structured_output(Plan)

    replanner_llm = ChatLiteLLM(
        model="anthropic/claude-sonnet-3.7",
        api_key=os.getenv("LITELLM_API_KEY"),
        api_base=os.getenv("LITELLM_BASE_URL"),
        temperature=0,
        callbacks=callbacks,
        metadata={
            "component": "replanner_llm",
            "session_id": session_id
        }
    )

    replanner_template = get_replanner_prompt(planning_tool_descriptions)
    replanner_prompt = ChatPromptTemplate.from_template(replanner_template)
    replanner = replanner_prompt | replanner_llm.with_structured_output(Act)

    def execute_step(state: PlanExecuteState):
        """Execute step with tracing metadata."""
        plan = state["plan"]
        past_steps = state.get("past_steps", [])
        if not plan:
            return {"past_steps": []}

        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        original_query = state.get("input", "")
        task_formatted = get_task_format_template(original_query=original_query, plan_str=plan_str, task=task)

        try:
            # Add metadata to the agent execution
            config = {
                "metadata": {
                    "step": "execute",
                    "task": task,
                    "session_id": session_id,
                    "step_number": len(past_steps) + 1
                },
                "callbacks": callbacks  # Pass callbacks in config instead
            }
            
            agent_response = agent_executor.invoke(
                {"messages": [("user", task_formatted)]},
                config=config
            )
            response_content = agent_response["messages"][-1].content
            return {"past_steps": [(task, response_content)]}
        except Exception as e:
            return {"past_steps": [(task, f"Error executing task: {str(e)}")]}

    def plan_step(state: PlanExecuteState):
        """Planning step with tracing metadata."""
        try:
            query = state["input"]
            conversation_history = state.get("conversation_history", [])
            messages = []

            if conversation_history and len(conversation_history) > 0:
                context_lines = []
                for msg in conversation_history[-4:]:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    truncated_content = content[:200] + "..." if len(content) > 200 else content
                    context_lines.append(f"- {role.title()}: {truncated_content}")

                context_summary = "\n".join(context_lines)
                context_message = get_context_aware_prompt(context_summary, query)
                messages.append(("user", context_message))
            else:
                standard_message = get_standard_planning_prompt(query)
                messages.append(("user", standard_message))

            # Add metadata for planning
            config = {
                "metadata": {
                    "step": "planning",
                    "query": query,
                    "has_context": len(conversation_history) > 0,
                    "session_id": session_id
                },
                "callbacks": callbacks  # Pass callbacks in config
            }

            plan = planner.invoke({"messages": messages}, config=config)

            if plan is None or not hasattr(plan, 'steps'):
                return {"plan": [f"Search for information about: {query}"]}
            return {"plan": plan.steps}
        except Exception as e:
            fallback_plan = [f"Search for information about: {query}"]
            return {"plan": fallback_plan}

    def replan_step(state: PlanExecuteState):
        """Replanning step with tracing metadata."""
        try:
            config = {
                "metadata": {
                    "step": "replanning",
                    "past_steps_count": len(state.get("past_steps", [])),
                    "session_id": session_id
                },
                "callbacks": callbacks  # Pass callbacks in config
            }
            
            output = replanner.invoke(state, config=config)
            if output.action_type == "response":
                return {"final_response": output.response}
            else:
                return {"plan": output.steps}
        except Exception as e:
            return {"final_response": f"Error during replanning: {str(e)}"}

    def should_continue_or_end(state: PlanExecuteState) -> Literal["replan", "complete", "__end__"]:
        plan = state.get("plan", [])
        past_steps = state.get("past_steps", [])

        if state.get("final_response"):
            return "complete"
        if not plan or not past_steps:
            return "replan"
        if len(past_steps) >= len(plan):
            return "complete"
        return "replan"

    def complete_step(state: PlanExecuteState):
        """Complete step with final response formatting."""
        if state.get("final_response"):
            final_response = state["final_response"]
        elif state.get("past_steps"):
            recent_step = state["past_steps"][-1][1]
            original_query = state.get("input", "")
            final_response = format_enhanced_response(recent_step, original_query, state.get("past_steps", []))
        else:
            final_response = "Research completed successfully."

        print("\n📌 FINAL OUTPUT")
        print(f"🟡 Query: {state.get('input', '')}")
        print(f"✅ Final response:\n{final_response}\n")

        return {"response": final_response}

    def format_enhanced_response(response: str, original_query: str, all_steps: List) -> str:
        response = response.strip()
        if len(response) < 100:
            return f"**Answer:** {response}"
        if any(indicator in response for indicator in ['##', '**', '###', '- ', '1.', '2.']):
            return response
        if len(all_steps) > 1:
            context_note = f"\n\n*Based on analysis of {len(all_steps)} research steps using the Swedish publications database.*"
            return f"**Research Results:**\n\n{response}{context_note}"
        return f"**Research Analysis:**\n\n{response}"

    def should_end(state: PlanExecuteState) -> Literal["agent", "__end__"]:
        if "response" in state and state["response"]:
            return "__end__"
        else:
            return "agent"

    # Build the workflow
    workflow = StateGraph(PlanExecuteState)
    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)
    workflow.add_node("complete", complete_step)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    workflow.add_conditional_edges("agent", should_continue_or_end, ["replan", "complete", END])
    workflow.add_edge("complete", END)
    workflow.add_edge("replan", "agent")

    return workflow

def compile_research_agent(es_client=None, index_name: str = "research-publications-static", recursion_limit: int = 50, session_id: str = None) -> Any:
    """Compile the research agent with LangSmith tracing."""
    workflow = create_research_workflow(es_client, index_name, session_id)
    app = workflow.compile()
    return app

def run_research_query(query: str, es_client=None, index_name: str = "research-publications-static", recursion_limit: int = 50, stream: bool = False, conversation_history: Optional[List[Dict]] = None, session_id: str = None) -> Dict[str, Any]:
    """Run research query with LangSmith tracing."""
    
    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    app = compile_research_agent(es_client, index_name, recursion_limit, session_id)
    initial_state = {
        "input": query,
        "conversation_history": conversation_history,
        "plan": [],
        "past_steps": [],
        "response": None,
        "session_id": session_id,
        "total_results": None,
        "current_step": 0,
        "error": None
    }
    
    # Add LangSmith metadata to config
    config = {
        "recursion_limit": recursion_limit,
        "metadata": {
            "query": query,
            "session_id": session_id,
            "index_name": index_name,
            "has_conversation_history": bool(conversation_history)
        },
        "tags": ["research_agent", "plan_execute", f"session_{session_id}"]
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

class ResearchAgent:
    """Research Agent with LangSmith tracing support."""
    
    def __init__(self, es_client=None, index_name: str = "research-publications-static", recursion_limit: int = 50):
        self.es_client = es_client
        self.index_name = index_name
        self.recursion_limit = recursion_limit
        self.app = None
        self.langsmith_client = setup_langsmith_tracing()

    def _compile_agent(self, session_id: str = None):
        """Compile agent with session-specific tracing."""
        self.app = compile_research_agent(
            self.es_client, 
            self.index_name, 
            self.recursion_limit,
            session_id
        )

    async def stream_query(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Any:
        """Stream query with LangSmith tracing."""
        
        # Generate session ID for this query
        session_id = str(uuid.uuid4())
        
        # Compile agent with session ID
        self._compile_agent(session_id)
        
        initial_state = {
            "input": query,
            "conversation_history": conversation_history,
            "plan": [],
            "past_steps": [],
            "response": None,
            "session_id": session_id,
            "total_results": None,
            "current_step": 0,
            "error": None
        }
        
        # Add LangSmith configuration
        config = {
            "recursion_limit": self.recursion_limit,
            "metadata": {
                "query": query,
                "session_id": session_id,
                "index_name": self.index_name,
                "has_conversation_history": bool(conversation_history),
                "stream_mode": True
            },
            "tags": ["research_agent", "streaming", f"session_{session_id}"]
        }
        
        async for event in self.app.astream(initial_state, config=config):
            yield event