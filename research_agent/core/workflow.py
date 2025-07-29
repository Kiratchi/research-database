"""
Updated workflow - Clean integration with new tools system
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
import json
import asyncio

# LangSmith imports (unchanged)
from langsmith import Client
from langchain_core.tracers import LangChainTracer
from langchain_core.callbacks import BaseCallbackHandler
import uuid

# Your existing imports (unchanged)
from .state import PlanExecuteState
from .models import Plan, Response, Act
from .prompt_loader import (
    get_executor_prompt,
    get_planner_system_prompt, 
    get_replanner_prompt,
    get_task_format_template,
    get_context_aware_prompt,
    get_standard_planning_prompt
)

# NEW TOOLS IMPORT - This is the main change!
from ..tools.search.registry import get_all_tools



# Your existing ResearchAgentTracer (unchanged)
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
            chain_name = 'unknown'
            if serialized and isinstance(serialized, dict):
                chain_name = serialized.get('name', 'unknown')
            elif hasattr(serialized, 'get'):
                chain_name = serialized.get('name', 'unknown')
        except Exception as e:
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
            pass


def setup_langsmith_tracing():
    """Setup LangSmith tracing with environment variables."""
    load_dotenv()
    
    langsmith_config = {
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "true"),
        "LANGCHAIN_ENDPOINT": os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY"),
        "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT", "research-publications-agent")
    }
    
    for key, value in langsmith_config.items():
        if value:
            os.environ[key] = value
    
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("⚠️ WARNING: LANGCHAIN_API_KEY not set. LangSmith tracing disabled.")
        return None
    
    try:
        client = Client(
            api_url=os.getenv("LANGCHAIN_ENDPOINT"),
            api_key=os.getenv("LANGCHAIN_API_KEY")
        )
        client.list_runs(limit=1)
        print("✅ LangSmith tracing initialized")
        return client
    except Exception as e:
        print(f"❌ Failed to initialize LangSmith: {e}")
        return None


def load_prompt_from_file(filename: str, **kwargs) -> str:
    """Load prompt from text file and format with kwargs."""
    try:
        # Try to load from prompts directory
        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', filename)
        
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                template = f.read()
            return template.format(**kwargs)
        else:
            print(f"⚠️ Prompt file not found: {filename}")
            return f"Prompt file {filename} not found. Using fallback."
    except Exception as e:
        print(f"❌ Error loading prompt {filename}: {e}")
        return f"Error loading prompt {filename}."


def get_enhanced_tool_descriptions_for_planning() -> str:
    """
    Generate enhanced tool descriptions using the new tools system.
    This replaces your old get_tool_descriptions_for_planning function.
    """
    # Get tools without ES client (descriptions don't need actual connection)
    try:
        sample_tools = get_all_tools()
        
        descriptions = []
        for tool in sample_tools:
            # Get the full tool description and input schema
            tool_desc = f"""**{tool.name}**: {tool.description}

**Input Parameters:**"""
            
            # Extract parameter descriptions from the tool's args_schema
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema
                
                # Handle Pydantic model fields
                if hasattr(schema, 'model_fields'):
                    # Pydantic v2
                    for field_name, field_info in schema.model_fields.items():
                        field_desc = getattr(field_info, 'description', 'No description')
                        default_val = getattr(field_info, 'default', None)
                        if default_val is not None and default_val != ...:
                            tool_desc += f"\n- {field_name}: {field_desc} (default: {default_val})"
                        else:
                            tool_desc += f"\n- {field_name}: {field_desc}"
                elif hasattr(schema, '__fields__'):
                    # Pydantic v1 fallback
                    for field_name, field_info in schema.__fields__.items():
                        field_desc = getattr(field_info.field_info, 'description', 'No description')
                        tool_desc += f"\n- {field_name}: {field_desc}"
            
            # Add usage guidance based on tool name
            if tool.name == "unified_search":
                tool_desc += """

**Key Capabilities:**
- Advanced filtering by authors, years, publication types, sources, keywords, language
- Multiple sorting options: relevance, year_desc, year_asc, title_asc, citations_desc
- Field-specific searching: title, abstract, authors, keywords, source, or all
- Pagination support with has_more indicators
- Error handling with helpful suggestions

**Best Practices:**
- Use filters to narrow results for specific research questions
- Use sort_by="year_desc" for recent publications
- Use field_selection="minimal" for quick searches, "full" for comprehensive analysis
- Combine multiple filters for precise searches (e.g., specific author + year range + publication type)"""
            
            descriptions.append(tool_desc)
        
        return "\n\n" + "="*80 + "\n\n".join(descriptions)
        
    except Exception as e:
        print(f"⚠️ Could not load tool descriptions: {e}")
        return "**unified_search**: Comprehensive search tool for research publications with advanced filtering and sorting capabilities."


def get_enhanced_tool_descriptions_for_execution() -> str:
    """
    Generate concise tool descriptions for executor prompts.
    This replaces your old get_tool_descriptions_for_execution function.
    """
    try:
        sample_tools = get_all_tools()
        
        descriptions = []
        for tool in sample_tools:
            tool_desc = f"- **{tool.name}**: {tool.description}"
            descriptions.append(tool_desc)
        
        return "\n\n".join(descriptions)
        
    except Exception as e:
        print(f"⚠️ Could not load execution descriptions: {e}")
        return "- **unified_search**: Comprehensive search tool for research publications"


def create_research_workflow(es_client=None, index_name: str = "research-publications-static", session_id: str = None) -> StateGraph:
    """Create the research workflow with new tools integration."""
    
    # Setup LangSmith tracing
    langsmith_client = setup_langsmith_tracing()
    
    # NEW TOOLS INTEGRATION - Simple and clean!
    if es_client:
        # Get all tools with your ES client
        tools = get_all_tools(es_client=es_client, index_name=index_name)
        print(f"✅ Initialized {len(tools)} research tools with ES client")
    else:
        # For testing/development - uses default ES client from settings
        tools = get_all_tools()
        print(f"✅ Initialized {len(tools)} research tools with default settings")
    
    # Get enhanced tool descriptions
    planning_tool_descriptions = get_enhanced_tool_descriptions_for_planning()
    execution_tool_descriptions = get_enhanced_tool_descriptions_for_execution()

    # Create LLM with tracing callbacks
    callbacks = []
    if langsmith_client:
        callbacks.append(ResearchAgentTracer(session_id))

    # FIXED: Use provider prefix for LangChain LiteLLM integration
    llm = ChatLiteLLM(
        model="anthropic/claude-sonnet-4",
        api_key=os.getenv("LITELLM_API_KEY"),
        api_base=os.getenv("LITELLM_BASE_URL"),
        temperature=0,
        callbacks=callbacks,
        metadata={"component": "main_llm", "session_id": session_id}
    )

    # Create REACT agent with new tools
    executor_prompt = get_executor_prompt(execution_tool_descriptions)
    agent_executor = create_react_agent(llm, tools, prompt=executor_prompt)

    # Create planner using prompt from file
    def create_planner():
        def plan_with_file_prompt(inputs):
            # Load batched planning prompt from file
            prompt_text = load_prompt_from_file(
                'batched_planning_prompt.txt',
                tool_descriptions=planning_tool_descriptions,
                query=inputs.get('query', ''),
                conversation_context=inputs.get('conversation_context', '')
            )
            
            # Use the prompt with LLM
            response = llm.invoke([("user", prompt_text)])
            
            # Parse response into Plan format
            try:
                # Simple parsing - look for numbered steps or bullet points
                content = response.content
                steps = []
                
                # Try to extract steps from response
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                        # Clean up the step text
                        step = line.lstrip('0123456789.-• ').strip()
                        if step:
                            steps.append(step)
                
                # Fallback if no steps found
                if not steps:
                    steps = [f"Research comprehensive information about: {inputs.get('query', '')}"]
                
                return Plan(steps=steps)
            except Exception as e:
                print(f"⚠️ Error parsing plan: {e}")
                return Plan(steps=[f"Research information about: {inputs.get('query', '')}"])
        
        return plan_with_file_prompt

    planner = create_planner()

    # FIXED: Use provider prefix for LangChain LiteLLM integration
    replanner_llm = ChatLiteLLM(
        model="anthropic/claude-haiku-3.5",
        api_key=os.getenv("LITELLM_API_KEY"),
        api_base=os.getenv("LITELLM_BASE_URL"),
        temperature=0,
        callbacks=callbacks,
        metadata={"component": "replanner_llm", "session_id": session_id}
    )

    replanner_template = get_replanner_prompt(planning_tool_descriptions)
    replanner_prompt = ChatPromptTemplate.from_template(replanner_template)
    replanner = replanner_prompt | replanner_llm.with_structured_output(Act)

    def execute_step(state: PlanExecuteState):
        """Execute step with memory-aware context from file prompt."""
        plan = state["plan"]
        past_steps = state.get("past_steps", [])
        conversation_history = state.get("conversation_history", [])
        
        if not plan:
            return {"past_steps": []}

        task = plan[0]  # Always take first task
        original_query = state.get("input", "")
        
        # Build conversation context
        conversation_context = ""
        if conversation_history:
            recent_context = []
            for msg in conversation_history[-4:]:  # Last 2 Q&A pairs
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:150]
                recent_context.append(f"{role.title()}: {content}...")
            conversation_context = "\n".join(recent_context)
        else:
            conversation_context = "No previous conversation."

        # Build previous steps context
        previous_steps = ""
        if past_steps:
            steps_context = []
            for i, (step_task, step_result) in enumerate(past_steps, 1):
                result_preview = step_result[:150] + "..." if len(step_result) > 150 else step_result
                steps_context.append(f"{i}. {step_task}\n   Result: {result_preview}")
            previous_steps = "\n\n".join(steps_context)
        else:
            previous_steps = "No previous steps."

        # Load execution prompt from file
        execution_prompt = load_prompt_from_file(
            'memory_aware_execution_prompt.txt',
            task=task,
            original_query=original_query,
            conversation_context=conversation_context,
            previous_steps=previous_steps
        )

        try:
            config = {
                "metadata": {
                    "step": "execute",
                    "task": task,
                    "session_id": session_id,
                    "step_number": len(past_steps) + 1
                },
                "callbacks": callbacks
            }
            
            agent_response = agent_executor.invoke(
                {"messages": [("user", execution_prompt)]},
                config=config
            )
            
            response_content = agent_response["messages"][-1].content
            return {"past_steps": [(task, response_content)]}
            
        except Exception as e:
            error_response = f"Error executing task: {str(e)}"
            print(f"❌ Error in execute_step: {e}")
            return {"past_steps": [(task, error_response)]}

    def plan_step(state: PlanExecuteState):
        """Planning step using prompt from file."""
        try:
            query = state["input"]
            conversation_history = state.get("conversation_history", [])
            
            # Build conversation context
            conversation_context = ""
            if conversation_history:
                context_lines = []
                for msg in conversation_history[-6:]:  # Last 3 Q&A pairs
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    truncated_content = content[:200] + "..." if len(content) > 200 else content
                    context_lines.append(f"- {role.title()}: {truncated_content}")
                conversation_context = "\n".join(context_lines)
            else:
                conversation_context = "No previous conversation history."

            # Create planning input
            planning_input = {
                "query": query,
                "conversation_context": conversation_context
            }

            config = {
                "metadata": {
                    "step": "planning",
                    "query": query,
                    "has_context": len(conversation_history) > 0,
                    "session_id": session_id
                },
                "callbacks": callbacks
            }

            plan = planner(planning_input)

            if plan and hasattr(plan, 'steps') and plan.steps:
                print(f"📋 Created plan with {len(plan.steps)} steps: {plan.steps}")
                return {"plan": plan.steps}
            else:
                fallback_plan = [f"Research comprehensive information about: {query}"]
                return {"plan": fallback_plan}
            
        except Exception as e:
            print(f"❌ Error in planning: {e}")
            fallback_plan = [f"Research information about: {query}"]
            return {"plan": fallback_plan}

    def replan_step(state: PlanExecuteState):
        """Replanning step with better error handling."""
        try:
            config = {
                "metadata": {
                    "step": "replanning",
                    "past_steps_count": len(state.get("past_steps", [])),
                    "session_id": session_id
                },
                "callbacks": callbacks
            }
            
            output = replanner.invoke(state, config=config)
            if output.action_type == "response":
                return {"final_response": output.response}
            else:
                return {"plan": output.steps}
                
        except Exception as e:
            print(f"❌ Error in replanning: {e}")
            # If replanning fails, try to create a reasonable response
            past_steps = state.get("past_steps", [])
            if past_steps:
                latest_result = past_steps[-1][1]
                return {"final_response": latest_result}
            else:
                return {"final_response": f"Error during replanning: {str(e)}"}

    def should_continue_or_end(state: PlanExecuteState) -> Literal["replan", "complete", "__end__"]:
        """Decide whether to continue, replan, or end - with replan limit."""
        plan = state.get("plan", [])
        past_steps = state.get("past_steps", [])

        # Complete if we have final response
        if state.get("final_response"):
            print("✅ COMPLETE: Final response available")
            return "complete"
        
        # FIXED: Limit replanning cycles to prevent timeout
        if len(past_steps) >= 3:  # Max 3 steps total
            print("✅ COMPLETE: Maximum steps reached (preventing timeout)")
            return "complete"
        
        # Complete if we've done all planned steps
        if len(past_steps) >= len(plan):
            print("✅ COMPLETE: All plan steps executed")
            return "complete"
        
        # Replan for more steps (but limited by above)
        print("🔄 REPLAN: Continuing with replanning")
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

        print(f"\n📌 FINAL OUTPUT")
        print(f"🟡 Query: {state.get('input', '')}")
        print(f"✅ Response length: {len(final_response)} characters")

        return {"response": final_response}

    def format_enhanced_response(response: str, original_query: str, all_steps: List) -> str:
        """Simple response formatting."""
        if not response:
            return "No response generated."
            
        response = response.strip()
        
        if len(response) < 100:
            return f"**Answer:** {response}"
            
        if any(indicator in response for indicator in ['##', '**', '###', '- ', '1.', '2.']):
            return response
            
        if len(all_steps) > 1:
            context_note = f"\n\n*Based on {len(all_steps)} research steps using the publications database.*"
            return f"**Research Results:**\n\n{response}{context_note}"
            
        return f"**Research Analysis:**\n\n{response}"

    # Build the simplified workflow
    workflow = StateGraph(PlanExecuteState)
    
    # Add nodes
    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)
    workflow.add_node("complete", complete_step)

    # Simple edges - always replan
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    workflow.add_conditional_edges("agent", should_continue_or_end, ["replan", "complete", END])
    workflow.add_edge("complete", END)
    workflow.add_edge("replan", "agent")

    return workflow


# Rest of your helper functions remain the same
def compile_research_agent(es_client=None, index_name: str = "research-publications-static", recursion_limit: int = 50, session_id: str = None) -> Any:
    """Compile the research agent."""
    workflow = create_research_workflow(es_client, index_name, session_id)
    app = workflow.compile()
    return app


def run_research_query(query: str, es_client=None, index_name: str = "research-publications-static", recursion_limit: int = 50, stream: bool = False, conversation_history: Optional[List[Dict]] = None, session_id: str = None) -> Dict[str, Any]:
    """Run research query."""
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    app = compile_research_agent(es_client, index_name, recursion_limit, session_id)
    initial_state = {
        "input": query,
        "conversation_history": conversation_history or [],
        "plan": [],
        "past_steps": [],
        "response": None,
        "session_id": session_id,
        "total_results": None,
        "current_step": 0,
        "error": None
    }
    
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


# Updated ResearchAgent class
class ResearchAgent:
    """Research Agent with new tools integration."""
    
    def __init__(self, es_client=None, index_name: str = "research-publications-static", recursion_limit: int = 50):
        self.es_client = es_client
        self.index_name = index_name
        self.recursion_limit = recursion_limit
        self.app = None
        self.langsmith_client = setup_langsmith_tracing()

    def _compile_agent(self, session_id: str = None):
        """Compile agent."""
        self.app = compile_research_agent(
            self.es_client, 
            self.index_name, 
            self.recursion_limit,
            session_id
        )

    async def stream_query(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Any:
        """Standard LangGraph streaming."""
        
        session_id = str(uuid.uuid4())
        self._compile_agent(session_id)
        
        initial_state = {
            "input": query,
            "conversation_history": conversation_history or [],
            "plan": [],
            "past_steps": [],
            "response": None,
            "session_id": session_id,
            "total_results": None,
            "current_step": 0,
            "error": None
        }
        
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
        
        # Standard LangGraph streaming
        async for event in self.app.astream(initial_state, config=config):
            yield event