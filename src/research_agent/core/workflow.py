"""
Main LangGraph workflow for the research publications agent.

Following LangChain's official plan-and-execute pattern from DEMO_plan-and-execute.ipynb
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

from .state import PlanExecuteState
from ..tools.elasticsearch_tools import (
    initialize_elasticsearch_tools, 
    create_elasticsearch_tools,
    get_tool_descriptions_for_planning,
    get_tool_descriptions_for_execution
)

# Import models from models.py
from .models import Plan, Response, Act

from .prompt_loader import (
    get_executor_prompt,
    get_planner_system_prompt, 
    get_replanner_prompt,
    get_task_format_template,
    get_context_aware_prompt,
    get_standard_planning_prompt
)

def create_research_workflow(
    es_client=None,
    index_name: str = "research-publications-static"
) -> StateGraph:
    """
    Create the main research agent workflow with dynamic tool descriptions.
    
    Args:
        es_client: Elasticsearch client instance
        index_name: Name of the publications index
        
    Returns:
        LangGraph workflow (not compiled)
    """
    # Initialize tools if ES client is provided
    if es_client:
        initialize_elasticsearch_tools(es_client, index_name)
    
    # Get tools for the executor
    tools = create_elasticsearch_tools()
    
    print(f"🔧 TOOLS DEBUG:")
    for tool in tools:
        print(f"  - Tool: {tool.name}")
        print(f"    Description: {tool.description[:200]}...")
        if hasattr(tool, 'args_schema') and tool.args_schema:
            print(f"    Args schema: {tool.args_schema}")
            # Try to show the actual parameters
            if hasattr(tool.args_schema, 'model_fields'):
                print(f"    Parameters: {list(tool.args_schema.model_fields.keys())}")
            elif hasattr(tool.args_schema, '__fields__'):
                print(f"    Parameters: {list(tool.args_schema.__fields__.keys())}")
        else:
            print(f"    Args schema: None")
        print()

    # Get dynamic tool descriptions
    planning_tool_descriptions = get_tool_descriptions_for_planning()
    execution_tool_descriptions = get_tool_descriptions_for_execution()
    
    # Load environment variables for LiteLLM
    load_dotenv()
    
    # Choose the LLM that will drive the agent
    llm = ChatLiteLLM(
        model="anthropic/claude-sonnet-4",
        api_key=os.getenv("LITELLM_API_KEY"),
        api_base=os.getenv("LITELLM_BASE_URL"),
        temperature=0
    )
    
    # Create the execution agent with external prompt
    executor_prompt = get_executor_prompt(execution_tool_descriptions)
    agent_executor = create_react_agent(llm, tools, prompt=executor_prompt)
    
    # Create planner with external prompt
    planner_system_prompt = get_planner_system_prompt(planning_tool_descriptions)
    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", planner_system_prompt),
        ("placeholder", "{messages}"),
    ])
    planner = planner_prompt | llm.with_structured_output(Plan)
    
    # Create replanner with external prompt
    replanner_llm = ChatLiteLLM(
        model="anthropic/claude-sonet-3.7",
        api_key=os.getenv("LITELLM_API_KEY"),
        api_base=os.getenv("LITELLM_BASE_URL"),
        temperature=0
    )
    
    replanner_template = get_replanner_prompt(planning_tool_descriptions)
    replanner_prompt = ChatPromptTemplate.from_template(replanner_template)
    replanner = replanner_prompt | replanner_llm.with_structured_output(Act)
    
    # Define workflow steps with enhanced logic
    def execute_step(state: PlanExecuteState):
        """Execute the current step using the research tools."""
        plan = state["plan"]
        past_steps = state.get("past_steps", [])
        
        print(f"🔧 EXECUTE_STEP DEBUG:")
        print(f"  - Plan length: {len(plan)}")
        print(f"  - Past steps length: {len(past_steps)}")
        print(f"  - Current plan: {plan}")
        print(f"  - Past steps: {[step[0] for step in past_steps]}")  # Just the task names
        
        if not plan:
            print("❌ EXECUTE_STEP: No plan available!")
            return {"past_steps": []}
        
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        
        print(f"🎯 EXECUTE_STEP: About to execute task: {task}")
        
        original_query = state.get("input", "")
        
        # Use external prompt template
        task_formatted = get_task_format_template(
            original_query=original_query,
            plan_str=plan_str,
            task=task
        )
        
        print(f"📝 EXECUTE_STEP: Sending formatted task to agent...")
        
        # Use sync invoke to avoid async issues with LiteLLM
        try:
            agent_response = agent_executor.invoke(
                {"messages": [("user", task_formatted)]}
            )
            
            response_content = agent_response["messages"][-1].content
            print(f"✅ EXECUTE_STEP: Agent response length: {len(response_content)} characters")
            print(f"📄 EXECUTE_STEP: Agent response preview: {response_content[:200]}...")
            
            return {
                "past_steps": [(task, response_content)],
            }
        except Exception as e:
            print(f"❌ EXECUTE_STEP: Exception during agent execution: {str(e)}")
            return {"past_steps": [(task, f"Error executing task: {str(e)}")]}
            
    def plan_step(state: PlanExecuteState):
        """Create the initial plan for the research query with conversation context."""
        try:
            query = state["input"]
            print(f"🔍 Enhanced Planner: Creating plan for query: {query}")
            
            # Get conversation history from state
            conversation_history = state.get("conversation_history", [])
            print(f"🔍 Enhanced Planner: Conversation history available: {len(conversation_history) if conversation_history else 0} messages")
            
            # Build context-aware messages
            messages = []
            
            # Add recent conversation context if available
            if conversation_history and len(conversation_history) > 0:
                # Build context from recent conversation
                context_lines = []
                for msg in conversation_history[-4:]:  # Last 2 exchanges
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    # Truncate very long messages but keep key info
                    truncated_content = content[:200] + "..." if len(content) > 200 else content
                    context_lines.append(f"- {role.title()}: {truncated_content}")
                
                context_summary = "\n".join(context_lines)
                
                # Use external context-aware prompt
                context_message = get_context_aware_prompt(context_summary, query)
                messages.append(("user", context_message))
                print(f"🔍 Enhanced Planner: Using context-aware prompt with {len(conversation_history)} previous messages")
            else:
                # Use external standard planning prompt
                standard_message = get_standard_planning_prompt(query)
                messages.append(("user", standard_message))
                print(f"🔍 Enhanced Planner: No conversation context available, using standard planning approach")
            
            # Use sync invoke to avoid async issues with LiteLLM
            plan = planner.invoke({"messages": messages})
            
            print(f"🔍 Enhanced Planner: Raw plan result: {plan}")
            print(f"🔍 Enhanced Planner: Plan type: {type(plan)}")
            
            if plan is None:
                print("❌ Enhanced Planner: Plan is None - LLM likely failed to generate structured output")
                fallback_plan = [f"Search for information about: {query}"]
                print(f"🔄 Enhanced Planner: Using fallback plan: {fallback_plan}")
                return {"plan": fallback_plan}
            
            if not hasattr(plan, 'steps'):
                print(f"❌ Enhanced Planner: Plan missing 'steps' attribute. Plan object: {plan}")
                fallback_plan = [f"Search for information about: {query}"]
                return {"plan": fallback_plan}
            
            print(f"✅ Enhanced Planner: Successfully created plan with {len(plan.steps)} steps: {plan.steps}")
            return {"plan": plan.steps}
            
        except Exception as e:
            print(f"❌ Enhanced Planner: Exception occurred: {str(e)}")
            print(f"❌ Enhanced Planner: Full traceback: {traceback.format_exc()}")
            fallback_plan = [f"Search for information about: {query}"]
            print(f"🔄 Enhanced Planner: Using fallback plan due to exception: {fallback_plan}")
            return {"plan": fallback_plan}
    
    def replan_step(state: PlanExecuteState):
        """Replan based on the results so far with enhanced decision making."""
        print(f"🔄 REPLAN_STEP DEBUG:")
        print(f"  - Current plan: {state.get('plan', [])}")
        print(f"  - Past steps count: {len(state.get('past_steps', []))}")
        
        # Use sync invoke to avoid async issues with LiteLLM
        try:
            output = replanner.invoke(state)
            print(f"🔄 REPLAN_STEP: Replanner output type: {type(output)}")
            print(f"🔄 REPLAN_STEP: Action type: {getattr(output, 'action_type', 'MISSING')}")
            
            if output.action_type == "response":
                print(f"🎯 REPLAN_STEP: Providing final response")
                return {"response": output.response}
            else:
                print(f"🔄 REPLAN_STEP: Continuing with new plan: {getattr(output, 'steps', 'MISSING')}")
                return {"plan": output.steps}
        except Exception as e:
            print(f"❌ REPLAN_STEP: Exception: {str(e)}")
            return {"response": f"Error during replanning: {str(e)}"}
    
    def should_continue_or_end(state: PlanExecuteState) -> Literal["replan", "complete", "__end__"]:
        """Enhanced logic for determining workflow continuation."""
        plan = state.get("plan", [])
        past_steps = state.get("past_steps", [])
        
        print(f"🎯 SHOULD_CONTINUE DEBUG:")
        print(f"  - Plan length: {len(plan)}")
        print(f"  - Past steps length: {len(past_steps)}")
        print(f"  - Plan steps: {plan}")
        
        if not plan or not past_steps:
            print(f"🎯 SHOULD_CONTINUE: Missing plan or past_steps → replan")
            return "replan"
        
        # Check if all plan steps are completed
        if len(past_steps) >= len(plan):
            print(f"🎯 SHOULD_CONTINUE: All steps completed ({len(past_steps)}/{len(plan)}) → complete")
            return "complete"
        
        # NOT all steps completed - should continue
        print(f"🎯 SHOULD_CONTINUE: {len(past_steps)}/{len(plan)} steps completed → replan to continue")
        return "replan"
    
    def complete_step(state: PlanExecuteState):
        """Complete the workflow with enhanced response formatting."""
        if state.get("past_steps"):
            recent_step = state["past_steps"][-1][1]
            original_query = state.get("input", "")
            
            # Enhanced response formatting
            formatted_response = format_enhanced_response(recent_step, original_query, state.get("past_steps", []))
            return {"response": formatted_response}
        return {"response": "Research completed successfully."}
    
    def format_enhanced_response(response: str, original_query: str, all_steps: List) -> str:
        """Enhanced response formatting with better structure and context."""
        # Clean up the response
        response = response.strip()
        
        # If response is very short, assume it's a direct answer
        if len(response) < 100:
            return f"**Answer:** {response}"
        
        # If response already has good structure, preserve it
        if any(indicator in response for indicator in ['##', '**', '###', '- ', '1.', '2.']):
            return response
        
        # For comprehensive responses, add context about completeness
        if len(all_steps) > 1:
            context_note = f"\n\n*Based on analysis of {len(all_steps)} research steps using the Swedish publications database.*"
            return f"**Research Results:**\n\n{response}{context_note}"
        
        # Default formatting for single-step comprehensive responses
        return f"**Research Analysis:**\n\n{response}"
    
    def should_end(state: PlanExecuteState) -> Literal["agent", "__end__"]:
        """Determine if we should end or continue after replan."""
        if "response" in state and state["response"]:
            return "__end__"
        else:
            return "agent"
    
    # Create the state graph
    workflow = StateGraph(PlanExecuteState)
    
    # Add nodes
    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)
    workflow.add_node("complete", complete_step)
    
    # Add edges with enhanced routing
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    
    # Add conditional edge from agent
    workflow.add_conditional_edges(
        "agent",
        should_continue_or_end,
        ["replan", "complete", END]
    )
    
    # Add edge from complete to END
    workflow.add_edge("complete", END)
    
    # Add conditional edge for ending after replan
    workflow.add_conditional_edges(
        "replan",
        should_end,
        ["agent", END]
    )
    
    return workflow


# Update other functions to use the enhanced workflow
def compile_research_agent(
    es_client=None,
    index_name: str = "research-publications-static",
    recursion_limit: int = 50
) -> Any:
    """
    Compile the enhanced research agent workflow into a runnable.
    """
    workflow = create_research_workflow(es_client, index_name)
    app = workflow.compile()
    return app


def run_research_query(
    query: str,
    es_client=None,
    index_name: str = "research-publications-static",
    recursion_limit: int = 50,
    stream: bool = False,
    conversation_history: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Run a research query through the enhanced workflow.
    """
    # Compile the agent
    app = compile_research_agent(es_client, index_name, recursion_limit)
    
    # Create initial state with conversation history
    initial_state = {
        "input": query,
        "conversation_history": conversation_history,  # Include context
        "plan": [],
        "past_steps": [],
        "response": None,
        "session_id": None,
        "total_results": None,
        "current_step": 0,
        "error": None
    }
    
    # Configuration
    config = {"recursion_limit": recursion_limit}
    
    if stream:
        # Stream the execution
        final_state = None
        for event in app.stream(initial_state, config=config):
            for k, v in event.items():
                if k != "__end__":
                    print(f"Step: {k}")
                    print(f"Result: {v}")
                    print("-" * 50)
                    final_state = v
        
        return final_state
    else:
        # Run synchronously
        result = app.invoke(initial_state, config=config)
        return result


class ResearchAgent:
    """
    Enhanced research agent class with dynamic tool descriptions.
    """
    
    def __init__(
        self,
        es_client=None,
        index_name: str = "research-publications-static",
        recursion_limit: int = 50
    ):
        """Initialize the enhanced research agent."""
        self.es_client = es_client
        self.index_name = index_name
        self.recursion_limit = recursion_limit
        self.app = None
        
        # Compile the agent
        self._compile_agent()
    
    def _compile_agent(self):
        """Compile the enhanced LangGraph workflow."""
        self.app = compile_research_agent(
            self.es_client,
            self.index_name,
            self.recursion_limit
        )
    
    async def stream_query(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Any:
        """
        Stream a research query execution with enhanced context handling.
        """
        initial_state = {
            "input": query,
            "conversation_history": conversation_history,
            "plan": [],
            "past_steps": [],
            "response": None,
            "session_id": None,
            "total_results": None,
            "current_step": 0,
            "error": None
        }
        
        config = {"recursion_limit": self.recursion_limit}
        
        async for event in self.app.astream(initial_state, config=config):
            yield event