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

from .state import PlanExecuteState
from ..tools.elasticsearch_tools import initialize_elasticsearch_tools, create_elasticsearch_tools


# Import models from models.py
from .models import Plan, Response, Act


def create_research_workflow(
    es_client=None,
    index_name: str = "research-publications-static"
) -> StateGraph:
    """
    Create the main research agent workflow using LangGraph.
    
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
    
    # Load environment variables for LiteLLM
    load_dotenv()
    
    # Choose the LLM that will drive the agent - using LiteLLM proxy
    llm = ChatLiteLLM(
        model="anthropic/claude-sonet-4",  # Use Claude Sonet 4 for complex reasoning
        api_key=os.getenv("LITELLM_API_KEY"),
        api_base=os.getenv("LITELLM_BASE_URL"),
        temperature=0
    )
    
    # Create the execution agent using langgraph prebuilt
    agent_executor = create_react_agent(
        llm, 
        tools, 
        prompt="You are a helpful research assistant. Use the available tools to search and analyze research publications."
    )
    
    # Create planner following DEMO patterns
    planner_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """For the given objective, come up with a simple step by step plan for searching research publications. \
This plan should involve individual tasks using the available tools, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Available tools:
- search_publications: Search research publications using full-text search
- search_by_author: Search publications by author name
- get_field_statistics: Get statistics for specific fields
- get_publication_details: Get detailed information about a publication
- get_database_summary: Get database summary statistics

Focus on using these tools effectively for research publication queries.""",
        ),
        ("placeholder", "{messages}"),
    ])
    
    planner = planner_prompt | llm.with_structured_output(Plan)
    
    # Create a separate LLM instance for replanner (could use different model for efficiency)
    replanner_llm = ChatLiteLLM(
        model="anthropic/claude-haiku-3.5",  # Use faster model for replanning
        api_key=os.getenv("LITELLM_API_KEY"),
        api_base=os.getenv("LITELLM_BASE_URL"),
        temperature=0
    )
    
    # Create replanner following DEMO patterns
    replanner_prompt = ChatPromptTemplate.from_template(
        """For the given objective, come up with a simple step by step plan for searching research publications. \
This plan should involve individual tasks using the available tools, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.

Available tools:
- search_publications: Search research publications using full-text search
- search_by_author: Search publications by author name
- get_field_statistics: Get statistics for specific fields
- get_publication_details: Get detailed information about a publication
- get_database_summary: Get database summary statistics

IMPORTANT: You must respond with either:
1. action_type: "response" with a final response to the user
2. action_type: "plan" with a list of remaining steps

Do not mix response and plan fields. Choose one action type only.

RESPONSE FORMATTING GUIDELINES:
- Provide direct, concise answers to the user's specific question
- Include the most relevant information prominently
- Use clear formatting (bullet points, numbering, headers) when appropriate
- Avoid unnecessary technical details about search processes
- Focus on actionable information the user can use"""
    )
    
    replanner = replanner_prompt | replanner_llm.with_structured_output(Act)
    
    # Define workflow steps following DEMO patterns
    def execute_step(state: PlanExecuteState):
        """Execute the current step using the research tools."""
        plan = state["plan"]
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        
        # Check if this is a simple query that can be completed in one step
        original_query = state.get("input", "")
        
        # General prompting for all query types
        task_formatted = f"""Original query: {original_query}

Current plan:
{plan_str}

You are executing: {task}

Use the available research publication tools to complete this step. Provide clear, specific answers that directly address the user's question. If you need additional information to fully answer the query, continue with the next step in the plan."""
        
        # Use sync invoke to avoid async issues with LiteLLM
        agent_response = agent_executor.invoke(
            {"messages": [("user", task_formatted)]}
        )
        
        response_content = agent_response["messages"][-1].content
        
        # Check if this response seems to fully answer the original query
        if ("total publications" in response_content.lower() or 
            "publications found" in response_content.lower() or
            ("has published" in response_content.lower() and "publications" in response_content.lower())):
            # This might be a complete answer - let the routing decide
            pass
        
        return {
            "past_steps": [(task, response_content)],
        }
    
    def plan_step(state: PlanExecuteState):
        """Create the initial plan for the research query."""
        # Use sync invoke to avoid async issues with LiteLLM
        plan = planner.invoke({"messages": [("user", state["input"])]})
        return {"plan": plan.steps}
    
    def replan_step(state: PlanExecuteState):
        """Replan based on the results so far."""
        # Use sync invoke to avoid async issues with LiteLLM
        output = replanner.invoke(state)
        if output.action_type == "response":
            return {"response": output.response}
        else:
            return {"plan": output.steps}
    
    def should_continue_or_end(state: PlanExecuteState) -> Literal["replan", "complete", "__end__"]:
        """Determine if agent should continue with replan or end."""
        # Check if we have a plan and past steps
        plan = state.get("plan", [])
        past_steps = state.get("past_steps", [])
        
        if not plan or not past_steps:
            return "replan"
        
        # Key insight: Only complete if ALL plan steps are done OR if the user's specific question is answered
        original_query = state.get("input", "").lower()
        recent_step = past_steps[-1][1] if past_steps else ""
        
        # Check if all plan steps are completed
        if len(past_steps) >= len(plan):
            return "complete"
        
        # General completion logic: check if the response seems to directly answer the question
        # Look for indicators that this is a complete response rather than intermediate results
        
        # If the response is asking the user for input, it's not complete
        if any(phrase in recent_step.lower() for phrase in ["would you like", "do you want", "shall i", "should i"]):
            return "replan"
        
        # If the response seems to be providing a direct answer to the question
        answer_indicators = ["the answer is", "based on", "according to", "results show", "found that"]
        if any(indicator in recent_step.lower() for indicator in answer_indicators):
            return "complete"
        
        # If the response is quite substantial and contains specific information
        if len(recent_step) > 200 and len(past_steps) > 0:
            # Check if this seems like a final answer rather than a search summary
            if not any(phrase in recent_step.lower() for phrase in ["now we can", "next step", "proceed to"]):
                return "complete"
        
        # Default: continue with replan
        return "replan"
    
    def complete_step(state: PlanExecuteState):
        """Complete the workflow by setting the response from the last step."""
        if state.get("past_steps"):
            recent_step = state["past_steps"][-1][1]
            original_query = state.get("input", "")
            
            # Format response based on query type
            formatted_response = format_final_response(recent_step, original_query)
            return {"response": formatted_response}
        return {"response": "Task completed."}
    
    def format_final_response(response: str, original_query: str) -> str:
        """Format the final response for better user experience."""
        # General formatting improvements without query-specific logic
        
        # Clean up the response
        response = response.strip()
        
        # If response is very short, assume it's a direct answer
        if len(response) < 50:
            return f"**Answer:** {response}"
        
        # If response contains structured information, preserve it
        if any(indicator in response for indicator in ['**', '###', '- ', '1.', '2.']):
            return response
        
        # For longer responses, add a subtle header
        return f"**Result:**\n\n{response}"
    
    
    def should_end(state: PlanExecuteState) -> Literal["agent", "__end__"]:
        """Determine if we should end or continue after replan."""
        if "response" in state and state["response"]:
            return "__end__"
        else:
            return "agent"
    
    # Create the state graph exactly like DEMO
    workflow = StateGraph(PlanExecuteState)
    
    # Add nodes
    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)
    workflow.add_node("complete", complete_step)
    
    # Add edges following DEMO pattern with proper conditional routing
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    
    # Add conditional edge from agent - let agent decide next action
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


def compile_research_agent(
    es_client=None,
    index_name: str = "research-publications-static",
    recursion_limit: int = 50
) -> Any:
    """
    Compile the research agent workflow into a runnable.
    
    Args:
        es_client: Elasticsearch client instance
        index_name: Name of the publications index
        recursion_limit: Maximum number of steps to execute
        
    Returns:
        Compiled LangGraph agent
    """
    workflow = create_research_workflow(es_client, index_name)
    
    # Compile the workflow
    app = workflow.compile()
    
    return app


def run_research_query(
    query: str,
    es_client=None,
    index_name: str = "research-publications-static",
    recursion_limit: int = 50,
    stream: bool = False
) -> Dict[str, Any]:
    """
    Run a research query through the complete workflow.
    
    Args:
        query: Natural language query about research publications
        es_client: Elasticsearch client instance
        index_name: Name of the publications index
        recursion_limit: Maximum number of steps to execute
        stream: Whether to stream intermediate results
        
    Returns:
        Final result from the workflow
    """
    # Compile the agent
    app = compile_research_agent(es_client, index_name, recursion_limit)
    
    # Create initial state following DEMO patterns
    initial_state = {
        "input": query,
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
        # Run synchronously - avoid async issues with LiteLLM
        result = app.invoke(initial_state, config=config)
        return result


class ResearchAgent:
    """
    Main research agent class that wraps the LangGraph workflow.
    
    Following the exact pattern from DEMO_plan-and-execute.ipynb
    """
    
    def __init__(
        self,
        es_client=None,
        index_name: str = "research-publications-static",
        recursion_limit: int = 50
    ):
        """
        Initialize the research agent.
        
        Args:
            es_client: Elasticsearch client instance
            index_name: Name of the publications index
            recursion_limit: Maximum number of steps to execute
        """
        self.es_client = es_client
        self.index_name = index_name
        self.recursion_limit = recursion_limit
        self.app = None
        
        # Compile the agent
        self._compile_agent()
    
    def _compile_agent(self):
        """Compile the LangGraph workflow."""
        self.app = compile_research_agent(
            self.es_client,
            self.index_name,
            self.recursion_limit
        )
    
    async def query(self, query: str, stream: bool = False) -> Dict[str, Any]:
        """
        Execute a research query.
        
        Args:
            query: Natural language query about research publications
            stream: Whether to stream intermediate results
            
        Returns:
            Final result from the workflow
        """
        return await run_research_query(
            query,
            self.es_client,
            self.index_name,
            self.recursion_limit,
            stream
        )
    
    async def stream_query(self, query: str) -> Any:
        """
        Stream a research query execution.
        
        Args:
            query: Natural language query about research publications
            
        Yields:
            Intermediate results from the workflow
        """
        initial_state = {
            "input": query,
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
    
    def get_graph_visualization(self) -> bytes:
        """
        Get a visualization of the workflow graph.
        
        Returns:
            PNG bytes of the graph visualization
        """
        return self.app.get_graph(xray=True).draw_mermaid_png()


if __name__ == "__main__":
    # Example usage matching DEMO patterns
    import asyncio
    
    async def test_workflow():
        """Test the research agent workflow."""
        # Create a research agent
        agent = ResearchAgent()
        
        # Test query
        query = "How many papers has Christian Fager published?"
        
        print(f"Query: {query}")
        print("=" * 50)
        
        # This would work with proper ES client and OpenAI API key
        # result = await agent.query(query, stream=True)
        # print(f"Final result: {result}")
        
        print("Research agent workflow created successfully!")
        print("Add ES client and OpenAI API key to test with real data.")
    
    # Uncomment to test
    # asyncio.run(test_workflow())