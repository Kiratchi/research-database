"""
Main LangGraph workflow for the research publications agent.

Following LangChain's official plan-and-execute pattern from DEMO_plan-and-execute.ipynb
"""

from typing import Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Union, List
import os

from .state import PlanExecuteState
from ..tools.elasticsearch_tools import initialize_elasticsearch_tools, create_elasticsearch_tools


# Pydantic models following DEMO patterns
class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class Response(BaseModel):
    """Response to user."""
    response: str


class Act(BaseModel):
    """Action to perform."""
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


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
    
    # Choose the LLM that will drive the agent
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
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
- get_database_summary: Get database summary statistics"""
    )
    
    replanner = replanner_prompt | llm.with_structured_output(Act)
    
    # Define workflow steps following DEMO patterns
    async def execute_step(state: PlanExecuteState):
        """Execute the current step using the research tools."""
        plan = state["plan"]
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_formatted = f"""For the following plan:
{plan_str}

You are tasked with executing step 1: {task}

Use the available research publication tools to complete this step."""
        
        agent_response = await agent_executor.ainvoke(
            {"messages": [("user", task_formatted)]}
        )
        
        return {
            "past_steps": [(task, agent_response["messages"][-1].content)],
        }
    
    async def plan_step(state: PlanExecuteState):
        """Create the initial plan for the research query."""
        plan = await planner.ainvoke({"messages": [("user", state["input"])]})
        return {"plan": plan.steps}
    
    async def replan_step(state: PlanExecuteState):
        """Replan based on the results so far."""
        output = await replanner.ainvoke(state)
        if isinstance(output.action, Response):
            return {"response": output.action.response}
        else:
            return {"plan": output.action.steps}
    
    def should_end(state: PlanExecuteState) -> Literal["agent", "__end__"]:
        """Determine if we should end or continue."""
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
    
    # Add edges exactly like DEMO
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    workflow.add_edge("agent", "replan")
    
    # Add conditional edge for ending
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


async def run_research_query(
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
        async for event in app.astream(initial_state, config=config):
            for k, v in event.items():
                if k != "__end__":
                    print(f"Step: {k}")
                    print(f"Result: {v}")
                    print("-" * 50)
                    final_state = v
        
        return final_state
    else:
        # Run synchronously
        result = await app.ainvoke(initial_state, config=config)
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