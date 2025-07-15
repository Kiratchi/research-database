"""
LangGraph-based executor for research publications agent.

Following LangChain's official plan-and-execute pattern.
"""

from typing import Dict, Any, List
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from ..core.state import PlanExecuteState
from ..tools.elasticsearch_tools import create_elasticsearch_tools


def create_executor(
    tools: List[Tool] = None,
    model_name: str = "gpt-4-turbo-preview",
    temperature: float = 0.1
) -> Any:
    """
    Create an executor agent for executing individual plan steps.
    
    Args:
        tools: List of tools available to the executor
        model_name: OpenAI model to use
        temperature: Temperature for generation
        
    Returns:
        LangGraph agent executor
    """
    if tools is None:
        tools = create_elasticsearch_tools()
    
    # Create the LLM
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    
    # Create the prompt for the executor
    prompt = """You are a research publications assistant. Your job is to execute individual steps of a research plan.

You have access to tools for searching and analyzing research publications:
- search_publications: Search by topic, keywords, or filters
- search_by_author: Search for publications by a specific author
- get_more_results: Get additional pages from previous searches
- get_field_statistics: Analyze field distributions
- get_statistics_summary: Get overall database statistics

For each step you execute:
1. Understand what information is needed
2. Choose the appropriate tool(s)
3. Execute the tool with correct parameters
4. Return the results clearly and concisely

Focus on providing accurate, specific information about research publications.
Always include session IDs when they're returned for potential follow-up queries."""
    
    # Create the ReAct agent
    agent_executor = create_react_agent(llm, tools, prompt=prompt)
    
    return agent_executor


async def execute_step(state: PlanExecuteState) -> Dict[str, Any]:
    """
    Execute the current step in the plan.
    
    Args:
        state: Current state with plan and past steps
        
    Returns:
        Updated state with execution results
    """
    if not state.get("plan"):
        return {"error": "No plan available to execute"}
    
    try:
        # Get the current step
        current_step_index = state.get("current_step", 0)
        
        if current_step_index >= len(state["plan"]):
            return {"error": "No more steps to execute"}
        
        current_step = state["plan"][current_step_index]
        
        # Create the executor
        executor = create_executor()
        
        # Format the task for the executor
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(state["plan"]))
        
        # Include context from past steps
        context = ""
        if state.get("past_steps"):
            context = "\n\nContext from previous steps:\n"
            for step, result in state["past_steps"]:
                context += f"- {step}: {result[:200]}...\n"
        
        task_formatted = f"""For the following plan:
{plan_str}

You are tasked with executing step {current_step_index + 1}: {current_step}

{context}

Execute this step and provide the results."""
        
        # Execute the step
        agent_response = await executor.ainvoke({
            "messages": [("user", task_formatted)]
        })
        
        # Extract the final response
        final_message = agent_response["messages"][-1].content
        
        # Update the state
        return {
            "past_steps": [(current_step, final_message)],
            "current_step": current_step_index + 1,
            "error": None
        }
        
    except Exception as e:
        return {
            "error": f"Step execution failed: {str(e)}"
        }


def create_research_agent(
    es_client=None,
    index_name: str = "research-publications-static",
    model_name: str = "gpt-4-turbo-preview"
) -> Any:
    """
    Create a complete research agent with tools initialized.
    
    Args:
        es_client: Elasticsearch client instance
        index_name: Name of the publications index
        model_name: OpenAI model to use
        
    Returns:
        Configured research agent
    """
    # Initialize tools if ES client is provided
    if es_client:
        from ..tools.elasticsearch_tools import initialize_elasticsearch_tools
        tools = initialize_elasticsearch_tools(es_client, index_name)
    else:
        tools = create_elasticsearch_tools()
    
    # Create the executor
    executor = create_executor(tools=tools, model_name=model_name)
    
    return executor


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def test_executor():
        # Create a mock state
        state = {
            "input": "How many papers has Christian Fager published?",
            "plan": [
                "Search for publications by Christian Fager",
                "Count the total number of publications found"
            ],
            "past_steps": [],
            "current_step": 0
        }
        
        # Execute the first step
        result = await execute_step(state)
        print("Execution result:", result)
    
    # Note: This will fail without proper ES client initialization
    # asyncio.run(test_executor())