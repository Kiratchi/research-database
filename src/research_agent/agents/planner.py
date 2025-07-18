"""
LangGraph-based planner for research publications agent.

Following LangChain's official plan-and-execute pattern.
"""

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..core.models import Plan, Act, Response
from ..core.state import PlanExecuteState


def create_planner(model_name: str = "gpt-4o", temperature: float = 0) -> Any:
    """
    Create a planner that generates execution plans for research queries.
    
    Args:
        model_name: OpenAI model to use
        temperature: Temperature for generation
        
    Returns:
        LangChain runnable for planning
    """
    planner_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """For the given research publications query, come up with a simple step by step plan.
This plan should involve individual tasks that, if executed correctly, will yield the correct answer.
Do not add any superfluous steps. The result of the final step should be the final answer.
Make sure that each step has all the information needed - do not skip steps.

Available tools for research publications:
- search_publications: Search for publications by topic, keywords, or filters (supports pagination with offset parameter)
- search_by_author: Search for publications by a specific author (supports pagination with offset parameter)
- get_field_statistics: Analyze field distributions from search results
- get_publication_details: Get detailed information about a specific publication
- get_database_summary: Get overall database statistics

Important pagination guidelines:
- Both search_publications and search_by_author support pagination with offset parameter
- Default max_results is 10, use offset to get more results (e.g., offset=10 for next page)
- Check pagination info in results to see if more pages are available
- For large result sets (>50 items), consider using multiple paginated calls
- Always inform users about total results and provide pagination strategy

Focus on research publication queries like:
- Author publication counts (may require pagination for prolific authors)
- Topic searches with filters
- Publication statistics and trends
- Cross-referencing different searches"""
        ),
        ("placeholder", "{messages}"),
    ])
    
    planner = planner_prompt | ChatOpenAI(
        model=model_name,
        temperature=temperature
    ).with_structured_output(Plan)
    
    return planner


def create_replanner(model_name: str = "gpt-4o", temperature: float = 0) -> Any:
    """
    Create a replanner that modifies plans based on execution results.
    
    Args:
        model_name: OpenAI model to use  
        temperature: Temperature for generation
        
    Returns:
        LangChain runnable for replanning
    """
    replanner_prompt = ChatPromptTemplate.from_template(
        """For the given research publications objective, come up with a simple step by step plan.
This plan should involve individual tasks that, if executed correctly, will yield the correct answer.
Do not add any superfluous steps. The result of the final step should be the final answer.
Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the following steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that.
Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done.
Do not return previously done steps as part of the plan.

Available tools for research publications:
- search_publications: Search for publications by topic, keywords, or filters  
- search_by_author: Search for publications by a specific author
- get_more_results: Get additional pages from a previous search
- get_field_statistics: Analyze field distributions from search results
- get_statistics_summary: Get overall database statistics

Consider the results from previous steps when deciding what to do next.
If you have enough information to answer the user's question, provide a Response.
If you need more information, provide a Plan with the remaining steps."""
    )
    
    replanner = replanner_prompt | ChatOpenAI(
        model=model_name,
        temperature=temperature
    ).with_structured_output(Act)
    
    return replanner


async def plan_step(state: PlanExecuteState) -> Dict[str, Any]:
    """
    Planning step for the research agent.
    
    Args:
        state: Current state containing the input query
        
    Returns:
        Updated state with generated plan
    """
    planner = create_planner()
    
    try:
        plan = await planner.ainvoke({"messages": [("user", state["input"])]})
        
        return {
            "plan": plan.steps,
            "current_step": 0,
            "error": None
        }
    
    except Exception as e:
        return {
            "plan": [],
            "error": f"Planning failed: {str(e)}"
        }


async def replan_step(state: PlanExecuteState) -> Dict[str, Any]:
    """
    Replanning step for the research agent.
    
    Args:
        state: Current state with past steps and results
        
    Returns:
        Updated state with new plan or final response
    """
    replanner = create_replanner()
    
    try:
        # Format past steps for the replanner
        past_steps_str = "\n".join([
            f"Step: {step}\nResult: {result}"
            for step, result in state["past_steps"]
        ])
        
        output = await replanner.ainvoke({
            "input": state["input"],
            "plan": "\n".join([f"{i+1}. {step}" for i, step in enumerate(state["plan"])]),
            "past_steps": past_steps_str
        })
        
        if isinstance(output.action, Response):
            return {
                "response": output.action.response,
                "error": None
            }
        else:
            return {
                "plan": output.action.steps,
                "current_step": 0,
                "error": None
            }
            
    except Exception as e:
        return {
            "error": f"Replanning failed: {str(e)}"
        }


def should_end(state: PlanExecuteState) -> str:
    """
    Determine if the agent should end or continue.
    
    Args:
        state: Current state
        
    Returns:
        Next step: "END" to finish, "agent" to continue execution
    """
    if state.get("error"):
        return "END"
    
    if state.get("response"):
        return "END"
    
    if not state.get("plan"):
        return "END"
    
    return "agent"


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def test_planner():
        planner = create_planner()
        
        plan = await planner.ainvoke({
            "messages": [("user", "How many papers has Christian Fager published?")]
        })
        
        print("Generated plan:")
        for i, step in enumerate(plan.steps, 1):
            print(f"{i}. {step}")
    
    asyncio.run(test_planner())