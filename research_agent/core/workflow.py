"""
Main LangGraph workflow for the research publications agent - FIXED VERSION

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

def create_research_workflow(es_client=None, index_name: str = "research-publications-static") -> StateGraph:
    if es_client:
        initialize_elasticsearch_tools(es_client, index_name)

    tools = create_elasticsearch_tools()
    planning_tool_descriptions = get_tool_descriptions_for_planning()
    execution_tool_descriptions = get_tool_descriptions_for_execution()
    load_dotenv()

    llm = ChatLiteLLM(
        model="anthropic/claude-sonnet-4",
        api_key=os.getenv("LITELLM_API_KEY"),
        api_base=os.getenv("LITELLM_BASE_URL"),
        temperature=0
    )

    executor_prompt = get_executor_prompt(execution_tool_descriptions)
    agent_executor = create_react_agent(llm, tools, prompt=executor_prompt)

    planner_system_prompt = get_planner_system_prompt(planning_tool_descriptions)
    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", planner_system_prompt),
        ("placeholder", "{messages}"),
    ])
    planner = planner_prompt | llm.with_structured_output(Plan)

    replanner_llm = ChatLiteLLM(
        model="anthropic/claude-sonet-3.7",
        api_key=os.getenv("LITELLM_API_KEY"),
        api_base=os.getenv("LITELLM_BASE_URL"),
        temperature=0
    )

    replanner_template = get_replanner_prompt(planning_tool_descriptions)
    replanner_prompt = ChatPromptTemplate.from_template(replanner_template)
    replanner = replanner_prompt | replanner_llm.with_structured_output(Act)

    def execute_step(state: PlanExecuteState):
        plan = state["plan"]
        past_steps = state.get("past_steps", [])
        if not plan:
            return {"past_steps": []}

        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        original_query = state.get("input", "")
        task_formatted = get_task_format_template(original_query=original_query, plan_str=plan_str, task=task)

        try:
            agent_response = agent_executor.invoke({"messages": [("user", task_formatted)]})
            response_content = agent_response["messages"][-1].content
            return {"past_steps": [(task, response_content)]}
        except Exception as e:
            return {"past_steps": [(task, f"Error executing task: {str(e)}")]}               

    def plan_step(state: PlanExecuteState):
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

            plan = planner.invoke({"messages": messages})

            if plan is None or not hasattr(plan, 'steps'):
                return {"plan": [f"Search for information about: {query}"]}
            return {"plan": plan.steps}
        except Exception as e:
            fallback_plan = [f"Search for information about: {query}"]
            return {"plan": fallback_plan}

    def replan_step(state: PlanExecuteState):
        try:
            output = replanner.invoke(state)
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

def compile_research_agent(es_client=None, index_name: str = "research-publications-static", recursion_limit: int = 50) -> Any:
    workflow = create_research_workflow(es_client, index_name)
    app = workflow.compile()
    return app

def run_research_query(query: str, es_client=None, index_name: str = "research-publications-static", recursion_limit: int = 50, stream: bool = False, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    app = compile_research_agent(es_client, index_name, recursion_limit)
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
    config = {"recursion_limit": recursion_limit}
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
    def __init__(self, es_client=None, index_name: str = "research-publications-static", recursion_limit: int = 50):
        self.es_client = es_client
        self.index_name = index_name
        self.recursion_limit = recursion_limit
        self.app = None
        self._compile_agent()

    def _compile_agent(self):
        self.app = compile_research_agent(self.es_client, self.index_name, self.recursion_limit)

    async def stream_query(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Any:
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