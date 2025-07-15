"""
LLM-based query planner for research publications agent.

This module provides a plan-and-execute approach using OpenAI's function calling
to generate execution plans for natural language queries about research publications.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.tools import Tool
from pydantic.v1 import BaseModel, Field

from langchain_tools import create_langchain_tools

load_dotenv()


class PlanStepType(Enum):
    """Types of plan steps."""
    SEARCH = "search"
    ANALYZE = "analyze"
    COMPARE = "compare"
    AGGREGATE = "aggregate"


@dataclass
class PlanStep:
    """A single step in an execution plan."""
    step_id: str
    step_type: PlanStepType
    tool_name: str
    parameters: Dict[str, Any]
    description: str
    depends_on: Optional[List[str]] = None
    output_variable: Optional[str] = None


@dataclass
class ExecutionPlan:
    """Complete execution plan for a query."""
    query: str
    steps: List[PlanStep]
    final_output: str
    estimated_complexity: int  # 1-5 scale


class QueryPlanner:
    """LLM-based query planner using OpenAI function calling."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1):
        """
        Initialize the query planner.
        
        Args:
            model_name: OpenAI model to use for planning
            temperature: Temperature for LLM responses (lower = more deterministic)
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key
        )
        
        # Get available tools
        self.tools = create_langchain_tools()
        self.tool_descriptions = self._create_tool_descriptions()
        
        # System message for planning
        self.system_message = self._create_system_message()
    
    def _create_tool_descriptions(self) -> str:
        """Create a formatted description of available tools."""
        descriptions = []
        for tool in self.tools:
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)
    
    def _create_system_message(self) -> str:
        """Create the system message for the planning LLM."""
        return f"""You are a query planning assistant for a research publications database. 
Your job is to analyze natural language queries and create step-by-step execution plans.

Available tools:
{self.tool_descriptions}

For each query, you should:
1. Understand the user's intent
2. Break down complex queries into simple steps
3. Choose the appropriate tools for each step
4. Consider parameter passing between steps
5. Provide clear, actionable plans

Guidelines:
- Use search_by_author for author-specific queries
- Use search_publications for topic/keyword searches
- Use get_field_statistics for trend analysis
- Use get_statistics_summary for general database info
- Always consider whether follow-up steps are needed

Your response should be a JSON object with this structure:
{{
    "query_analysis": {{
        "intent": "description of what the user wants",
        "complexity": 1-5,
        "query_type": "author_search|topic_search|statistics|comparison|trend_analysis"
    }},
    "execution_plan": {{
        "steps": [
            {{
                "step_id": "step_1",
                "tool_name": "tool_name",
                "parameters": {{"param": "value"}},
                "description": "What this step does",
                "depends_on": ["step_id"] or null,
                "output_variable": "variable_name"
            }}
        ],
        "final_output": "Description of expected final result"
    }}
}}

Be concise but thorough. Always provide working parameters for each tool."""
    
    def plan_query(self, query: str) -> ExecutionPlan:
        """
        Generate an execution plan for a natural language query.
        
        Args:
            query: Natural language query about research publications
            
        Returns:
            ExecutionPlan object with step-by-step instructions
        """
        # Create messages for the LLM
        messages = [
            SystemMessage(content=self.system_message),
            HumanMessage(content=f"Plan how to answer this query: {query}")
        ]
        
        # Get response from LLM
        response = self.llm(messages)
        
        # Parse the JSON response
        try:
            plan_data = json.loads(response.content)
            return self._parse_plan_response(query, plan_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
    
    def _parse_plan_response(self, query: str, plan_data: Dict) -> ExecutionPlan:
        """Parse the LLM response into an ExecutionPlan object."""
        analysis = plan_data.get("query_analysis", {})
        plan = plan_data.get("execution_plan", {})
        
        # Parse steps
        steps = []
        for step_data in plan.get("steps", []):
            step_type = self._infer_step_type(step_data.get("tool_name", ""))
            
            step = PlanStep(
                step_id=step_data.get("step_id"),
                step_type=step_type,
                tool_name=step_data.get("tool_name"),
                parameters=step_data.get("parameters", {}),
                description=step_data.get("description", ""),
                depends_on=step_data.get("depends_on"),
                output_variable=step_data.get("output_variable")
            )
            steps.append(step)
        
        return ExecutionPlan(
            query=query,
            steps=steps,
            final_output=plan.get("final_output", ""),
            estimated_complexity=analysis.get("complexity", 1)
        )
    
    def _infer_step_type(self, tool_name: str) -> PlanStepType:
        """Infer the step type from the tool name."""
        if tool_name in ["search_publications", "search_by_author"]:
            return PlanStepType.SEARCH
        elif tool_name in ["get_field_statistics"]:
            return PlanStepType.ANALYZE
        elif tool_name in ["get_statistics_summary"]:
            return PlanStepType.AGGREGATE
        else:
            return PlanStepType.SEARCH  # Default fallback
    
    def validate_plan(self, plan: ExecutionPlan) -> List[str]:
        """
        Validate an execution plan for potential issues.
        
        Args:
            plan: The execution plan to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check if we have tools for all steps
        tool_names = [tool.name for tool in self.tools]
        for step in plan.steps:
            if step.tool_name not in tool_names:
                errors.append(f"Unknown tool: {step.tool_name}")
        
        # Check step dependencies
        step_ids = [step.step_id for step in plan.steps]
        for step in plan.steps:
            if step.depends_on:
                for dep in step.depends_on:
                    if dep not in step_ids:
                        errors.append(f"Step {step.step_id} depends on non-existent step {dep}")
        
        # Check for required parameters
        for step in plan.steps:
            if step.tool_name == "search_by_author" and "author_name" not in step.parameters:
                errors.append(f"Step {step.step_id}: search_by_author requires author_name parameter")
            elif step.tool_name == "get_more_results" and "session_id" not in step.parameters:
                errors.append(f"Step {step.step_id}: get_more_results requires session_id parameter")
        
        return errors
    
    def explain_plan(self, plan: ExecutionPlan) -> str:
        """
        Generate a human-readable explanation of the execution plan.
        
        Args:
            plan: The execution plan to explain
            
        Returns:
            Formatted explanation string
        """
        explanation = f"Query: {plan.query}\n"
        explanation += f"Complexity: {plan.estimated_complexity}/5\n"
        explanation += f"Steps: {len(plan.steps)}\n\n"
        
        for i, step in enumerate(plan.steps, 1):
            explanation += f"Step {i}: {step.description}\n"
            explanation += f"  Tool: {step.tool_name}\n"
            explanation += f"  Parameters: {step.parameters}\n"
            if step.depends_on:
                explanation += f"  Depends on: {step.depends_on}\n"
            explanation += "\n"
        
        explanation += f"Expected output: {plan.final_output}\n"
        
        return explanation


# Example usage functions
def create_simple_plans():
    """Create some example simple plans for testing."""
    examples = [
        {
            "query": "How many papers has Christian Fager published?",
            "expected_plan": {
                "steps": [
                    {
                        "step_id": "step_1",
                        "tool_name": "search_by_author",
                        "parameters": {"author_name": "Christian Fager"},
                        "description": "Search for all publications by Christian Fager"
                    }
                ],
                "complexity": 1
            }
        },
        {
            "query": "Find machine learning papers from 2023",
            "expected_plan": {
                "steps": [
                    {
                        "step_id": "step_1", 
                        "tool_name": "search_publications",
                        "parameters": {"query": "machine learning", "filters": {"year": 2023}},
                        "description": "Search for machine learning publications from 2023"
                    }
                ],
                "complexity": 1
            }
        },
        {
            "query": "What are the publication trends by year?",
            "expected_plan": {
                "steps": [
                    {
                        "step_id": "step_1",
                        "tool_name": "get_statistics_summary", 
                        "parameters": {},
                        "description": "Get overall database statistics including year distribution"
                    }
                ],
                "complexity": 2
            }
        }
    ]
    
    return examples


if __name__ == "__main__":
    # Example usage
    planner = QueryPlanner()
    
    # Test with a simple query
    query = "How many papers has Christian Fager published?"
    plan = planner.plan_query(query)
    
    print("Generated Plan:")
    print(planner.explain_plan(plan))
    
    # Validate the plan
    errors = planner.validate_plan(plan)
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"- {error}")
    else:
        print("Plan is valid!")