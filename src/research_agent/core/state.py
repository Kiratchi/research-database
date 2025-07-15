"""
State management for the research agent.

Following LangChain's official plan-and-execute pattern with LangGraph state.
"""

import operator
from typing import Annotated, List, Tuple, Optional
from typing_extensions import TypedDict


class PlanExecuteState(TypedDict):
    """State for plan-and-execute agent."""
    
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple[str, str]], operator.add]
    response: Optional[str]
    
    # Additional fields for research agent
    session_id: Optional[str]
    total_results: Optional[int]
    current_step: Optional[int]
    error: Optional[str]


class AgentState(TypedDict):
    """Legacy agent state for backward compatibility."""
    
    query: str
    parsed_query: dict
    search_results: dict
    formatted_response: str
    debug_info: dict
    session_id: str