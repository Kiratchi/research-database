"""
Research Publications Agent
A LangChain-based plan-and-execute agent for querying research publications
using Elasticsearch with natural language processing capabilities.

Now using standard LangGraph plan-and-execute patterns for better reliability.
"""

__version__ = "0.3.0"  # Updated version for standard implementation
__author__ = "Research Agent Team"

from .core.state import PlanExecuteState
from .core.models import Plan, Response, Act
from .core.workflow import ResearchAgent, run_research_query, compile_research_agent

# NEW TOOLS INTEGRATION
from .tools import get_all_tools

__all__ = [
    "PlanExecuteState",
    "Plan", 
    "Response", 
    "Act",
    "ResearchAgent",
    "run_research_query", 
    "compile_research_agent",
    "get_all_tools",  # New tools system
]