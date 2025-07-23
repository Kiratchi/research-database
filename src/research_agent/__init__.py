"""
Research Publications Agent
A LangChain-based plan-and-execute agent for querying research publications
using Elasticsearch with natural language processing capabilities.
"""

__version__ = "0.2.0"
__author__ = "Research Agent Team"

from .core.state import PlanExecuteState
from .core.models import Plan, Response, Act
from .core.workflow import ResearchAgent, run_research_query, compile_research_agent
from .tools.elasticsearch_tools import create_elasticsearch_tools

__all__ = [
    "PlanExecuteState",
    "Plan", 
    "Response",
    "Act",
    "ResearchAgent",
    "run_research_query", 
    "compile_research_agent",
    "create_elasticsearch_tools",
]