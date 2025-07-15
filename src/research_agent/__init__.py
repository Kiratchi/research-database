"""
Research Publications Agent

A LangChain-based plan-and-execute agent for querying research publications
using Elasticsearch with natural language processing capabilities.
"""

__version__ = "0.2.0"
__author__ = "Research Agent Team"

from .core.state import PlanExecuteState
from .core.models import Plan, Response, Act
from .agents.planner import create_planner
from .agents.executor import create_executor
from .tools.elasticsearch_tools import create_elasticsearch_tools

__all__ = [
    "PlanExecuteState",
    "Plan",
    "Response", 
    "Act",
    "create_planner",
    "create_executor",
    "create_elasticsearch_tools",
]