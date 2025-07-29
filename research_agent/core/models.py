"""
Pydantic models for the research agent.

Following LangChain's official plan-and-execute pattern with structured output.
"""

from typing import List, Union, Optional
from pydantic import BaseModel, Field


class Plan(BaseModel):
    """Plan to follow for query execution."""
    
    steps: List[str] = Field(
        description="Different steps to follow, should be in sorted order"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "steps": [
                    "Search for publications by Christian Fager",
                    "Count the total number of publications found"
                ]
            }
        }


class Response(BaseModel):
    """Final response to user query."""
    
    response: str = Field(
        description="The final answer to the user's question"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Christian Fager has published 25 research papers."
            }
        }


class Act(BaseModel):
    """Action to perform - either respond or continue planning."""
    
    action_type: str = Field(
        description="Type of action: 'response' to respond to user, 'plan' to continue planning"
    )
    response: Optional[str] = Field(
        default=None,
        description="Final response to user (only if action_type is 'response')"
    )
    steps: Optional[List[str]] = Field(
        default=None,
        description="Plan steps to execute (only if action_type is 'plan')"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "action_type": "response",
                "response": "The answer is 42.",
                "steps": None
            }
        }


# Legacy models for backward compatibility
class QueryIntent:
    """Legacy query intent enumeration."""
    COUNT = "count"
    LIST = "list"
    SEARCH = "search"
    STATS = "stats"
    UNKNOWN = "unknown"


class ParsedQuery(BaseModel):
    """Legacy parsed query model."""
    intent: str
    entity_type: str = None
    author_name: str = None
    journal_name: str = None
    search_terms: str = None
    limit: int = None
    group_by: str = None
    filters: dict = Field(default_factory=dict)
    confidence: float = 0.5