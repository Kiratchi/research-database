"""
Pydantic models for the research agent.

Following LangChain's official plan-and-execute pattern with structured output.
"""

from typing import List, Union
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
    
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "action": {
                    "response": "The answer is 42."
                }
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