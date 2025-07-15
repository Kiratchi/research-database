"""
LangChain tools for Elasticsearch research publications.

These tools wrap the existing agent_tools.py functions with proper LangChain
tool schemas and documentation for use with LangChain agents.
"""

from typing import Dict, List, Any, Optional
from pydantic.v1 import BaseModel, Field
from langchain.tools import Tool
from langchain.schema import AgentAction, AgentFinish

# Import existing agent tools
from agent_tools import (
    initialize_tools,
    search_publications,
    search_by_author,
    get_more_results,
    refine_search,
    get_field_statistics,
    get_publication_details,
    get_statistics_summary,
    list_active_sessions
)


# Pydantic schemas for tool inputs
class SearchPublicationsInput(BaseModel):
    """Input schema for search_publications tool."""
    query: Optional[str] = Field(
        None,
        description="Free text search query to search in title, abstract, and keywords"
    )
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Dictionary of filters, e.g., {'year': 2023}, {'year_range': {'gte': 2020, 'lte': 2024}}, {'author': 'Fager'}"
    )
    size: int = Field(
        20,
        description="Number of results to retrieve initially (default: 20)"
    )


class SearchByAuthorInput(BaseModel):
    """Input schema for search_by_author tool."""
    author_name: str = Field(
        description="Name of the author to search for (e.g., 'Christian Fager' or 'Fager')"
    )
    year_range: Optional[Dict[str, int]] = Field(
        None,
        description="Optional year range filter, e.g., {'gte': 2020, 'lte': 2024}"
    )
    strategy: str = Field(
        "auto",
        description="Search strategy: 'exact' for full names, 'partial' for surnames, 'fuzzy' for variations, 'auto' for automatic detection"
    )


class GetMoreResultsInput(BaseModel):
    """Input schema for get_more_results tool."""
    session_id: str = Field(description="Session ID from a previous search")
    page_number: int = Field(0, description="Page number to retrieve (0-based)")
    page_size: int = Field(20, description="Number of results per page")


class RefineSearchInput(BaseModel):
    """Input schema for refine_search tool."""
    session_id: str = Field(description="Session ID from a previous search")
    additional_filters: Dict[str, Any] = Field(
        description="Additional filters to apply to the existing search"
    )


class GetFieldStatisticsInput(BaseModel):
    """Input schema for get_field_statistics tool."""
    session_id: str = Field(description="Session ID from a previous search")
    field: str = Field(
        description="Field name to analyze (e.g., 'Year', 'PublicationType.NameEng')"
    )
    top_n: int = Field(20, description="Number of top values to return")


class GetPublicationDetailsInput(BaseModel):
    """Input schema for get_publication_details tool."""
    session_id: str = Field(description="Session ID from a previous search")
    position: int = Field(description="Position of the publication in search results (0-based)")


# Tool wrapper functions
def _search_publications_wrapper(
    query: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    size: int = 20
) -> str:
    """Wrapper for search_publications that returns formatted string."""
    try:
        result = search_publications(query=query, filters=filters, size=size)
        
        # Format the result for LLM consumption
        formatted_result = f"""
Search Results Summary:
- Total Results: {result['total_results']}
- Session ID: {result['session_id']}
- Sample Results: {len(result['sample_results'])} shown

Top Results:
"""
        for i, pub in enumerate(result['sample_results'][:5]):
            source = pub['_source']
            formatted_result += f"{i+1}. {source.get('Title', 'No title')}\n"
            formatted_result += f"   Authors: {', '.join(source.get('authors', []))}\n"
            formatted_result += f"   Year: {source.get('Year', 'Unknown')}\n\n"
        
        if result['aggregations']:
            formatted_result += f"\nAggregations available: {list(result['aggregations'].keys())}\n"
        
        formatted_result += f"\nUse session_id '{result['session_id']}' for follow-up queries.\n"
        
        return formatted_result
    
    except Exception as e:
        return f"Error in search_publications: {str(e)}"


def _search_by_author_wrapper(
    author_name: str,
    year_range: Optional[Dict[str, int]] = None,
    strategy: str = "auto"
) -> str:
    """Wrapper for search_by_author that returns formatted string."""
    try:
        result = search_by_author(author_name=author_name, year_range=year_range, strategy=strategy)
        
        # Format the result for LLM consumption
        formatted_result = f"""
Author Search Results:
- Author: {author_name}
- Strategy Used: {strategy}
- Total Publications: {result['total_results']}
- Session ID: {result['session_id']}

Publications Found:
"""
        for i, pub in enumerate(result['sample_results'][:5]):
            source = pub['_source']
            formatted_result += f"{i+1}. {source.get('Title', 'No title')} ({source.get('Year', 'Unknown')})\n"
        
        if result['total_results'] > 5:
            formatted_result += f"\n... and {result['total_results'] - 5} more publications.\n"
        
        formatted_result += f"\nUse session_id '{result['session_id']}' for more details.\n"
        
        return formatted_result
    
    except Exception as e:
        return f"Error in search_by_author: {str(e)}"


def _get_more_results_wrapper(
    session_id: str,
    page_number: int = 0,
    page_size: int = 20
) -> str:
    """Wrapper for get_more_results that returns formatted string."""
    try:
        result = get_more_results(session_id=session_id, page_number=page_number, page_size=page_size)
        
        if 'error' in result:
            return f"Error: {result['error']}"
        
        formatted_result = f"""
Page {page_number + 1} of {result['total_pages']} (Session: {session_id}):

"""
        for i, pub in enumerate(result['results']):
            source = pub['_source']
            formatted_result += f"{page_number * page_size + i + 1}. {source.get('Title', 'No title')}\n"
            formatted_result += f"   Authors: {', '.join(source.get('authors', []))}\n"
            formatted_result += f"   Year: {source.get('Year', 'Unknown')}\n\n"
        
        if result['has_next']:
            formatted_result += f"More results available. Use page_number={page_number + 1} for next page.\n"
        
        return formatted_result
    
    except Exception as e:
        return f"Error in get_more_results: {str(e)}"


def _get_field_statistics_wrapper(
    session_id: str,
    field: str,
    top_n: int = 20
) -> str:
    """Wrapper for get_field_statistics that returns formatted string."""
    try:
        result = get_field_statistics(session_id=session_id, field=field, top_n=top_n)
        
        if 'error' in result:
            return f"Error: {result['error']}"
        
        formatted_result = f"""
Field Statistics for '{field}' (Session: {session_id}):

Top {len(result['values'])} values:
"""
        for item in result['values']:
            formatted_result += f"- {item['value']}: {item['count']} publications\n"
        
        return formatted_result
    
    except Exception as e:
        return f"Error in get_field_statistics: {str(e)}"


def _get_statistics_summary_wrapper() -> str:
    """Wrapper for get_statistics_summary that returns formatted string."""
    try:
        result = get_statistics_summary()
        
        formatted_result = f"""
Database Statistics Summary:
- Total Publications: {result['total_publications']}

Publication Years:
"""
        for year_data in result['years'][:10]:  # Show top 10 years
            formatted_result += f"- {year_data['key']}: {year_data['doc_count']} publications\n"
        
        formatted_result += f"\nPublication Types:\n"
        for type_data in result['publication_types'][:5]:  # Show top 5 types
            formatted_result += f"- {type_data['key']}: {type_data['doc_count']} publications\n"
        
        formatted_result += f"\nTop Authors:\n"
        for author_data in result['top_authors'][:5]:  # Show top 5 authors
            formatted_result += f"- {author_data['key']}: {author_data['doc_count']} publications\n"
        
        return formatted_result
    
    except Exception as e:
        return f"Error in get_statistics_summary: {str(e)}"


def create_langchain_tools() -> List[Tool]:
    """
    Create a list of LangChain tools from the existing agent tools.
    
    Returns:
        List of LangChain Tool objects ready for use with agents
    """
    tools = [
        Tool(
            name="search_publications",
            description="""Search for research publications using natural language queries and filters. 
            This is the main search tool that can handle both text queries and structured filters.
            Use this when you need to find publications by topic, keywords, or apply filters like year ranges.
            Always returns a session_id for follow-up queries.""",
            func=_search_publications_wrapper,
            args_schema=SearchPublicationsInput
        ),
        
        Tool(
            name="search_by_author",
            description="""Search for publications by a specific author name. 
            This tool automatically detects the best search strategy (exact, partial, fuzzy) based on the input.
            Use this when you need to find all publications by a particular author.
            Examples: 'Christian Fager', 'Fager', 'Anna Dubois'""",
            func=_search_by_author_wrapper,
            args_schema=SearchByAuthorInput
        ),
        
        Tool(
            name="get_more_results",
            description="""Get additional pages of results from a previous search session.
            Use this when you need to see more publications beyond the initial results.
            Requires a session_id from a previous search.""",
            func=_get_more_results_wrapper,
            args_schema=GetMoreResultsInput
        ),
        
        Tool(
            name="get_field_statistics",
            description="""Get statistics about a specific field from search results.
            Use this to analyze the distribution of values in fields like Year, PublicationType, etc.
            Requires a session_id from a previous search.""",
            func=_get_field_statistics_wrapper,
            args_schema=GetFieldStatisticsInput
        ),
        
        Tool(
            name="get_statistics_summary",
            description="""Get overall statistics about the entire publications database.
            Use this to understand the general scope and distribution of publications.
            Does not require a session_id.""",
            func=_get_statistics_summary_wrapper,
            args_schema=None
        )
    ]
    
    return tools


# Initialize tools when module is imported
def initialize_langchain_tools(es_client, index_name: str = "research-publications-static"):
    """
    Initialize the LangChain tools with an Elasticsearch client.
    
    Args:
        es_client: Elasticsearch client instance
        index_name: Name of the publications index
    """
    # Initialize the underlying agent tools
    initialize_tools(es_client, index_name)
    
    # Return the LangChain tools
    return create_langchain_tools()


if __name__ == "__main__":
    # Example usage
    print("LangChain tools available:")
    tools = create_langchain_tools()
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")