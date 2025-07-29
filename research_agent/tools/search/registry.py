"""Tool registry for easy access to all tools."""
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch

from .unified_search import UnifiedSearchTool
from ..config.settings import settings


def get_all_tools(
    es_client: Optional[Elasticsearch] = None,
    index_name: Optional[str] = None
) -> List[Any]:
    """
    Get all available tools configured with the provided Elasticsearch client.
    
    Args:
        es_client: Elasticsearch client instance. If None, creates one from settings.
        index_name: Index name to use. If None, uses default from settings.
        
    Returns:
        List of configured LangChain tools
    """
    # Create ES client if not provided
    if es_client is None:
        es_client = settings.get_es_client()
    
    # Use default index if not provided
    if index_name is None:
        index_name = settings.ES_INDEX
    
    # Initialize all tools
    tools = [
        UnifiedSearchTool(es_client=es_client, index_name=index_name),
        # Future tools will be added here:
        # GetDocumentTool(es_client=es_client, index_name=index_name),
        # AggregateStatsTool(es_client=es_client, index_name=index_name),
        # etc.
    ]
    
    return tools


def get_tool_by_name(
    name: str,
    es_client: Optional[Elasticsearch] = None,
    index_name: Optional[str] = None
) -> Optional[Any]:
    """
    Get a specific tool by name.
    
    Args:
        name: Tool name (e.g., "unified_search")
        es_client: Elasticsearch client instance
        index_name: Index name to use
        
    Returns:
        The requested tool or None if not found
    """
    tools = get_all_tools(es_client=es_client, index_name=index_name)
    
    for tool in tools:
        if tool.name == name:
            return tool
    
    return None


def get_tools_descriptions() -> Dict[str, str]:
    """
    Get a dictionary of tool names and their descriptions.
    
    Useful for understanding what tools are available.
    
    Returns:
        Dict mapping tool names to descriptions
    """
    # Create temporary tools just to get metadata
    tools = get_all_tools()
    
    return {
        tool.name: tool.description
        for tool in tools
    }