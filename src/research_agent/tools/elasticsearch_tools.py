"""
Elasticsearch tools for research publications.

Following LangChain best practices from DEMO_plan-and-execute.ipynb
Compatible with Elasticsearch 6.8.23 server using elasticsearch>=7.0.0 client
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from langchain.tools import Tool
from langchain_core.tools import BaseTool
from elasticsearch import Elasticsearch
from datetime import datetime, timedelta
import json
import uuid


# Global ES client and index name
_es_client = None
_index_name = "research-publications-static"


def initialize_elasticsearch_tools(es_client: Elasticsearch, index_name: str = "research-publications-static"):
    """Initialize the Elasticsearch tools with client and index name."""
    global _es_client, _index_name
    _es_client = es_client
    _index_name = index_name


# Pydantic v2 schemas for tool inputs following LangChain patterns
class SearchPublicationsInput(BaseModel):
    """Input schema for searching publications."""
    query: str = Field(description="Search query string")
    max_results: int = Field(default=10, description="Maximum number of results to return")
    fields: Optional[List[str]] = Field(default=None, description="Specific fields to search in")


class SearchByAuthorInput(BaseModel):
    """Input schema for searching by author."""
    author_name: str = Field(description="Author name to search for")
    strategy: str = Field(default="partial", description="Search strategy: exact, partial, or fuzzy")
    max_results: int = Field(default=10, description="Maximum number of results to return")


class GetFieldStatisticsInput(BaseModel):
    """Input schema for getting field statistics."""
    field: str = Field(description="Field name to get statistics for")
    size: int = Field(default=10, description="Number of top values to return")


class GetPublicationDetailsInput(BaseModel):
    """Input schema for getting publication details."""
    publication_id: str = Field(description="Publication ID to get details for")


# Core search functions compatible with ES 6.8.23
def search_publications(query: str, max_results: int = 10, fields: Optional[List[str]] = None) -> str:
    """
    Search publications using full-text search.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        fields: Specific fields to search in
        
    Returns:
        JSON string with search results
    """
    if not _es_client:
        return json.dumps({"error": "Elasticsearch client not initialized"})
    
    try:
        # Build search query compatible with ES 6.8.23
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": fields or ["Title^2", "Abstract", "Persons.PersonData.DisplayName", "Keywords"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            },
            "size": max_results,
            "sort": [{"_score": {"order": "desc"}}]
        }
        
        response = _es_client.search(
            index=_index_name,
            body=search_body
        )
        
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            
            # Extract author information from Persons field
            authors = []
            persons = source.get('Persons', [])
            for person in persons:
                person_data = person.get('PersonData', {})
                display_name = person_data.get('DisplayName', '')
                if display_name:
                    authors.append(display_name)
            
            result = {
                "id": hit['_id'],
                "score": hit['_score'],
                "title": source.get('Title', 'No title'),
                "authors": ', '.join(authors) if authors else 'No authors',
                "year": source.get('Year', 'No year'),
                "abstract": source.get('Abstract', 'No abstract')[:300] + "..." if source.get('Abstract', '') else "No abstract"
            }
            results.append(result)
        
        return json.dumps({
            "total_hits": response['hits']['total'],
            "results": results,
            "query": query
        })
        
    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})


def search_by_author(author_name: str, strategy: str = "partial", max_results: int = 10) -> str:
    """
    Search publications by author name with different strategies.
    
    Args:
        author_name: Author name to search for
        strategy: Search strategy (exact, partial, fuzzy)
        max_results: Maximum number of results to return
        
    Returns:
        JSON string with search results
    """
    if not _es_client:
        return json.dumps({"error": "Elasticsearch client not initialized"})
    
    try:
        # Build query based on strategy - using Persons field structure
        if strategy == "exact":
            query = {
                "match_phrase": {
                    "Persons.PersonData.DisplayName": author_name
                }
            }
        elif strategy == "fuzzy":
            query = {
                "fuzzy": {
                    "Persons.PersonData.DisplayName": {
                        "value": author_name,
                        "fuzziness": "AUTO"
                    }
                }
            }
        else:  # partial (default) - use match_phrase for exact name matching
            query = {
                "match_phrase": {
                    "Persons.PersonData.DisplayName": author_name
                }
            }
        
        search_body = {
            "query": query,
            "size": max_results,
            "sort": [{"year": {"order": "desc"}}]
        }
        
        try:
            response = _es_client.search(
                index=_index_name,
                body=search_body
            )
        except Exception as sort_error:
            # Fallback: Try without sorting if sorting fails due to mapping issues
            search_body = {
                "query": query,
                "size": max_results
            }
            response = _es_client.search(
                index=_index_name,
                body=search_body
            )
        
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            
            # Extract author information from Persons field
            authors = []
            persons = source.get('Persons', [])
            for person in persons:
                person_data = person.get('PersonData', {})
                display_name = person_data.get('DisplayName', '')
                if display_name:
                    authors.append(display_name)
            
            result = {
                "id": hit['_id'],
                "title": source.get('Title', 'No title'),
                "authors": ', '.join(authors) if authors else 'No authors',
                "year": source.get('Year', 'No year'),
                "journal": source.get('Source', 'No journal'),
                "publication_type": source.get('PublicationType', 'No type'),
                "abstract": source.get('Abstract', 'No abstract')[:200] + "..." if source.get('Abstract') else 'No abstract'
            }
            results.append(result)
        
        return json.dumps({
            "total_hits": response['hits']['total'],
            "results": results,
            "author": author_name,
            "strategy": strategy
        })
        
    except Exception as e:
        return json.dumps({"error": f"Author search failed: {str(e)}"})


def get_field_statistics(field: str, size: int = 10) -> str:
    """
    Get statistics for a specific field.
    
    Args:
        field: Field name to get statistics for
        size: Number of top values to return
        
    Returns:
        JSON string with field statistics
    """
    if not _es_client:
        return json.dumps({"error": "Elasticsearch client not initialized"})
    
    try:
        # ES 6.8.23 compatible aggregation
        search_body = {
            "size": 0,
            "aggs": {
                "field_stats": {
                    "terms": {
                        "field": f"{field}.keyword" if field in ["Persons.PersonData.DisplayName", "Source", "PublicationType"] else field,
                        "size": size
                    }
                }
            }
        }
        
        response = _es_client.search(
            index=_index_name,
            body=search_body
        )
        
        buckets = response['aggregations']['field_stats']['buckets']
        stats = [{"value": bucket['key'], "count": bucket['doc_count']} for bucket in buckets]
        
        return json.dumps({
            "field": field,
            "total_documents": response['hits']['total'],
            "top_values": stats
        })
        
    except Exception as e:
        return json.dumps({"error": f"Statistics failed: {str(e)}"})


def get_publication_details(publication_id: str) -> str:
    """
    Get detailed information about a specific publication.
    
    Args:
        publication_id: Publication ID to get details for
        
    Returns:
        JSON string with publication details
    """
    if not _es_client:
        return json.dumps({"error": "Elasticsearch client not initialized"})
    
    try:
        response = _es_client.get(
            index=_index_name,
            id=publication_id
        )
        
        source = response['_source']
        
        # Extract author information from Persons field
        authors = []
        persons = source.get('Persons', [])
        for person in persons:
            person_data = person.get('PersonData', {})
            display_name = person_data.get('DisplayName', '')
            if display_name:
                authors.append(display_name)
        
        details = {
            "id": publication_id,
            "title": source.get('Title', 'No title'),
            "authors": ', '.join(authors) if authors else 'No authors',
            "year": source.get('Year', 'No year'),
            "journal": source.get('Source', 'No journal'),
            "publication_type": source.get('PublicationType', 'No type'),
            "abstract": source.get('Abstract', 'No abstract'),
            "keywords": source.get('Keywords', 'No keywords'),
            "doi": source.get('IdentifierDoi', 'No DOI'),
            "url": source.get('DetailsUrlEng', 'No URL')
        }
        
        return json.dumps(details)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to get publication details: {str(e)}"})


def get_statistics_summary() -> Dict[str, Any]:
    """
    Get a summary of database statistics.
    
    Returns:
        Dictionary with database statistics
    """
    if not _es_client:
        return {"error": "Elasticsearch client not initialized"}
    
    try:
        # Get total count
        count_response = _es_client.count(index=_index_name)
        total_publications = count_response['count']
        
        # Get top years
        years_response = _es_client.search(
            index=_index_name,
            body={
                "size": 0,
                "aggs": {
                    "years": {
                        "terms": {
                            "field": "Year",
                            "size": 5,
                            "order": {"_key": "desc"}
                        }
                    }
                }
            }
        )
        
        # Get top publication types
        types_response = _es_client.search(
            index=_index_name,
            body={
                "size": 0,
                "aggs": {
                    "types": {
                        "terms": {
                            "field": "PublicationType.keyword",
                            "size": 5
                        }
                    }
                }
            }
        )
        
        # Get author count (approximate)
        authors_response = _es_client.search(
            index=_index_name,
            body={
                "size": 0,
                "aggs": {
                    "author_count": {
                        "cardinality": {
                            "field": "Persons.PersonData.DisplayName.keyword"
                        }
                    }
                }
            }
        )
        
        years = [{"value": b['key'], "count": b['doc_count']} for b in years_response['aggregations']['years']['buckets']]
        types = [{"value": b['key'], "count": b['doc_count']} for b in types_response['aggregations']['types']['buckets']]
        
        return {
            "total_publications": total_publications,
            "latest_year": years[0]['value'] if years else None,
            "most_common_type": types[0]['value'] if types else None,
            "total_authors": authors_response['aggregations']['author_count']['value'],
            "years": years,
            "publication_types": types
        }
        
    except Exception as e:
        return {"error": f"Failed to get statistics: {str(e)}"}


# Create LangChain tools following DEMO patterns
def create_elasticsearch_tools() -> List[BaseTool]:
    """Create a list of Elasticsearch tools for LangChain agents."""
    
    tools = [
        Tool(
            name="search_publications",
            description="Search research publications using full-text search. Use this for general queries about topics, keywords, or content.",
            func=lambda query: search_publications(query, max_results=10),
            args_schema=SearchPublicationsInput
        ),
        
        Tool(
            name="search_by_author",
            description="Search publications by author name. Use 'exact' for full name matches, 'partial' for surname matches, 'fuzzy' for typo-tolerant search.",
            func=lambda author_name, strategy="partial": search_by_author(author_name, strategy, max_results=10),
            args_schema=SearchByAuthorInput
        ),
        
        Tool(
            name="get_field_statistics",
            description="Get statistics for a specific field like 'year', 'authors', 'journal', or 'publication_type'. Shows top values and counts.",
            func=lambda field: get_field_statistics(field, size=10),
            args_schema=GetFieldStatisticsInput
        ),
        
        Tool(
            name="get_publication_details",
            description="Get detailed information about a specific publication using its ID.",
            func=get_publication_details,
            args_schema=GetPublicationDetailsInput
        ),
        
        Tool(
            name="get_database_summary",
            description="Get a summary of the database including total publications, top years, publication types, and author counts.",
            func=lambda: json.dumps(get_statistics_summary())
        )
    ]
    
    return tools


# Convenience function for backward compatibility
def get_elasticsearch_tools() -> List[BaseTool]:
    """Get all Elasticsearch tools for the research agent."""
    return create_elasticsearch_tools()