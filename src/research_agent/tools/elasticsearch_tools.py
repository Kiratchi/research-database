"""
Enhanced Elasticsearch tools with self-contained descriptions for research publications.

Each tool now contains detailed, specific descriptions that are automatically
injected into planner and replanner prompts for better tool selection.
"""

from typing import Dict, List, Any, Optional
import json

# Core dependencies
from pydantic import BaseModel, Field
from elasticsearch import Elasticsearch


# LangChain imports with error handling  
from langchain.tools import Tool
from langchain_core.tools import BaseTool


_es_client = None
_index_name = "research-publications-static"


def initialize_elasticsearch_tools(es_client: Elasticsearch, index_name: str = "research-publications-static"):
    """Initialize the Elasticsearch tools with client and index name."""
    global _es_client, _index_name
    _es_client = es_client
    _index_name = index_name


# Enhanced Pydantic schemas with better descriptions
class SearchPublicationsInput(BaseModel):
    """Input schema for comprehensive publication search."""
    query: str = Field(
        description="Search query string (keywords, topics, concepts). Examples: 'machine learning', 'climate change', 'artificial intelligence'"
    )
    max_results: int = Field(
        default=10, 
        description="Maximum number of results to return (1-100, default: 10). Use higher values for comprehensive searches."
    )
    offset: int = Field(
        default=0, 
        description="Number of results to skip for pagination (default: 0). Use for getting additional pages: offset=10 for page 2, offset=20 for page 3, etc."
    )
    fields: Optional[List[str]] = Field(
        default=None, 
        description="Specific fields to search in. Available: 'Title', 'Abstract', 'Persons.PersonData.DisplayName', 'Keywords'. Leave empty to search all fields."
    )


class SearchByAuthorInput(BaseModel):
    """Input schema for author-specific publication search."""
    author_name: str = Field(
        description="Full author name to search for. Examples: 'John Smith', 'Maria González', 'Per-Olof Arnäs'. Use complete names for best results."
    )
    strategy: str = Field(
        default="partial", 
        description="Search strategy: 'exact' for exact phrase matching (most precise), 'partial' for standard matching (recommended), 'fuzzy' for typo-tolerant search"
    )
    max_results: int = Field(
        default=10, 
        description="Maximum number of results to return (1-100, default: 10). Use higher values for prolific authors."
    )
    offset: int = Field(
        default=0, 
        description="Number of results to skip for pagination (default: 0). Essential for authors with many publications."
    )


class GetFieldStatisticsInput(BaseModel):
    """Input schema for database field analysis."""
    field: str = Field(
        description="Field name to analyze. Valid options: 'Year' (publication years), 'Persons.PersonData.DisplayName' (authors), 'Source' (journals/venues), 'PublicationType' (article types)"
    )
    size: int = Field(
        default=10, 
        description="Number of top values to return (1-50, default: 10). Use higher values for comprehensive analysis."
    )


class GetPublicationDetailsInput(BaseModel):
    """Input schema for detailed publication information."""
    publication_id: str = Field(
        description="Elasticsearch document ID of the publication. Obtained from search results. Format: alphanumeric string."
    )


# Core search functions
def search_publications(query: str, max_results: int = 10, offset: int = 0, fields: Optional[List[str]] = None) -> str:
    """Search publications using full-text search with automatic relevance ranking."""
    if not _es_client:
        return json.dumps({"error": "Elasticsearch client not initialized"})
    
    try:
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
            "from": offset,
            "sort": [{"_score": {"order": "desc"}}]
        }
        
        response = _es_client.search(index=_index_name, body=search_body)
        
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            
            # Extract author information
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
        
        total_hits = response['hits']['total']
        if isinstance(total_hits, dict):
            total_hits = total_hits.get('value', total_hits.get('count', 0))
        
        return json.dumps({
            "total_hits": total_hits,
            "results": results,
            "query": query,
            "pagination": {
                "offset": offset,
                "limit": max_results,
                "has_more": offset + max_results < total_hits,
                "next_offset": offset + max_results if offset + max_results < total_hits else None
            }
        })
        
    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})


def search_by_author(author_name: str, strategy: str = "partial", max_results: int = 10, offset: int = 0) -> str:
    """Search publications by specific author with different matching strategies."""
    if not _es_client:
        return json.dumps({"error": "Elasticsearch client not initialized"})
    
    try:
        # Build query based on strategy
        if strategy == "exact":
            query = {"match_phrase": {"Persons.PersonData.DisplayName": author_name}}
        elif strategy == "fuzzy":
            query = {"fuzzy": {"Persons.PersonData.DisplayName": {"value": author_name, "fuzziness": "AUTO"}}}
        else:  # partial (default)
            query = {"match_phrase": {"Persons.PersonData.DisplayName": author_name}}
        
        search_body = {
            "query": query,
            "size": max_results,
            "from": offset,
            "sort": [{"Year": {"order": "desc"}}]
        }
        
        try:
            response = _es_client.search(index=_index_name, body=search_body)
        except Exception:
            # Fallback without sorting if field mapping issues
            search_body = {"query": query, "size": max_results, "from": offset}
            response = _es_client.search(index=_index_name, body=search_body)
        
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            
            # Extract author information
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
        
        total_hits = response['hits']['total']
        if isinstance(total_hits, dict):
            total_hits = total_hits.get('value', total_hits.get('count', 0))
        
        return json.dumps({
            "total_hits": total_hits,
            "results": results,
            "author": author_name,
            "strategy": strategy,
            "pagination": {
                "offset": offset,
                "limit": max_results,
                "has_more": offset + max_results < total_hits,
                "next_offset": offset + max_results if offset + max_results < total_hits else None
            }
        })
        
    except Exception as e:
        return json.dumps({"error": f"Author search failed: {str(e)}"})


def get_field_statistics(field: str, size: int = 10) -> str:
    """Analyze distribution of values in database fields."""
    if not _es_client:
        return json.dumps({"error": "Elasticsearch client not initialized"})
    
    try:
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
        
        response = _es_client.search(index=_index_name, body=search_body)
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
    """Retrieve complete information about a specific publication."""
    if not _es_client:
        return json.dumps({"error": "Elasticsearch client not initialized"})
    
    try:
        response = _es_client.get(index=_index_name, id=publication_id)
        source = response['_source']
        
        # Extract author information
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
    """Generate comprehensive database overview with key metrics."""
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
                        "terms": {"field": "Year", "size": 5, "order": {"_key": "desc"}}
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
                        "terms": {"field": "PublicationType.keyword", "size": 5}
                    }
                }
            }
        )
        
        # Get author count
        authors_response = _es_client.search(
            index=_index_name,
            body={
                "size": 0,
                "aggs": {
                    "author_count": {
                        "cardinality": {"field": "Persons.PersonData.DisplayName.keyword"}
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


# Tool registry with enhanced descriptions
TOOL_REGISTRY = [
    {
        "name": "search_publications",
        "function": search_publications,
        "args_schema": SearchPublicationsInput,
        "short_description": "Comprehensive full-text search across all publication fields",
        "detailed_description": """Search research publications using intelligent full-text search with automatic relevance ranking.

USAGE EXAMPLES:
- search_publications(query="machine learning", max_results=20, offset=0)
- search_publications(query="climate change", max_results=10, offset=10)
- search_publications(query="artificial intelligence", max_results=50)

**When to use:**
- Looking for publications on specific topics, keywords, or concepts
- Need broad search across titles, abstracts, author names, and keywords
- Want relevance-ranked results with fuzzy matching for typos
- Searching for interdisciplinary topics or general concepts

**Key features:**
- Multi-field search with title boosting (2x weight)
- Automatic fuzzy matching for typo tolerance
- Pagination support for large result sets
- Relevance scoring and sorting

**Input parameters:**
- query (REQUIRED): Search terms in quotes (e.g., "machine learning", "climate change")
- max_results (optional): Number of results (1-100, default: 10)
- offset (optional): Pagination offset (0, 10, 20, etc.)
- fields (optional): Specific fields to search

**Returns:**
- total_hits: Total matching publications
- results: Array with id, score, title, authors, year, abstract
- pagination: Navigation info (has_more, next_offset)""",
        
        "planning_guidance": {
            "use_when": [
                "User asks about topics, concepts, or keywords",
                "Need to find publications on interdisciplinary subjects",
                "Looking for research on specific technologies or methods",
                "Want comprehensive search across all fields"
            ],
            "combine_with": [
                "get_field_statistics for trend analysis",
                "get_publication_details for specific paper information"
            ],
            "pagination_strategy": "Use offset parameter for results beyond first 10-50"
        }
    },
    
    {
        "name": "search_by_author",
        "function": search_by_author,
        "args_schema": SearchByAuthorInput,
        "short_description": "Find all publications by specific authors with flexible matching",
        "detailed_description": """Search publications by author name with multiple matching strategies and comprehensive pagination.

CRITICAL USAGE EXAMPLES:
- search_by_author(author_name="Per-Olof Arnäs", max_results=10, offset=0)
- search_by_author(author_name="Per-Olof Arnäs", max_results=10, offset=10)
- search_by_author(author_name="John Smith", strategy="exact", max_results=20)
- search_by_author(author_name="Maria González", strategy="partial", max_results=15, offset=5)

**When to use:**
- Counting publications by specific authors
- Getting complete publication lists for researchers
- Analyzing author productivity over time
- Verifying author information or name variations

**Search strategies:**
- 'partial' (default): Standard phrase matching, recommended for most searches
- 'exact': Precise phrase matching, use for common names to reduce false matches  
- 'fuzzy': Typo-tolerant matching, use when uncertain about spelling

**Key features:**
- Year-based sorting (newest first) when available
- Comprehensive author metadata extraction
- Pagination essential for prolific authors
- Fallback handling for mapping issues

**Input parameters:**
- author_name (REQUIRED): Full author name in quotes (e.g., "Per-Olof Arnäs", "Maria González")
- strategy (optional): Matching approach - "partial", "exact", or "fuzzy" (default: "partial") 
- max_results (optional): Results per page (1-100, default: 10)
- offset (optional): Page offset for pagination (default: 0)

**Returns:**
- total_hits: Total publications by this author
- results: Publications with id, title, authors, year, journal, type, abstract
- pagination: Navigation info for additional pages

**Pagination strategy:**
For prolific authors (>10 publications), use multiple calls:
- First call: search_by_author(author_name="Author", max_results=10, offset=0)
- Second call: search_by_author(author_name="Author", max_results=10, offset=10) 
- Continue: offset=20, 30, etc. until has_more=false""",
        
        "planning_guidance": {
            "use_when": [
                "User asks 'How many papers has [author] published?'",
                "Need complete publication list for specific researcher",
                "Analyzing author productivity or career trajectory",
                "User references author by name"
            ],
            "combine_with": [
                "get_field_statistics to analyze publication years",
                "get_publication_details for specific paper information"
            ],
            "pagination_strategy": "Essential for prolific authors - use multiple calls with offset"
        }
    },
    
    {
        "name": "get_field_statistics",
        "function": get_field_statistics,
        "args_schema": GetFieldStatisticsInput,
        "short_description": "Analyze distribution and trends in database fields",
        "detailed_description": """Get statistical analysis of field distributions with top values and counts.

USAGE EXAMPLES:
- get_field_statistics(field="Year", size=10)
- get_field_statistics(field="Persons.PersonData.DisplayName", size=20)
- get_field_statistics(field="Source", size=15)

**When to use:**
- Analyzing publication trends by year
- Finding most prolific authors in database
- Identifying top journals or publication venues  
- Understanding publication type distributions
- Supporting data-driven insights about research landscape

**Available fields:**
- 'Year': Publication years (for temporal trend analysis)
- 'Persons.PersonData.DisplayName': Author names (for productivity analysis)
- 'Source': Journals/venues (for publication outlet analysis)
- 'PublicationType': Types (articles, books, etc.)

**Analysis capabilities:**
- Top N values with publication counts
- Percentage distributions 
- Trend identification over time
- Comparative analysis between fields

**Input parameters:**
- field (REQUIRED): Field to analyze (must be from valid options above)
- size (optional): Number of top values (1-50, default: 10)

**Returns:**
- field: Field name analyzed
- total_documents: Total publications in database
- top_values: Array of value and count objects sorted by count""",
        
        "planning_guidance": {
            "use_when": [
                "User asks about trends, distributions, or 'most/top' anything",
                "Need supporting data for author or topic analysis",
                "Analyzing publication patterns over time",
                "Comparing research activity between years or venues"
            ],
            "combine_with": [
                "search_by_author after finding top authors",
                "search_publications for specific year ranges"
            ],
            "analysis_patterns": "Use multiple calls for comparative analysis across different fields"
        }
    },
    
    {
        "name": "get_publication_details", 
        "function": get_publication_details,
        "args_schema": GetPublicationDetailsInput,
        "short_description": "Retrieve complete metadata for specific publications",
        "detailed_description": """Get comprehensive details about a specific publication using its database ID.

USAGE EXAMPLES:
- get_publication_details(publication_id="abc123def456")
- get_publication_details(publication_id="xyz789ghi012")

**When to use:**
- User asks about specific paper from search results
- Need complete publication information (abstract, DOI, URL)
- Following up on search results with detailed analysis
- User references publications by position ("the 3rd one", "that 2019 paper")

**Required input:**
- publication_id (REQUIRED): Document ID from search results (string format)

**Complete information returned:**
- Basic: id, title, authors, year, journal, publication_type
- Content: full abstract, keywords
- Identifiers: DOI, detailed URL links
- Metadata: All available database fields

**Integration with search tools:**
1. Use search_publications or search_by_author first
2. Get publication IDs from results
3. Use this tool for detailed information about specific papers
4. Reference results by position for user clarity

**Returns:**
- Complete publication record as JSON object
- All metadata fields available in database
- Formatted for easy reading and analysis""",
        
        "planning_guidance": {
            "use_when": [
                "User asks 'What is [publication] about?'",
                "Need full abstract or detailed information",
                "User references specific papers from previous results",
                "Following up search results with detailed analysis"
            ],
            "combine_with": [
                "search_publications or search_by_author to get publication IDs first"
            ],
            "reference_handling": "Use for numbered references like 'the 3rd paper' or 'that 2019 study'"
        }
    },
    
    {
        "name": "get_database_summary",
        "function": lambda: json.dumps(get_statistics_summary()),
        "args_schema": None,
        "short_description": "Get comprehensive database overview and key statistics",
        "detailed_description": """Generate high-level overview of the research publications database.

USAGE: get_database_summary() - No parameters required

**When to use:**
- User asks about database size, coverage, or general statistics
- Need context about research landscape scope
- Starting point for broad research questions
- Understanding temporal coverage and publication types

**No parameters required** - returns complete overview automatically.

**Comprehensive metrics provided:**
- total_publications: Complete database size
- latest_year: Most recent publication year
- most_common_type: Primary publication type
- total_authors: Unique author count (approximate)
- years: Top 5 publication years with counts
- publication_types: Distribution of publication types

**Use cases:**
- Database orientation for new users
- Context setting for research analysis
- Baseline metrics for comparative analysis
- Understanding scope before specific searches""",
        
        "planning_guidance": {
            "use_when": [
                "User asks general questions about database size or coverage", 
                "Need context before specific research queries",
                "User wants overview of research landscape",
                "Starting point for exploratory analysis"
            ],
            "combine_with": [
                "get_field_statistics for deeper analysis of specific aspects"
            ],
            "context_usage": "Use early in conversations to establish scope and context"
        }
    }
]


def get_tool_descriptions_for_planning() -> str:
    """Generate comprehensive tool descriptions for planner prompts."""
    descriptions = []
    
    for tool_info in TOOL_REGISTRY:
        name = tool_info["name"]
        short_desc = tool_info["short_description"]
        detailed_desc = tool_info["detailed_description"]
        guidance = tool_info.get("planning_guidance", {})
        
        # Build comprehensive description
        tool_desc = f"""**{name}**: {short_desc}

{detailed_desc}

**Planning guidance:**"""
        
        if guidance.get("use_when"):
            tool_desc += f"\n- Use when: {', '.join(guidance['use_when'])}"
            
        if guidance.get("combine_with"):
            tool_desc += f"\n- Combine with: {', '.join(guidance['combine_with'])}"
            
        if guidance.get("pagination_strategy"):
            tool_desc += f"\n- Pagination: {guidance['pagination_strategy']}"
            
        if guidance.get("analysis_patterns"):
            tool_desc += f"\n- Analysis: {guidance['analysis_patterns']}"
            
        if guidance.get("reference_handling"):
            tool_desc += f"\n- References: {guidance['reference_handling']}"
            
        if guidance.get("context_usage"):
            tool_desc += f"\n- Context: {guidance['context_usage']}"
        
        descriptions.append(tool_desc)
    
    return "\n\n" + "="*80 + "\n\n".join(descriptions)


def get_tool_descriptions_for_execution() -> str:
    """Generate concise tool descriptions for executor prompts."""
    descriptions = []
    
    for tool_info in TOOL_REGISTRY:
        name = tool_info["name"]
        short_desc = tool_info["short_description"]
        
        # Extract key parameters from args_schema (Pydantic v2 compatible)
        args_info = ""
        if tool_info["args_schema"]:
            schema = tool_info["args_schema"]
            field_info = []
            
            # Handle both Pydantic v1 and v2 field access
            try:
                # Try Pydantic v2 approach first
                if hasattr(schema, 'model_fields'):
                    for field_name, field in schema.model_fields.items():
                        # Pydantic v2 field info access
                        field_desc = getattr(field, 'description', None) or "No description"
                        default = f" (default: {field.default})" if hasattr(field, 'default') and field.default != ... else ""
                        field_info.append(f"  - {field_name}: {field_desc}{default}")
                # Fallback to Pydantic v1 approach
                elif hasattr(schema, '__fields__'):
                    for field_name, field in schema.__fields__.items():
                        # Pydantic v1 field info access  
                        field_desc = getattr(field.field_info, 'description', None) or "No description"
                        default = f" (default: {field.default})" if hasattr(field, 'default') and field.default != ... else ""
                        field_info.append(f"  - {field_name}: {field_desc}{default}")
                        
            except AttributeError:
                # If field access fails, just use the schema class name
                field_info = [f"  - Parameters defined in {schema.__name__}"]
            
            if field_info:
                args_info = f"\nParameters:\n" + "\n".join(field_info)
        
        tool_desc = f"- **{name}**: {short_desc}{args_info}"
        descriptions.append(tool_desc)
    
    return "\n\n".join(descriptions)


def create_elasticsearch_tools() -> List[BaseTool]:
    """Create LangChain tools from the tool registry."""
    tools = []
    
    for tool_info in TOOL_REGISTRY:
        # Use the full detailed description instead of truncating
        description = f"{tool_info['short_description']}.\n\n{tool_info['detailed_description']}"
        
        tool = Tool(
            name=tool_info["name"],
            description=description,
            func=tool_info["function"],
            args_schema=tool_info["args_schema"]
        )
        tools.append(tool)
    
    return tools


# Convenience functions
def get_elasticsearch_tools() -> List[BaseTool]:
    """Get all Elasticsearch tools for the research agent."""
    return create_elasticsearch_tools()


def get_available_tools_summary() -> Dict[str, Any]:
    """Get summary of all available tools for system status."""
    return {
        "total_tools": len(TOOL_REGISTRY),
        "tools": [
            {
                "name": tool["name"],
                "description": tool["short_description"],
                "has_parameters": tool["args_schema"] is not None
            }
            for tool in TOOL_REGISTRY
        ]
    }
    """Get summary of all available tools for system status."""
    return {
        "total_tools": len(TOOL_REGISTRY),
        "tools": [
            {
                "name": tool["name"],
                "description": tool["short_description"],
                "has_parameters": tool["args_schema"] is not None
            }
            for tool in TOOL_REGISTRY
        ]
    }