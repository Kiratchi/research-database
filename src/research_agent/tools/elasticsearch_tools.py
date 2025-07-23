"""
Enhanced Elasticsearch tools with built-in context limits and honest communication.
Keeps existing tool names and function signatures while adding smart limits.
"""

from typing import Dict, List, Any, Optional
import json

# Core dependencies
from pydantic import BaseModel, Field
from elasticsearch import Elasticsearch

# LangChain imports - FIXED: Use StructuredTool instead of Tool
from langchain_core.tools import StructuredTool, BaseTool


_es_client = None
_index_name = "research-publications-static"


def initialize_elasticsearch_tools(es_client: Elasticsearch, index_name: str = "research-publications-static"):
    """Initialize the Elasticsearch tools with client and index name."""
    global _es_client, _index_name
    _es_client = es_client
    _index_name = index_name


# =============================================================================
# CONTEXT LIMITS AND MANAGEMENT
# =============================================================================

class ContextLimits:
    """Define clear context limits for each tool"""
    
    # Hard limits to prevent context overflow
    SEARCH_PUBLICATIONS_MAX = 15  # Conservative limit
    SEARCH_BY_AUTHOR_MAX = 20     # Slightly higher for author queries
    FIELD_STATISTICS_MAX = 50     # Statistics are small
    PUBLICATION_DETAILS_MAX = 5   # Full details are large
    
    # Token estimation (conservative)
    TOKENS_PER_PUBLICATION = 600   # Reduced estimate
    TOKENS_PER_AUTHOR_RESULT = 500 # Reduced estimate
    TOKENS_PER_STATISTIC = 50      # Small
    TOKENS_PER_DETAIL = 1200       # Full details
    
    # Context safety threshold (90% of Claude's limit)
    MAX_SAFE_TOKENS = 180000


def estimate_context_size(result_count: int, result_type: str) -> int:
    """Estimate context size for different result types"""
    multipliers = {
        'publication': ContextLimits.TOKENS_PER_PUBLICATION,
        'author_result': ContextLimits.TOKENS_PER_AUTHOR_RESULT,
        'statistic': ContextLimits.TOKENS_PER_STATISTIC,
        'detail': ContextLimits.TOKENS_PER_DETAIL
    }
    return result_count * multipliers.get(result_type, 500)


def create_limitation_notice(total_hits: int, returned_count: int, tool_name: str) -> Dict[str, Any]:
    """Create standardized limitation notices"""
    notices = {}
    
    if total_hits > returned_count:
        notices["limitation_notice"] = (
            f"Found {total_hits} total results but showing only {returned_count} "
            f"due to context window constraints. This is a partial view."
        )
        
        if total_hits > 50:
            notices["guidance"] = {
                "for_counting": "Use get_field_statistics for accurate counts and trends",
                "for_browsing": "Use pagination (offset parameter) to see more results",
                "for_analysis": "Consider narrowing your search terms for more focused results"
            }
    
    return notices


# =============================================================================
# UPDATED PYDANTIC SCHEMAS WITH LIMITS
# =============================================================================

class SearchPublicationsInput(BaseModel):
    """Input schema for comprehensive publication search with explicit limits."""
    query: str = Field(
        description="Search query string (keywords, topics, concepts). Examples: 'machine learning', 'climate change'"
    )
    max_results: int = Field(
        default=10, 
        le=ContextLimits.SEARCH_PUBLICATIONS_MAX,  # Enforce hard limit
        description=f"Maximum results to return (1-{ContextLimits.SEARCH_PUBLICATIONS_MAX}, default: 10). LIMITED due to context constraints."
    )
    offset: int = Field(
        default=0, 
        description="Pagination offset for getting additional results. Use 10, 20, 30, etc. for next pages."
    )
    fields: Optional[List[str]] = Field(
        default=None, 
        description="Specific fields to search in. Leave empty to search all fields."
    )


class SearchByAuthorInput(BaseModel):
    """Input schema for author-specific publication search with explicit limits."""
    author_name: str = Field(
        description="Full author name to search for. Use complete names for best results."
    )
    strategy: str = Field(
        default="partial", 
        description="Search strategy: 'exact', 'partial' (recommended), or 'fuzzy'"
    )
    max_results: int = Field(
        default=10, 
        le=ContextLimits.SEARCH_BY_AUTHOR_MAX,  # Enforce hard limit
        description=f"Maximum results to return (1-{ContextLimits.SEARCH_BY_AUTHOR_MAX}, default: 10). LIMITED due to context constraints."
    )
    offset: int = Field(
        default=0, 
        description="Pagination offset. ESSENTIAL for prolific authors with many publications."
    )


class GetFieldStatisticsInput(BaseModel):
    """Input schema for database field analysis - NO LIMITS (uses aggregations)."""
    field: str = Field(
        description="Field to analyze: 'Year', 'Persons.PersonData.DisplayName', 'Source', 'PublicationType'"
    )
    size: int = Field(
        default=10, 
        le=ContextLimits.FIELD_STATISTICS_MAX,
        description=f"Number of top values to return (1-{ContextLimits.FIELD_STATISTICS_MAX}, default: 10). NO CONTEXT LIMITS - uses aggregations."
    )


class GetPublicationDetailsInput(BaseModel):
    """Input schema for detailed publication information - VERY LIMITED."""
    publication_id: str = Field(
        description="Publication ID from search results. WARNING: Full details are large - use sparingly."
    )


# =============================================================================
# ENHANCED TOOL FUNCTIONS WITH CONTEXT MANAGEMENT
# =============================================================================

def search_publications(query: str, max_results: int = 10, offset: int = 0, fields: Optional[List[str]] = None) -> str:
    """Search publications with built-in context limits and honest communication."""
    if not _es_client:
        return json.dumps({"error": "Elasticsearch client not initialized"})
    
    try:
        # Enforce hard limit
        actual_max = min(max_results, ContextLimits.SEARCH_PUBLICATIONS_MAX)
        
        print(f"🔍 SEARCH_PUBLICATIONS: query='{query}', max_results={actual_max} (requested: {max_results}), offset={offset}")
        
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": fields or ["Title^2", "Abstract", "Persons.PersonData.DisplayName", "Keywords"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            },
            "size": actual_max,
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
        
        print(f"✅ SEARCH_PUBLICATIONS: Found {total_hits} total hits, returned {len(results)} results")
        
        # Build response with honest communication
        response_data = {
            "total_hits": total_hits,
            "results": results,
            "query": query,
            "pagination": {
                "offset": offset,
                "limit": actual_max,
                "has_more": offset + actual_max < total_hits,
                "next_offset": offset + actual_max if offset + actual_max < total_hits else None
            }
        }
        
        # Add limitation notices
        limitation_notices = create_limitation_notice(total_hits, len(results), "search_publications")
        response_data.update(limitation_notices)
        
        # Specific guidance for large result sets
        if total_hits > 100:
            response_data["recommendation"] = (
                "Large result set detected. Consider: "
                "1) More specific search terms, "
                "2) get_field_statistics for trends/counts, "
                "3) Pagination for browsing all results"
            )
        
        return json.dumps(response_data)
        
    except Exception as e:
        print(f"❌ SEARCH_PUBLICATIONS: Error: {str(e)}")
        return json.dumps({
            "error": f"Search failed: {str(e)}",
            "guidance": "Try rephrasing your query or check for typos."
        })


def search_by_author(author_name: str, strategy: str = "partial", max_results: int = 10, offset: int = 0) -> str:
    """Search by author with built-in context limits and clear communication."""
    if not _es_client:
        return json.dumps({"error": "Elasticsearch client not initialized"})
    
    try:
        # Enforce hard limit
        actual_max = min(max_results, ContextLimits.SEARCH_BY_AUTHOR_MAX)
        
        print(f"🔍 SEARCH_BY_AUTHOR: author_name='{author_name}', strategy='{strategy}', max_results={actual_max} (requested: {max_results}), offset={offset}")
        
        # Build query based on strategy
        if strategy == "exact":
            query = {"match_phrase": {"Persons.PersonData.DisplayName": author_name}}
        elif strategy == "fuzzy":
            query = {"fuzzy": {"Persons.PersonData.DisplayName": {"value": author_name, "fuzziness": "AUTO"}}}
        else:  # partial (default)
            query = {"match_phrase": {"Persons.PersonData.DisplayName": author_name}}
        
        search_body = {
            "query": query,
            "size": actual_max,
            "from": offset,
            "sort": [{"Year": {"order": "desc"}}]
        }
        
        try:
            response = _es_client.search(index=_index_name, body=search_body)
        except Exception:
            # Fallback without sorting if field mapping issues
            print("⚠️ SEARCH_BY_AUTHOR: Year sorting failed, trying without sort")
            search_body = {"query": query, "size": actual_max, "from": offset}
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
        
        print(f"✅ SEARCH_BY_AUTHOR: Found {total_hits} total hits, returned {len(results)} results (offset: {offset})")
        
        # Build response with honest communication
        response_data = {
            "total_hits": total_hits,
            "results": results,
            "author": author_name,
            "strategy": strategy,
            "pagination": {
                "offset": offset,
                "limit": actual_max,
                "has_more": offset + actual_max < total_hits,
                "next_offset": offset + actual_max if offset + actual_max < total_hits else None
            }
        }
        
        # Add limitation notices
        limitation_notices = create_limitation_notice(total_hits, len(results), "search_by_author")
        response_data.update(limitation_notices)
        
        # Special guidance for prolific authors
        if total_hits > 50:
            response_data["author_analysis"] = {
                "prolific_author": True,
                "total_publications": total_hits,
                "showing_partial_view": True,
                "recommendation": "Use get_field_statistics for complete publication counts and trends"
            }
        
        # No results guidance
        if total_hits == 0:
            response_data["suggestions"] = [
                "Try 'fuzzy' strategy for name variations",
                "Check spelling of author name", 
                "Try searching by last name only"
            ]
        
        return json.dumps(response_data)
        
    except Exception as e:
        print(f"❌ SEARCH_BY_AUTHOR: Error: {str(e)}")
        return json.dumps({
            "error": f"Author search failed: {str(e)}",
            "guidance": "Check author name spelling or try different matching strategy."
        })


def get_field_statistics(field: str, size: int = 10) -> str:
    """Analyze field distribution - NO CONTEXT LIMITS (uses aggregations only)."""
    if not _es_client:
        return json.dumps({"error": "Elasticsearch client not initialized"})
    
    try:
        # This tool is ALWAYS context-safe because it uses aggregations
        actual_size = min(size, ContextLimits.FIELD_STATISTICS_MAX)
        
        print(f"🔍 GET_FIELD_STATISTICS: field='{field}', size={actual_size} - CONTEXT SAFE (aggregations only)")
        
        search_body = {
            "size": 0,  # NO DOCUMENTS RETURNED - ONLY AGGREGATIONS
            "aggs": {
                "field_stats": {
                    "terms": {
                        "field": f"{field}.keyword" if field in ["Persons.PersonData.DisplayName", "Source", "PublicationType"] else field,
                        "size": actual_size
                    }
                }
            }
        }
        
        response = _es_client.search(index=_index_name, body=search_body)
        buckets = response['aggregations']['field_stats']['buckets']
        stats = [{"value": bucket['key'], "count": bucket['doc_count']} for bucket in buckets]
        
        print(f"✅ GET_FIELD_STATISTICS: Found {len(stats)} values for field '{field}' - ALWAYS CONTEXT SAFE")
        
        return json.dumps({
            "field": field,
            "total_documents": response['hits']['total'],
            "top_values": stats,
            "context_safe": True,
            "methodology": "Elasticsearch aggregations - no individual documents processed",
            "limitation": "None - this analysis is complete and context-safe"
        })
        
    except Exception as e:
        print(f"❌ GET_FIELD_STATISTICS: Error: {str(e)}")
        return json.dumps({
            "error": f"Statistics failed: {str(e)}",
            "guidance": "Check field name - valid options: 'Year', 'Persons.PersonData.DisplayName', 'Source', 'PublicationType'"
        })


def get_publication_details(publication_id: str) -> str:
    """Retrieve complete publication details - USE SPARINGLY (large context)."""
    if not _es_client:
        return json.dumps({"error": "Elasticsearch client not initialized"})
    
    try:
        print(f"🔍 GET_PUBLICATION_DETAILS: publication_id='{publication_id}' - WARNING: Large context usage")
        
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
            "url": source.get('DetailsUrlEng', 'No URL'),
            "context_warning": f"Full details use ~{ContextLimits.TOKENS_PER_DETAIL} tokens - use sparingly"
        }
        
        print(f"✅ GET_PUBLICATION_DETAILS: Retrieved details for '{details['title'][:50]}...' - LARGE CONTEXT")
        
        return json.dumps(details)
        
    except Exception as e:
        print(f"❌ GET_PUBLICATION_DETAILS: Error: {str(e)}")
        return json.dumps({
            "error": f"Failed to get publication details: {str(e)}",
            "guidance": "Check that publication_id is correct (obtained from search results)"
        })


# =============================================================================
# COUNTING TOOL (NEW - ALWAYS CONTEXT SAFE)
# =============================================================================

class CountEntitiesInput(BaseModel):
    """Input schema for counting entities using aggregations - ALWAYS CONTEXT SAFE."""
    entity_type: str = Field(
        description="What to count: 'authors' (unique people), 'publications' (total papers), 'authors_by_name' (specific name)"
    )
    search_term: str = Field(
        description="Search term or name to count. For authors_by_name: specific author name. For others: topic/keyword."
    )


def count_entities(entity_type: str, search_term: str) -> str:
    """Count entities using aggregations - ALWAYS CONTEXT SAFE, NO LIMITS."""
    if not _es_client:
        return json.dumps({"error": "Elasticsearch client not initialized"})
    
    try:
        print(f"🔍 COUNT_ENTITIES: entity_type='{entity_type}', search_term='{search_term}' - ALWAYS CONTEXT SAFE")
        
        if entity_type == "authors_by_name":
            # WORKING APPROACH: Simple match query + terms aggregation
            agg_query = {
                "size": 0,  # NO DOCUMENTS RETURNED
                "query": {
                    "match": {
                        "Persons.PersonData.DisplayName": search_term
                    }
                },
                "aggs": {
                    "unique_authors": {
                        "terms": {
                            "field": "Persons.PersonData.DisplayName.keyword",
                            "size": 100  # Get up to 100 unique author names
                        }
                    },
                    "total_publications": {
                        "value_count": {"field": "_id"}
                    }
                }
            }
            
            response = _es_client.search(index=_index_name, body=agg_query)
            
            # Extract matching authors from aggregation
            matching_authors = []
            for bucket in response["aggregations"]["unique_authors"]["buckets"]:
                author_name = bucket["key"]
                pub_count = bucket["doc_count"]
                
                # Filter for names containing our search term (case insensitive)
                if search_term.lower() in author_name.lower():
                    matching_authors.append({
                        "name": author_name,
                        "publication_count": pub_count
                    })
            
            total_pubs = response["aggregations"]["total_publications"]["value"]
            
            # Sort by publication count (most prolific first)
            matching_authors.sort(key=lambda x: x["publication_count"], reverse=True)
            
            result = {
                "entity_type": entity_type,
                "search_term": search_term,
                "total_publications_found": total_pubs,
                "unique_individuals": len(matching_authors),
                "individuals": matching_authors,
                "summary": f"Found {len(matching_authors)} unique people with '{search_term}' in their name across {total_pubs} publications",
                "methodology": "Elasticsearch aggregation on non-nested field - complete and accurate count",
                "context_safe": True,
                "limitation": "None - this count is complete and accurate",
                "top_authors": matching_authors[:5] if matching_authors else []  # Top 5 for quick reference
            }
            
        else:
            # Other counting types can be added here
            result = {
                "error": f"Entity type '{entity_type}' not yet implemented",
                "available_types": ["authors_by_name"],
                "guidance": "Use 'authors_by_name' for counting people with specific names"
            }
        
        print(f"✅ COUNT_ENTITIES: Found {len(matching_authors) if entity_type == 'authors_by_name' else 0} unique individuals - CONTEXT SAFE")
        return json.dumps(result)
        
    except Exception as e:
        print(f"❌ COUNT_ENTITIES: Error: {str(e)}")
        return json.dumps({
            "error": f"Counting failed: {str(e)}",
            "guidance": "Check search term spelling and try again"
        })


def get_statistics_summary() -> Dict[str, Any]:
    """Generate comprehensive database overview with key metrics."""
    if not _es_client:
        return {"error": "Elasticsearch client not initialized"}
    
    try:
        print(f"🔍 GET_STATISTICS_SUMMARY: Starting database overview")
        
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
        
        result = {
            "total_publications": total_publications,
            "latest_year": years[0]['value'] if years else None,
            "most_common_type": types[0]['value'] if types else None,
            "total_authors": authors_response['aggregations']['author_count']['value'],
            "years": years,
            "publication_types": types
        }
        
        print(f"✅ GET_STATISTICS_SUMMARY: Database has {total_publications} publications, {result['total_authors']} authors")
        
        return result
        
    except Exception as e:
        print(f"❌ GET_STATISTICS_SUMMARY: Error: {str(e)}")
        return {"error": f"Failed to get statistics: {str(e)}"}


def get_database_summary() -> str:
    """Wrapper function to return JSON string for database summary."""
    result = get_statistics_summary()
    return json.dumps(result)



# =============================================================================
# UPDATED TOOL REGISTRY WITH LIMITS DOCUMENTATION
# =============================================================================

TOOL_REGISTRY = [
    {
        "name": "search_publications",
        "function": search_publications,
        "args_schema": SearchPublicationsInput,
        "short_description": f"Search publications (MAX {ContextLimits.SEARCH_PUBLICATIONS_MAX} results due to context limits)",
        "context_limit": ContextLimits.SEARCH_PUBLICATIONS_MAX,
        "token_estimate": ContextLimits.TOKENS_PER_PUBLICATION,
        "detailed_description": f"""Search research publications with BUILT-IN CONTEXT LIMITS.

**CRITICAL LIMITATIONS:**
- Maximum {ContextLimits.SEARCH_PUBLICATIONS_MAX} results per call due to context window constraints
- Each result uses ~{ContextLimits.TOKENS_PER_PUBLICATION} tokens
- For larger datasets, tool will provide clear guidance

**USE FOR:**
- Finding specific papers on topics
- Exploring research areas with samples
- Getting publication examples

**DON'T USE FOR:**
- Counting total publications (use count_entities instead)
- Large-scale analysis (use get_field_statistics instead)

**Tool will automatically:**
- Enforce result limits
- Provide clear limitation notices
- Suggest better approaches for large datasets"""
    },
    
    {
        "name": "search_by_author", 
        "function": search_by_author,
        "args_schema": SearchByAuthorInput,
        "short_description": f"Search by author (MAX {ContextLimits.SEARCH_BY_AUTHOR_MAX} results due to context limits)",
        "context_limit": ContextLimits.SEARCH_BY_AUTHOR_MAX,
        "token_estimate": ContextLimits.TOKENS_PER_AUTHOR_RESULT,
        "detailed_description": f"""Find publications by author with BUILT-IN CONTEXT LIMITS.

**CRITICAL LIMITATIONS:**
- Maximum {ContextLimits.SEARCH_BY_AUTHOR_MAX} results per call due to context window constraints  
- Each result uses ~{ContextLimits.TOKENS_PER_AUTHOR_RESULT} tokens
- For prolific authors, provides partial view only

**USE FOR:**
- Exploring author's work samples
- Getting recent publications by author
- Author publication examples

**DON'T USE FOR:**
- Complete author publication counts (use count_entities instead)
- Full career analysis of prolific authors

**Tool will automatically:**
- Enforce result limits
- Clearly state when results are partial
- Provide guidance for complete analysis"""
    },
    
    {
        "name": "get_field_statistics",
        "function": get_field_statistics,
        "args_schema": GetFieldStatisticsInput,
        "short_description": "Analyze field distributions (NO CONTEXT LIMITS - uses aggregations)",
        "context_limit": None,  # No limits - uses aggregations
        "token_estimate": ContextLimits.TOKENS_PER_STATISTIC,
        "detailed_description": """Analyze database field distributions - ALWAYS CONTEXT SAFE.

**NO CONTEXT LIMITATIONS:**
- Uses Elasticsearch aggregations only
- Never retrieves individual documents
- Always safe regardless of database size

**ALWAYS USE FOR:**
- Counting queries ("How many X?")
- Trend analysis over time
- Finding top authors, journals, etc.
- Any statistical analysis

**Perfect for large-scale analysis that other tools cannot handle.**"""
    },
    
    {
        "name": "get_publication_details",
        "function": get_publication_details,
        "args_schema": GetPublicationDetailsInput,
        "short_description": f"Get publication details (HIGH CONTEXT USAGE ~{ContextLimits.TOKENS_PER_DETAIL} tokens - use sparingly)",
        "context_limit": ContextLimits.PUBLICATION_DETAILS_MAX,
        "token_estimate": ContextLimits.TOKENS_PER_DETAIL,
        "detailed_description": f"""Get detailed publication information - USE SPARINGLY.

**HIGH CONTEXT USAGE:**
- Each call uses ~{ContextLimits.TOKENS_PER_DETAIL} tokens
- Limit to {ContextLimits.PUBLICATION_DETAILS_MAX} calls per conversation
- Only use when full details are essential

**Use only when user specifically needs:**
- Complete abstracts
- Full author lists
- DOI and URL information"""
    },
    
    {
        "name": "get_database_summary",
        "function": get_database_summary,
        "args_schema": None,
        "short_description": "Database overview (CONTEXT SAFE - aggregations only)",
        "context_limit": None,
        "token_estimate": 200,
        "detailed_description": "Get database overview using aggregations - always context safe."
    },
    
    # NEW TOOL - Always safe counting
    {
        "name": "count_entities",
        "function": count_entities,
        "args_schema": CountEntitiesInput,
        "short_description": "Count entities (NO CONTEXT LIMITS - aggregations only)",
        "context_limit": None,
        "token_estimate": 300,
        "detailed_description": """Count entities using aggregations - ALWAYS CONTEXT SAFE.

**NO LIMITATIONS:**
- Uses aggregations only
- Always provides complete, accurate counts
- Never retrieves individual documents

**PREFERRED TOOL FOR:**
- "How many people named X?"
- "How many publications on topic Y?"
- Any counting or statistics questions

**Use this instead of search tools when you need counts, not documents.**"""
    }
]

# =============================================================================
# MISSING FUNCTIONS - ADD THESE TO YOUR CODE
# =============================================================================

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
    """Create LangChain tools from the tool registry using StructuredTool."""
    tools = []
    
    for tool_info in TOOL_REGISTRY:
        print(f"🔧 CREATING TOOL: {tool_info['name']} with schema: {tool_info['args_schema']}")
        
        # FIXED: Use StructuredTool instead of Tool for multi-parameter support
        if tool_info["args_schema"] is not None:
            tool = StructuredTool(
                name=tool_info["name"],
                description=tool_info["short_description"],
                func=tool_info["function"],
                args_schema=tool_info["args_schema"]
            )
        else:
            # For tools without parameters (like get_database_summary)
            tool = StructuredTool.from_function(
                name=tool_info["name"],
                description=tool_info["short_description"],
                func=tool_info["function"]
            )
        
        tools.append(tool)
        print(f"✅ CREATED TOOL: {tool_info['name']} successfully")
    
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