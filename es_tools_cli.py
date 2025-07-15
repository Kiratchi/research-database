"""
Elasticsearch tools for exploring Chalmers research publications.
Designed for manual exploration and agent integration.
"""

import os
from typing import Optional, Dict, List, Any
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Initialize Elasticsearch client (6.x version)
es = Elasticsearch(
    hosts=[os.getenv('ES_HOST')],
    http_auth=(os.getenv('ES_USER'), os.getenv('ES_PASS')),  # Note: http_auth not basic_auth in 6.x
    use_ssl=True,
    verify_certs=True
)

# Constants
PUBLICATIONS_INDEX = "research-publications-static"


def search_publications(
    query: Optional[str] = None,
    filters: Optional[Dict] = None,
    size: int = 10,
    from_: int = 0,
    fields: Optional[List[str]] = None,
    sort: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Search publications with flexible query and filter options.
    
    Args:
        query: Free text search across all text fields
        filters: Dictionary of exact match filters
        size: Number of results to return (default 10, max 100)
        from_: Offset for pagination
        fields: List of fields to return (None returns all)
        sort: Sort criteria, e.g., {"Year": "desc"}
    
    Returns:
        Dict with hits, total count, and aggregations
    """
    # Build the query
    es_query = {"bool": {"must": []}}
    
    # Add free text search if provided
    if query:
        es_query["bool"]["must"].append({
            "multi_match": {
                "query": query,
                "fields": [
                    "Title^3",  # Title gets 3x weight
                    "Abstract^2",  # Abstract gets 2x weight
                    "Keywords.Value",
                    "Persons.PersonData.DisplayName",
                    "Organizations.OrganizationData.DisplayNameEng"
                ],
                "type": "best_fields"
            }
        })
    
    # Add filters
    if filters:
        for field, value in filters.items():
            if field == "year_range" and isinstance(value, dict):
                # Handle year range
                es_query["bool"]["must"].append({
                    "range": {"Year": value}
                })
            elif field == "authors.name":
                # Handle author name with exact match on keyword field
                es_query["bool"]["must"].append({
                    "nested": {
                        "path": "Persons",
                        "query": {
                            "match_phrase": {
                                "Persons.PersonData.DisplayName": value
                            }
                        }
                    }
                })
            elif field == "organization_id":
                # Handle organization ID
                es_query["bool"]["must"].append({
                    "term": {"AffiliatedIdsChalmers": value}
                })
            elif field == "has_fulltext":
                # Check if publication has fulltext
                es_query["bool"]["must"].append({
                    "exists": {"field": "DataObjects"} if value else
                    {"bool": {"must_not": {"exists": {"field": "DataObjects"}}}}
                })
            elif field == "publication_type":
                # Publication type filter
                es_query["bool"]["must"].append({
                    "term": {"PublicationType.Id": value}
                })
            else:
                # Generic exact match
                es_query["bool"]["must"].append({
                    "term": {field: value}
                })
    
    # If no conditions were added, search all
    if not es_query["bool"]["must"]:
        es_query = {"match_all": {}}
    
    # Build the request body
    body = {"query": es_query, "size": min(size, 100), "from": from_}
    
    # Add source filtering if specified
    if fields:
        body["_source"] = fields
    
    # Add sorting
    if sort:
        body["sort"] = [sort]
    else:
        body["sort"] = [{"Year": {"order": "desc"}}, "_score"]
    
    # Execute search (6.x compatible)
    response = es.search(
        index=PUBLICATIONS_INDEX, 
        doc_type="publication",  # 6.x requires doc_type
        body=body
    )
    
    # Format response (6.x compatible)
    return {
        "total": response["hits"]["total"],  # In 6.x, total is just an int
        "hits": response["hits"]["hits"],
        "max_score": response["hits"]["max_score"],
        "query_used": body  # For debugging
    }


def get_publication_stats(
    filters: Optional[Dict] = None,
    aggregations: List[str] = None
) -> Dict[str, Any]:
    """
    Get aggregated statistics about publications.
    
    Args:
        filters: Optional filters to apply before aggregating
        aggregations: List of fields to aggregate on
                     Default: ["year", "type", "author", "organization"]
    
    Returns:
        Dict with aggregation results
    """
    if aggregations is None:
        aggregations = ["year", "type", "author", "organization"]
    
    # Build base query
    if filters:
        # Reuse filter logic from search_publications
        query = {"bool": {"must": []}}
        # ... (same filter building logic as above)
    else:
        query = {"match_all": {}}
    
    # Build aggregations
    aggs = {}
    
    if "year" in aggregations:
        aggs["publications_by_year"] = {
            "terms": {
                "field": "Year",
                "size": 50,
                "order": {"_key": "desc"}
            }
        }
        # Also add a date histogram for trends
        aggs["publication_trend"] = {
            "date_histogram": {
                "field": "CreatedDate",
                "calendar_interval": "year",
                "min_doc_count": 1
            }
        }
    
    if "type" in aggregations:
        aggs["publication_types"] = {
            "terms": {
                "field": "PublicationType.NameEng.keyword",
                "size": 20
            }
        }
    
    if "author" in aggregations:
        aggs["top_authors"] = {
            "nested": {
                "path": "Persons"
            },
            "aggs": {
                "authors": {
                    "terms": {
                        "field": "Persons.PersonData.DisplayName.keyword",
                        "size": 20
                    }
                }
            }
        }
    
    if "organization" in aggregations:
        aggs["organizations"] = {
            "nested": {
                "path": "Organizations"
            },
            "aggs": {
                "org_names": {
                    "terms": {
                        "field": "Organizations.OrganizationData.DisplayNameEng.keyword",
                        "size": 20
                    }
                }
            }
        }
    
    if "keywords" in aggregations:
        aggs["top_keywords"] = {
            "nested": {
                "path": "Keywords"
            },
            "aggs": {
                "keyword_values": {
                    "terms": {
                        "field": "Keywords.Value.keyword",
                        "size": 30
                    }
                }
            }
        }
    
    # Execute aggregation query (6.x compatible)
    body = {
        "query": query,
        "size": 0,  # Don't return documents, only aggregations
        "aggs": aggs
    }
    
    response = es.search(
        index=PUBLICATIONS_INDEX,
        doc_type="publication",  # 6.x requires doc_type
        body=body
    )
    
    # Format response
    result = {
        "total_matching": response["hits"]["total"],  # In 6.x, total is just an int
        "aggregations": {}
    }
    
    # Flatten nested aggregation results for easier consumption
    for agg_name, agg_data in response["aggregations"].items():
        if "buckets" in agg_data:
            result["aggregations"][agg_name] = agg_data["buckets"]
        elif agg_name in ["top_authors", "organizations", "top_keywords"]:
            # Handle nested aggregations
            inner_key = list(agg_data.keys())[1]  # Skip 'doc_count'
            if inner_key in agg_data and "buckets" in agg_data[inner_key]:
                result["aggregations"][agg_name] = agg_data[inner_key]["buckets"]
    
    return result


# Helper function for common queries
def search_recent_publications(days: int = 30, size: int = 10) -> Dict[str, Any]:
    """
    Convenience function to get recent publications.
    
    Args:
        days: Number of days to look back
        size: Number of results to return
    
    Returns:
        Recent publications
    """
    date_from = (datetime.now() - timedelta(days=days)).isoformat()
    
    return search_publications(
        filters={"CreatedDate": {"gte": date_from}},
        size=size,
        sort={"CreatedDate": "desc"}
    )


# Example usage patterns for learning
if __name__ == "__main__":
    print("=== Example 1: Simple text search ===")
    results = search_publications(query="quantum computing", size=5)
    print(f"Found {results['total']} publications")
    for hit in results['hits']:
        pub = hit['_source']
        print(f"- {pub.get('Title', 'No title')} ({pub.get('Year', 'No year')})")
    
    print("\n=== Example 2: Search by author ===")
    results = search_publications(
        filters={"authors.name": "Christian Fager"},
        size=5
    )
    print(f"Found {results['total']} publications by Christian Fager")
    
    print("\n=== Example 3: Combined search ===")
    results = search_publications(
        query="microwave",
        filters={"year_range": {"gte": 2020, "lte": 2023}},
        size=5
    )
    print(f"Found {results['total']} publications about microwave from 2020-2023")
    
    print("\n=== Example 4: Get statistics ===")
    stats = get_publication_stats(aggregations=["year", "type"])
    print("Publications by year (top 5):")
    for bucket in stats['aggregations'].get('publications_by_year', [])[:5]:
        print(f"  {bucket['key']}: {bucket['doc_count']} publications")