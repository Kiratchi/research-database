"""
Search Session Management for Elasticsearch

Handles stateful search sessions with result caching, pagination,
and summarization for agent-friendly database interactions.
"""

import uuid
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json


class SearchSession:
    """
    Manages a single search session with cached results and metadata.
    """
    
    def __init__(self, es_client, index_name: str, session_id: Optional[str] = None):
        """
        Initialize a search session.
        
        Args:
            es_client: Elasticsearch client instance
            index_name: Name of the index to search
            session_id: Optional session ID (generates one if not provided)
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.es = es_client
        self.index_name = index_name
        
        # Session state
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.query_params = {}
        self.total_results = 0
        self.max_score = 0.0
        
        # Result cache
        self.cached_results = []
        self.cached_pages = set()  # Track which pages we've fetched
        self.page_size = 20  # Default page size
        
        # Aggregation cache
        self.aggregations = {}
        
    def execute_search(self, query: Optional[str] = None, 
                      filters: Optional[Dict] = None,
                      size: int = 100,
                      store_all: bool = False) -> Dict[str, Any]:
        """
        Execute a search and cache results.
        
        Args:
            query: Free text search query
            filters: Dictionary of filters
            size: Initial number of results to fetch
            store_all: If True, fetches all results (use with caution!)
            
        Returns:
            Summary of search results
        """
        self.last_accessed = datetime.now()
        self.query_params = {
            "query": query,
            "filters": filters,
            "index": self.index_name
        }
        
        # Build Elasticsearch query
        es_query = self._build_query(query, filters)
        
        # Execute search with aggregations
        body = {
            "query": es_query,
            "size": size if not store_all else 0,
            "aggs": self._get_standard_aggregations(),
            "track_total_hits": True  # Ensure we get accurate total count
        }
        
        response = self.es.search(index=self.index_name, body=body)
        
        # Store results
        self.total_results = response["hits"]["total"]
        if isinstance(self.total_results, dict):  # ES 7.x format
            self.total_results = self.total_results["value"]
            
        self.max_score = response["hits"].get("max_score", 0.0) or 0.0
        self.aggregations = response.get("aggregations", {})
        
        # Cache initial results
        self.cached_results = response["hits"]["hits"]
        if self.cached_results:
            self.cached_pages.add(0)  # Mark first page as cached
        
        # If store_all requested and reasonable size, fetch all
        if store_all and self.total_results < 1000:
            self._fetch_all_results(es_query)
        
        return self.get_summary()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the search results suitable for an agent.
        
        Returns:
            Dictionary with metadata and samples
        """
        summary = {
            "session_id": self.session_id,
            "total_results": self.total_results,
            "max_score": self.max_score,
            "query_params": self.query_params,
            "created_at": self.created_at.isoformat(),
            "cached_results": len(self.cached_results),
            "sample_results": self._get_sample_results(5),
            "aggregations": self._format_aggregations()
        }
        
        return summary
    
    def get_page(self, page_number: int = 0, page_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Get a specific page of results.
        
        Args:
            page_number: Zero-based page number
            page_size: Number of results per page (uses default if not specified)
            
        Returns:
            Dictionary with results and pagination info
        """
        self.last_accessed = datetime.now()
        
        if page_size:
            self.page_size = page_size
            
        start = page_number * self.page_size
        end = start + self.page_size
        
        # Check if we have this page cached
        if not self._is_page_cached(page_number):
            self._fetch_page(page_number)
        
        # Return the requested slice
        results = self.cached_results[start:end]
        
        return {
            "page": page_number,
            "page_size": self.page_size,
            "total_pages": (self.total_results + self.page_size - 1) // self.page_size,
            "results": results,
            "has_next": end < self.total_results,
            "has_previous": page_number > 0
        }
    
    def refine_search(self, additional_filters: Dict) -> Dict[str, Any]:
        """
        Refine the current search with additional filters.
        
        Args:
            additional_filters: New filters to add to existing query
            
        Returns:
            New summary after refinement
        """
        # Merge filters
        current_filters = self.query_params.get("filters", {})
        new_filters = {**current_filters, **additional_filters}
        
        # Execute new search
        return self.execute_search(
            query=self.query_params.get("query"),
            filters=new_filters
        )
    
    def get_field_values(self, field: str, size: int = 20) -> List[Dict[str, Any]]:
        """
        Get aggregated values for a specific field from cached results.
        
        Args:
            field: Field name to aggregate
            size: Number of top values to return
            
        Returns:
            List of {value, count} dictionaries
        """
        # If we have it in aggregations, use that
        if field in self.aggregations:
            return self._format_aggregation_buckets(self.aggregations[field])
        
        # Otherwise, aggregate from cached results
        value_counts = defaultdict(int)
        
        for hit in self.cached_results:
            source = hit["_source"]
            value = self._get_nested_value(source, field)
            
            if value:
                if isinstance(value, list):
                    for v in value:
                        value_counts[str(v)] += 1
                else:
                    value_counts[str(value)] += 1
        
        # Sort by count and return top N
        sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"value": v, "count": c} for v, c in sorted_values[:size]]
    
    # Private helper methods
    
    def _build_query(self, query: Optional[str], filters: Optional[Dict]) -> Dict:
        """Build Elasticsearch query from parameters."""
        must_clauses = []
        
        if query:
            must_clauses.append({
                "multi_match": {
                    "query": query,
                    "fields": ["Title^3", "Abstract^2", "Keywords.Value"],
                    "type": "best_fields"
                }
            })
        
        if filters:
            author_strategy = filters.get("author_strategy", "partial")
            
            for field, value in filters.items():
                if field == "year_range" and isinstance(value, dict):
                    must_clauses.append({"range": {"Year": value}})
                elif field == "author":
                    # Use strategy-based author query
                    author_query = self._build_author_query(value, author_strategy)
                    must_clauses.append(author_query)
                elif field == "organization":
                    must_clauses.append({
                        "match": {"Organizations.OrganizationData.DisplayNameEng": value}
                    })
                elif field == "author_strategy":
                    # Skip this - it's handled above
                    continue
                else:
                    must_clauses.append({"term": {field: value}})
        
        if must_clauses:
            return {"bool": {"must": must_clauses}}
        else:
            return {"match_all": {}}
    
    def _build_author_query(self, author_name: str, strategy: str) -> Dict:
        """
        Build author query based on strategy.
        
        Args:
            author_name: The author name to search for
            strategy: Query strategy ("exact", "partial", "fuzzy")
            
        Returns:
            Elasticsearch query dict
        """
        if strategy == "exact":
            # Use match_phrase for exact full name matching
            return {
                "match_phrase": {
                    "Persons.PersonData.DisplayName": author_name
                }
            }
        elif strategy == "partial":
            # Use match for partial name matching (good for surnames)
            return {
                "match": {
                    "Persons.PersonData.DisplayName": author_name
                }
            }
        elif strategy == "fuzzy":
            # Use fuzzy matching for typos/variations
            return {
                "fuzzy": {
                    "Persons.PersonData.DisplayName": {
                        "value": author_name,
                        "fuzziness": "AUTO"
                    }
                }
            }
        else:
            # Default to the original match query
            return {
                "match": {
                    "Persons.PersonData.DisplayName": author_name
                }
            }
    
    def _get_standard_aggregations(self) -> Dict:
        """Get standard aggregations for publications."""
        return {
            "years": {
                "terms": {
                    "field": "Year",
                    "size": 50,
                    "order": {"_key": "desc"}
                }
            },
            "types": {
                "terms": {
                    "field": "PublicationType.NameEng.keyword",
                    "size": 20
                }
            },
            "top_authors": {
                "terms": {
                    "field": "Persons.PersonData.DisplayName.keyword",
                    "size": 20
                }
            }
        }
    
    def _get_sample_results(self, count: int) -> List[Dict]:
        """Get simplified sample results."""
        samples = []
        
        for hit in self.cached_results[:count]:
            source = hit["_source"]
            samples.append({
                "title": source.get("Title", "No title"),
                "year": source.get("Year"),
                "score": hit.get("_score", 0),
                "authors": self._extract_author_names(source),
                "abstract": source.get("Abstract", "")[:200] + "..." if source.get("Abstract") else None
            })
        
        return samples
    
    def _extract_author_names(self, source: Dict) -> List[str]:
        """Extract author names from document."""
        authors = []
        for person in source.get("Persons", []):
            if "PersonData" in person and "DisplayName" in person["PersonData"]:
                authors.append(person["PersonData"]["DisplayName"])
        return authors[:5]  # Limit to first 5 authors
    
    def _format_aggregations(self) -> Dict:
        """Format aggregations for summary."""
        formatted = {}
        
        for agg_name, agg_data in self.aggregations.items():
            if "buckets" in agg_data:
                formatted[agg_name] = [
                    {"value": b["key"], "count": b["doc_count"]} 
                    for b in agg_data["buckets"][:10]
                ]
        
        return formatted
    
    def _is_page_cached(self, page_number: int) -> bool:
        """Check if a page is already cached."""
        start = page_number * self.page_size
        end = start + self.page_size
        return start < len(self.cached_results) and end <= len(self.cached_results)
    
    def _fetch_page(self, page_number: int):
        """Fetch a specific page from Elasticsearch."""
        start = page_number * self.page_size
        
        # Rebuild the query
        es_query = self._build_query(
            self.query_params.get("query"),
            self.query_params.get("filters")
        )
        
        body = {
            "query": es_query,
            "from": start,
            "size": self.page_size,
            "sort": [{"Year": {"order": "desc"}}, "_score"]
        }
        
        response = self.es.search(index=self.index_name, body=body)
        
        # Extend cache if needed
        new_results = response["hits"]["hits"]
        needed_size = start + len(new_results)
        
        if len(self.cached_results) < needed_size:
            self.cached_results.extend([None] * (needed_size - len(self.cached_results)))
        
        # Insert results at correct position
        for i, result in enumerate(new_results):
            self.cached_results[start + i] = result
            
        self.cached_pages.add(page_number)
    
    def _fetch_all_results(self, query: Dict):
        """Fetch all results using scroll API (use with caution!)."""
        # This is a simplified version - in production you'd use scroll API
        # For now, just fetch up to 1000 results
        body = {
            "query": query,
            "size": min(self.total_results, 1000),
            "sort": [{"Year": {"order": "desc"}}, "_score"]
        }
        
        response = self.es.search(index=self.index_name, body=body)
        self.cached_results = response["hits"]["hits"]
    
    def _get_nested_value(self, obj: Dict, path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = path.split(".")
        value = obj
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
                
        return value
    
    def _format_aggregation_buckets(self, agg_data: Dict) -> List[Dict]:
        """Format aggregation buckets."""
        if "buckets" in agg_data:
            return [
                {"value": b["key"], "count": b["doc_count"]} 
                for b in agg_data["buckets"]
            ]
        return []
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session has expired."""
        age = datetime.now() - self.last_accessed
        return age > timedelta(minutes=timeout_minutes)