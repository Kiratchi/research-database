"""
Agent-friendly tools for Elasticsearch research publications.

These functions provide a simple interface for LLM agents to search
and explore the publications database with automatic session management.
"""

from typing import Dict, List, Any, Optional
from search_session import SearchSession
from datetime import datetime, timedelta


class SessionManager:
    """
    Manages multiple search sessions for the agent.
    """
    
    def __init__(self, es_client, index_name: str = "research-publications-static"):
        self.es = es_client
        self.index_name = index_name
        self.sessions: Dict[str, SearchSession] = {}
        self.timeout_minutes = 30
    
    def create_session(self, session_id: Optional[str] = None) -> SearchSession:
        """Create a new search session."""
        session = SearchSession(self.es, self.index_name, session_id)
        self.sessions[session.session_id] = session
        self._cleanup_expired_sessions()
        return session
    
    def get_session(self, session_id: str) -> Optional[SearchSession]:
        """Get an existing session by ID."""
        self._cleanup_expired_sessions()
        return self.sessions.get(session_id)
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions to free memory."""
        expired = [
            sid for sid, session in self.sessions.items()
            if session.is_expired(self.timeout_minutes)
        ]
        for sid in expired:
            del self.sessions[sid]


# Global session manager instance
_session_manager = None


def initialize_tools(es_client, index_name: str = "research-publications-static"):
    """
    Initialize the agent tools with an Elasticsearch client.
    
    Args:
        es_client: Elasticsearch client instance
        index_name: Name of the publications index
    """
    global _session_manager
    _session_manager = SessionManager(es_client, index_name)


def search_publications(
    query: Optional[str] = None,
    filters: Optional[Dict] = None,
    size: int = 20
) -> Dict[str, Any]:
    """
    Search for publications and return a summary with session ID.
    
    This is the main entry point for agents to search the database.
    Results are automatically cached in a session for follow-up queries.
    
    Args:
        query: Free text search query (searches title, abstract, keywords)
        filters: Dictionary of filters, e.g.:
                 - {"year": 2023}
                 - {"year_range": {"gte": 2020, "lte": 2023}}
                 - {"author": "Fager"}
                 - {"organization": "Chalmers"}
        size: Initial number of results to retrieve
    
    Returns:
        Dictionary containing:
        - session_id: ID to use for follow-up queries
        - total_results: Total number of matching publications
        - sample_results: First few results with key fields
        - aggregations: Statistics about results (years, types, authors)
        - instructions: How to get more results
    
    Example:
        >>> result = search_publications(query="machine learning", filters={"year_range": {"gte": 2020}})
        >>> print(f"Found {result['total_results']} papers")
        >>> print(f"Session ID: {result['session_id']}")
    """
    if not _session_manager:
        raise RuntimeError("Tools not initialized. Call initialize_tools() first.")
    
    # Create new session and execute search
    session = _session_manager.create_session()
    summary = session.execute_search(query, filters, size)
    
    # Add instructions for the agent
    summary["instructions"] = {
        "get_more": f"Use get_more_results('{session.session_id}', page_number) to see more",
        "refine": f"Use refine_search('{session.session_id}', new_filters) to narrow results",
        "stats": f"Use get_field_statistics('{session.session_id}', 'field_name') for field analysis"
    }
    
    return summary


def get_more_results(session_id: str, page_number: int = 0, page_size: int = 20) -> Dict[str, Any]:
    """
    Get a specific page of results from a previous search.
    
    Args:
        session_id: The session ID from a previous search
        page_number: Which page to retrieve (0-based)
        page_size: Number of results per page
    
    Returns:
        Dictionary containing:
        - results: List of publications for this page
        - page: Current page number
        - total_pages: Total number of pages available
        - has_next: Whether there are more pages
        - has_previous: Whether there are previous pages
    
    Example:
        >>> # Get the second page of results
        >>> page2 = get_more_results("abc123", page_number=1)
        >>> for pub in page2['results']:
        ...     print(pub['_source']['Title'])
    """
    if not _session_manager:
        raise RuntimeError("Tools not initialized. Call initialize_tools() first.")
    
    session = _session_manager.get_session(session_id)
    if not session:
        return {
            "error": f"Session {session_id} not found or expired",
            "suggestion": "Run a new search with search_publications()"
        }
    
    return session.get_page(page_number, page_size)


def refine_search(session_id: str, additional_filters: Dict) -> Dict[str, Any]:
    """
    Refine an existing search by adding more filters.
    
    This is useful when the agent wants to narrow down results
    based on what it learned from the initial search.
    
    Args:
        session_id: The session ID from a previous search
        additional_filters: New filters to add to the existing query
    
    Returns:
        New search summary with refined results
    
    Example:
        >>> # After seeing results from multiple years, narrow to just 2023
        >>> refined = refine_search("abc123", {"year": 2023})
        >>> print(f"Refined to {refined['total_results']} papers from 2023")
    """
    if not _session_manager:
        raise RuntimeError("Tools not initialized. Call initialize_tools() first.")
    
    session = _session_manager.get_session(session_id)
    if not session:
        return {
            "error": f"Session {session_id} not found or expired",
            "suggestion": "Run a new search with search_publications()"
        }
    
    return session.refine_search(additional_filters)


def get_field_statistics(session_id: str, field: str, top_n: int = 20) -> Dict[str, Any]:
    """
    Get statistics about a specific field from the search results.
    
    Useful for understanding the distribution of values in the results.
    
    Args:
        session_id: The session ID from a previous search
        field: Field name to analyze (e.g., "Year", "PublicationType.NameEng")
        top_n: Number of top values to return
    
    Returns:
        Dictionary containing:
        - field: The field name analyzed
        - values: List of {value, count} dictionaries
        - total_unique: Total number of unique values (if available)
    
    Example:
        >>> # See distribution of publication years
        >>> year_stats = get_field_statistics("abc123", "Year")
        >>> for item in year_stats['values']:
        ...     print(f"{item['value']}: {item['count']} publications")
    """
    if not _session_manager:
        raise RuntimeError("Tools not initialized. Call initialize_tools() first.")
    
    session = _session_manager.get_session(session_id)
    if not session:
        return {
            "error": f"Session {session_id} not found or expired",
            "suggestion": "Run a new search with search_publications()"
        }
    
    values = session.get_field_values(field, top_n)
    
    return {
        "field": field,
        "values": values,
        "session_id": session_id,
        "note": f"Showing top {len(values)} values from cached results"
    }


def get_publication_details(session_id: str, position: int) -> Dict[str, Any]:
    """
    Get detailed information about a specific publication from search results.
    
    Args:
        session_id: The session ID from a previous search
        position: Position of the publication in search results (0-based)
    
    Returns:
        Full publication details or error message
    
    Example:
        >>> # Get details of the first result
        >>> details = get_publication_details("abc123", 0)
        >>> print(details['Title'])
        >>> print(f"Authors: {', '.join(details['authors'])}")
    """
    if not _session_manager:
        raise RuntimeError("Tools not initialized. Call initialize_tools() first.")
    
    session = _session_manager.get_session(session_id)
    if not session:
        return {
            "error": f"Session {session_id} not found or expired",
            "suggestion": "Run a new search with search_publications()"
        }
    
    if position >= len(session.cached_results):
        # Try to fetch the page containing this position
        page_number = position // session.page_size
        session.get_page(page_number)
    
    if position < len(session.cached_results) and session.cached_results[position]:
        pub = session.cached_results[position]["_source"]
        
        # Extract key information
        return {
            "position": position,
            "title": pub.get("Title"),
            "year": pub.get("Year"),
            "abstract": pub.get("Abstract"),
            "authors": session._extract_author_names(pub),
            "keywords": [k.get("Value") for k in pub.get("Keywords", [])],
            "publication_type": pub.get("PublicationType", {}).get("NameEng"),
            "identifiers": {
                "doi": pub.get("IdentifierDoi", []),
                "scopus": pub.get("IdentifierScopusId", [])
            },
            "full_record": pub  # Include everything for advanced users
        }
    else:
        return {
            "error": f"Publication at position {position} not found",
            "total_results": session.total_results
        }


def list_active_sessions() -> List[Dict[str, Any]]:
    """
    List all active search sessions.
    
    Useful for agents to see what searches are available.
    
    Returns:
        List of session summaries
    """
    if not _session_manager:
        raise RuntimeError("Tools not initialized. Call initialize_tools() first.")
    
    sessions = []
    for sid, session in _session_manager.sessions.items():
        sessions.append({
            "session_id": sid,
            "created_at": session.created_at.isoformat(),
            "last_accessed": session.last_accessed.isoformat(),
            "query": session.query_params.get("query", "No text query"),
            "filters": session.query_params.get("filters", {}),
            "total_results": session.total_results,
            "cached_results": len(session.cached_results)
        })
    
    return sorted(sessions, key=lambda x: x["last_accessed"], reverse=True)


# Convenience functions for common queries

def _detect_author_strategy(author_name: str) -> str:
    """
    Automatically detect the best search strategy for an author name.
    
    Args:
        author_name: The author name to analyze
        
    Returns:
        Strategy string: "exact", "partial", or "fuzzy"
    """
    if not author_name or not author_name.strip():
        return "partial"
    
    name_parts = author_name.strip().split()
    
    # If it looks like a full name (2+ words), use exact matching
    if len(name_parts) >= 2:
        # Check if it looks like a proper name (starts with capitals)
        if all(part[0].isupper() for part in name_parts if part):
            return "exact"
        else:
            return "fuzzy"  # Might have typos or lowercase
    
    # Single word - check if it looks like a proper surname
    elif len(name_parts) == 1:
        name = name_parts[0]
        if name[0].isupper() and len(name) > 2:
            return "partial"  # Likely a surname
        else:
            return "fuzzy"  # Might have typos
    
    return "partial"  # Default fallback


def search_by_author(author_name: str, year_range: Optional[Dict] = None, strategy: str = "auto") -> Dict[str, Any]:
    """
    Convenience function to search for publications by a specific author.
    
    Args:
        author_name: Name of the author (partial match supported)
        year_range: Optional year range filter, e.g., {"gte": 2020}
        strategy: Query strategy to use:
                 - "exact": Use match_phrase for full names like "Christian Fager"
                 - "partial": Use prefix for single names like "Fager"
                 - "fuzzy": Use fuzzy matching for typos/variations
                 - "auto": Automatically detect and choose strategy (default)
    
    Returns:
        Search results summary with session ID
    """
    # Auto-detect strategy if not specified
    if strategy == "auto":
        strategy = _detect_author_strategy(author_name)
    
    filters = {"author": author_name, "author_strategy": strategy}
    if year_range:
        filters["year_range"] = year_range
    
    return search_publications(filters=filters)


def search_recent_publications(days: int = 30, query: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to search for recently added publications.
    
    Args:
        days: Number of days to look back
        query: Optional text query to combine with recency filter
    
    Returns:
        Search results summary with session ID
    """
    from_date = (datetime.now() - timedelta(days=days)).isoformat()
    filters = {"CreatedDate": {"gte": from_date}}
    
    return search_publications(query=query, filters=filters)


def get_statistics_summary() -> Dict[str, Any]:
    """
    Get overall statistics about the publications database.
    
    Returns:
        Summary statistics without needing a search session
    """
    # Create a temporary session for aggregations
    session = _session_manager.create_session()
    
    # Run empty search to get aggregations
    summary = session.execute_search(size=0)
    
    # Clean up session immediately since we don't need it
    del _session_manager.sessions[session.session_id]
    
    return {
        "total_publications": summary["total_results"],
        "years": summary["aggregations"].get("years", []),
        "publication_types": summary["aggregations"].get("types", []),
        "top_authors": summary["aggregations"].get("top_authors", [])[:10],
        "note": "Run search_publications() with filters to explore specific subsets"
    }