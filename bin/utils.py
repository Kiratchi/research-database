"""
Utility functions for Elasticsearch research tools.

Helper functions for formatting, validation, and common operations.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import re


def format_publication_for_display(publication: Dict[str, Any], 
                                 include_abstract: bool = False,
                                 max_authors: int = 5) -> str:
    """
    Format a publication for human-readable display.
    
    Args:
        publication: Publication document from Elasticsearch
        include_abstract: Whether to include the abstract
        max_authors: Maximum number of authors to display
    
    Returns:
        Formatted string representation
    """
    parts = []
    
    # Title and year
    title = publication.get("Title", "Untitled")
    year = publication.get("Year", "n.d.")
    parts.append(f"{title} ({year})")
    
    # Authors
    authors = extract_author_names(publication)
    if authors:
        if len(authors) > max_authors:
            author_str = ", ".join(authors[:max_authors]) + f" et al. (+{len(authors) - max_authors} more)"
        else:
            author_str = ", ".join(authors)
        parts.append(f"Authors: {author_str}")
    
    # Publication type
    pub_type = publication.get("PublicationType", {}).get("NameEng")
    if pub_type:
        parts.append(f"Type: {pub_type}")
    
    # Identifiers
    identifiers = []
    if publication.get("IdentifierDoi"):
        doi = publication["IdentifierDoi"]
        if isinstance(doi, list) and doi:
            identifiers.append(f"DOI: {doi[0]}")
        elif isinstance(doi, str):
            identifiers.append(f"DOI: {doi}")
    
    if identifiers:
        parts.append(" | ".join(identifiers))
    
    # Abstract
    if include_abstract and publication.get("Abstract"):
        abstract = publication["Abstract"]
        if len(abstract) > 200:
            abstract = abstract[:200] + "..."
        parts.append(f"\nAbstract: {abstract}")
    
    return "\n".join(parts)


def extract_author_names(publication: Dict[str, Any]) -> List[str]:
    """
    Extract author names from a publication document.
    
    Args:
        publication: Publication document from Elasticsearch
    
    Returns:
        List of author names
    """
    authors = []
    
    for person in publication.get("Persons", []):
        if "PersonData" in person and "DisplayName" in person["PersonData"]:
            name = person["PersonData"]["DisplayName"]
            if name:
                authors.append(name)
    
    return authors


def extract_organizations(publication: Dict[str, Any]) -> List[str]:
    """
    Extract organization names from a publication document.
    
    Args:
        publication: Publication document from Elasticsearch
    
    Returns:
        List of organization names
    """
    organizations = []
    
    for org in publication.get("Organizations", []):
        if "OrganizationData" in org:
            org_data = org["OrganizationData"]
            name = org_data.get("DisplayNameEng") or org_data.get("NameEng")
            if name:
                organizations.append(name)
    
    return organizations


def extract_keywords(publication: Dict[str, Any]) -> List[str]:
    """
    Extract keywords from a publication document.
    
    Args:
        publication: Publication document from Elasticsearch
    
    Returns:
        List of keywords
    """
    keywords = []
    
    for keyword in publication.get("Keywords", []):
        if isinstance(keyword, dict) and "Value" in keyword:
            keywords.append(keyword["Value"])
        elif isinstance(keyword, str):
            keywords.append(keyword)
    
    return keywords


def validate_year_range(year_range: Dict[str, int]) -> bool:
    """
    Validate a year range filter.
    
    Args:
        year_range: Dictionary with 'gte' and/or 'lte' keys
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(year_range, dict):
        return False
    
    valid_keys = {"gte", "lte", "gt", "lt"}
    if not all(key in valid_keys for key in year_range.keys()):
        return False
    
    # Check values are reasonable years
    current_year = datetime.now().year
    for key, value in year_range.items():
        if not isinstance(value, int) or value < 1900 or value > current_year + 5:
            return False
    
    # Check logical consistency
    if "gte" in year_range and "lte" in year_range:
        if year_range["gte"] > year_range["lte"]:
            return False
    
    return True


def parse_author_query(query: str) -> Dict[str, str]:
    """
    Parse an author query to extract first and last names.
    
    Args:
        query: Author name query string
    
    Returns:
        Dictionary with 'first_name' and 'last_name' if parseable
    """
    # Remove extra whitespace
    query = " ".join(query.split())
    
    # Simple split - last word is last name, rest is first name
    # This is naive but works for many cases
    parts = query.split()
    
    if len(parts) == 0:
        return {}
    elif len(parts) == 1:
        return {"last_name": parts[0]}
    else:
        return {
            "first_name": " ".join(parts[:-1]),
            "last_name": parts[-1]
        }


def build_filter_description(filters: Dict[str, Any]) -> str:
    """
    Build a human-readable description of active filters.
    
    Args:
        filters: Dictionary of filters
    
    Returns:
        Human-readable filter description
    """
    if not filters:
        return "No filters applied"
    
    descriptions = []
    
    for key, value in filters.items():
        if key == "year":
            descriptions.append(f"Year is {value}")
        elif key == "year_range":
            parts = []
            if "gte" in value:
                parts.append(f"from {value['gte']}")
            if "lte" in value:
                parts.append(f"to {value['lte']}")
            descriptions.append(f"Year {' '.join(parts)}")
        elif key == "author":
            descriptions.append(f"Author contains '{value}'")
        elif key == "organization":
            descriptions.append(f"Organization contains '{value}'")
        elif key == "has_fulltext":
            descriptions.append("Has fulltext" if value else "No fulltext")
        else:
            descriptions.append(f"{key}: {value}")
    
    return " AND ".join(descriptions)


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of tokens for LLM context management.
    
    Args:
        text: Text to estimate
    
    Returns:
        Estimated token count (rough approximation)
    """
    # Rough approximation: ~4 characters per token
    return len(text) // 4


def truncate_for_context(results: List[Dict], max_tokens: int = 2000) -> List[Dict]:
    """
    Truncate results to fit within token limit.
    
    Args:
        results: List of search results
        max_tokens: Maximum tokens allowed
    
    Returns:
        Truncated list of results
    """
    truncated = []
    total_tokens = 0
    
    for result in results:
        # Estimate tokens for this result
        result_text = str(result)
        result_tokens = estimate_tokens(result_text)
        
        if total_tokens + result_tokens > max_tokens:
            break
            
        truncated.append(result)
        total_tokens += result_tokens
    
    return truncated


def format_aggregation_results(aggregations: Dict[str, List[Dict]]) -> str:
    """
    Format aggregation results for display.
    
    Args:
        aggregations: Aggregation results from search
    
    Returns:
        Formatted string representation
    """
    lines = []
    
    for agg_name, buckets in aggregations.items():
        if buckets:
            lines.append(f"\n{agg_name.replace('_', ' ').title()}:")
            for item in buckets[:5]:  # Show top 5
                lines.append(f"  - {item['value']}: {item['count']:,}")
            if len(buckets) > 5:
                lines.append(f"  ... and {len(buckets) - 5} more")
    
    return "\n".join(lines)


def extract_identifiers(publication: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract all identifiers from a publication.
    
    Args:
        publication: Publication document
    
    Returns:
        Dictionary mapping identifier types to values
    """
    identifiers = {}
    
    # Direct identifier fields
    if publication.get("IdentifierDoi"):
        identifiers["DOI"] = publication["IdentifierDoi"] if isinstance(publication["IdentifierDoi"], list) else [publication["IdentifierDoi"]]
    
    if publication.get("IdentifierScopusId"):
        identifiers["Scopus"] = publication["IdentifierScopusId"] if isinstance(publication["IdentifierScopusId"], list) else [publication["IdentifierScopusId"]]
    
    if publication.get("IdentifierPubmedId"):
        identifiers["PubMed"] = publication["IdentifierPubmedId"] if isinstance(publication["IdentifierPubmedId"], list) else [publication["IdentifierPubmedId"]]
    
    # Identifiers array
    for identifier in publication.get("Identifiers", []):
        if "Type" in identifier and "Value" in identifier:
            type_name = identifier["Type"].get("Value", "Unknown")
            if type_name not in identifiers:
                identifiers[type_name] = []
            identifiers[type_name].append(identifier["Value"])
    
    return identifiers


def create_search_summary(total: int, query: Optional[str], filters: Optional[Dict]) -> str:
    """
    Create a summary of what was searched.
    
    Args:
        total: Total results found
        query: Text query used
        filters: Filters applied
    
    Returns:
        Human-readable search summary
    """
    parts = [f"Found {total:,} publications"]
    
    if query:
        parts.append(f"matching '{query}'")
    
    if filters:
        parts.append(f"with filters: {build_filter_description(filters)}")
    
    return " ".join(parts)