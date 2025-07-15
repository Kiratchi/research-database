"""
Chat message parser for converting natural language queries to structured queries.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import re


class QueryIntent(Enum):
    """Enumeration of possible query intents."""
    COUNT = "count"
    LIST = "list"
    SEARCH = "search"
    STATS = "stats"
    UNKNOWN = "unknown"


@dataclass
class ParsedQuery:
    """Structured representation of a parsed query."""
    intent: QueryIntent
    entity_type: Optional[str] = None
    author_name: Optional[str] = None
    journal_name: Optional[str] = None
    search_terms: Optional[str] = None
    limit: Optional[int] = None
    group_by: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5


class ChatParser:
    """Parses natural language queries into structured queries."""
    
    def __init__(self):
        """Initialize the parser with pattern matching rules."""
        self.intent_patterns = {
            QueryIntent.COUNT: [
                r"how many",
                r"count",
                r"number of",
            ],
            QueryIntent.LIST: [
                r"list",
                r"show me",
                r"display",
                r"find all",
            ],
            QueryIntent.SEARCH: [
                r"find",
                r"search",
                r"look for",
                r"tell me about",
            ],
            QueryIntent.STATS: [
                r"top \d+",
                r"what are the top",
                r"statistics",
                r"trends",
                r"per year",
            ],
        }
        
        self.author_patterns = [
            # Full names - be more specific to avoid capturing "how many papers"
            r"(?:has|by|from|author)\s+([A-Z][a-z]+ [A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s+(?:published|has)",
            r"(?:has|by|from|author)\s+([a-z]+ [a-z]+(?:-[a-z]+)?)\s+(?:published|has)",
            r"(?:has|by|from|author)\s+([A-Z][a-z]+ [a-z]+(?:-[a-z]+)?)\s+(?:published|has)",
            r"(?:has|by|from|author)\s+([a-z]+ [A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s+(?:published|has)",
            # Pattern for "Christian Fager published" style
            r"([A-Z][a-z]+ [A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s+(?:published|has)",
            r"([a-z]+ [a-z]+(?:-[a-z]+)?)\s+(?:published|has)",
            r"([A-Z][a-z]+ [a-z]+(?:-[a-z]+)?)\s+(?:published|has)",
            r"([a-z]+ [A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s+(?:published|has)",
            # Single names - only match if after specific words
            r"(?:by|from|author)\s+([A-Z][a-z]+)\s+(?:published|has)",
            r"(?:by|from|author)\s+([a-z]+)\s+(?:published|has)",
            r"([A-Z][a-z]+)\s+(?:published|has)",
            r"([a-z]+)\s+(?:published|has)",
        ]
        
        self.year_patterns = [
            r"from (\d{4}) to (\d{4})",
            r"between (\d{4}) and (\d{4})",
            r"since (\d{4})",
            r"before (\d{4})",
            r"in (\d{4})",
        ]
    
    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a natural language query into a structured query.
        
        Args:
            query: Natural language query string
            
        Returns:
            ParsedQuery object with extracted information
        """
        # Classify the intent
        intent = self._classify_intent(query)
        
        # Extract author name if present
        author_name = self._extract_author_name(query)
        
        # Extract year range if present
        year_range = self._extract_year_range(query)
        
        # Extract journal name if present
        journal_name = self._extract_journal_name(query)
        
        # Extract search terms if present
        search_terms = self._extract_search_terms(query)
        
        # Extract limit if present
        limit = self._extract_limit(query)
        
        # Extract group_by if present
        group_by = self._extract_group_by(query)
        
        # Determine entity type
        entity_type = "author" if author_name else "keywords" if "keywords" in query.lower() else "publication"
        
        # Set confidence based on successful extractions
        confidence = 0.0
        if intent != QueryIntent.UNKNOWN:
            # Base confidence for recognized intent
            confidence = 0.7
            # Boost confidence for specific extractions
            if author_name:
                confidence += 0.2
            if journal_name:
                confidence += 0.1
            if year_range:
                confidence += 0.1
            # Lower confidence for very short search terms
            if search_terms and len(search_terms) < 10:
                confidence -= 0.2
            # Cap at 1.0
            confidence = min(confidence, 1.0)
        
        # Build filters
        filters = {}
        if year_range:
            if year_range.get("gte") == year_range.get("lte"):
                # Single year
                filters["year"] = year_range["gte"]
            else:
                # Year range
                filters["year_range"] = year_range
        
        return ParsedQuery(
            intent=intent,
            entity_type=entity_type,
            author_name=author_name,
            journal_name=journal_name,
            search_terms=search_terms,
            limit=limit,
            group_by=group_by,
            filters=filters,
            confidence=confidence
        )
    
    def _classify_intent(self, query: str) -> QueryIntent:
        """Classify the intent of a query based on pattern matching."""
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return QueryIntent.UNKNOWN
    
    def _extract_author_name(self, query: str) -> Optional[str]:
        """Extract author name from query."""
        for pattern in self.author_patterns:
            match = re.search(pattern, query)
            if match:
                # Title case the extracted name to handle lowercase input
                name = match.group(1)
                return name.title()
        return None
    
    def _extract_year_range(self, query: str) -> Dict[str, int]:
        """Extract year range from query."""
        year_range = {}
        
        # Try different year patterns
        for pattern in self.year_patterns:
            match = re.search(pattern, query)
            if match:
                groups = match.groups()
                if len(groups) == 2:  # from X to Y, between X and Y
                    year_range = {"gte": int(groups[0]), "lte": int(groups[1])}
                elif len(groups) == 1:
                    year = int(groups[0])
                    if "since" in pattern:
                        year_range = {"gte": year}
                    elif "before" in pattern:
                        year_range = {"lte": year}
                    elif "in" in pattern:
                        year_range = {"gte": year, "lte": year}
                break
        
        return year_range
    
    def _extract_journal_name(self, query: str) -> Optional[str]:
        """Extract journal name from query."""
        # Look for patterns like "in Nature" or "published in Nature"
        journal_pattern = r"(?:in|published in) ([A-Z][a-zA-Z\s]+?)(?:\s+in\s+\d{4}|\s+from|\s+since|\s+before|$)"
        match = re.search(journal_pattern, query)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_search_terms(self, query: str) -> Optional[str]:
        """Extract search terms from query."""
        # Look for patterns like "about machine learning" or "papers about X"
        search_patterns = [
            r"about (.+?)(?:\s+from|\s+since|\s+before|\s+in\s+\d{4}|$)",
            r"papers (.+?)(?:\s+from|\s+since|\s+before|\s+in\s+\d{4}|$)",
            r"find (.+?)(?:\s+from|\s+since|\s+before|\s+in\s+\d{4}|$)",
        ]
        
        for pattern in search_patterns:
            match = re.search(pattern, query.lower())
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_limit(self, query: str) -> Optional[int]:
        """Extract limit number from query."""
        # Look for patterns like "top 10" or "first 5"
        limit_pattern = r"(?:top|first) (\d+)"
        match = re.search(limit_pattern, query.lower())
        if match:
            return int(match.group(1))
        return None
    
    def _extract_group_by(self, query: str) -> Optional[str]:
        """Extract group_by field from query."""
        if "per year" in query.lower():
            return "year"
        return None