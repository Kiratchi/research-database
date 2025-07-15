"""
Query builder for converting parsed queries to Elasticsearch function calls.
"""

from typing import Dict, Any, Optional
from chat_parser import ParsedQuery, QueryIntent


class QueryBuilderError(Exception):
    """Exception raised when query building fails."""
    pass


class QueryBuilder:
    """Converts parsed queries into Elasticsearch function calls."""
    
    def __init__(self, agent_tools):
        """
        Initialize the query builder.
        
        Args:
            agent_tools: The agent tools module containing ES functions
        """
        self.agent_tools = agent_tools
    
    def build_query(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """
        Build a query specification from a parsed query.
        
        Args:
            parsed_query: ParsedQuery object with structured query info
            
        Returns:
            Dictionary with function name, args, and post-processing info
            
        Raises:
            QueryBuilderError: If the query cannot be built
        """
        if parsed_query.intent == QueryIntent.UNKNOWN:
            raise QueryBuilderError(f"Unknown intent: {parsed_query.intent}")
        
        # Determine the function to call based on entity type
        if parsed_query.entity_type == "author" and parsed_query.author_name:
            function_name = "search_by_author"
            
            # Detect the best strategy for this author name
            strategy = self._detect_author_strategy(parsed_query.author_name)
            
            args = {
                "author_name": parsed_query.author_name,
                "strategy": strategy
            }
            
            # Add year range if specified
            if parsed_query.filters.get("year_range"):
                args["year_range"] = parsed_query.filters["year_range"]
                
        else:
            # Default to search_publications
            function_name = "search_publications"
            args = {}
            
            # Add query if we have search terms
            if parsed_query.search_terms:
                args["query"] = parsed_query.search_terms
            
            # Build filters
            filters = {}
            if parsed_query.journal_name:
                filters["journal_name"] = parsed_query.journal_name
            
            # Add year/year_range filters
            if parsed_query.filters.get("year"):
                filters["year"] = parsed_query.filters["year"]
            elif parsed_query.filters.get("year_range"):
                filters["year_range"] = parsed_query.filters["year_range"]
            
            args["filters"] = filters
            
            # Add size limit if specified
            if parsed_query.limit:
                args["size"] = parsed_query.limit
            
            # For keyword stats, we don't need actual documents
            if parsed_query.entity_type == "keywords":
                args["size"] = 0
        
        # Determine post-processing based on intent
        post_process = self._get_post_process_type(parsed_query.intent)
        
        result = {
            "function": function_name,
            "args": args,
            "post_process": post_process
        }
        
        # Add post-processing arguments for stats queries
        if parsed_query.intent == QueryIntent.STATS:
            result["post_process_args"] = {
                "limit": parsed_query.limit or 10,
                "group_by": parsed_query.group_by
            }
        
        return result
    
    def _get_post_process_type(self, intent: QueryIntent) -> str:
        """Get the post-processing type for a given intent."""
        if intent == QueryIntent.COUNT:
            return "count"
        elif intent == QueryIntent.LIST:
            return "list"
        elif intent == QueryIntent.SEARCH:
            return "list"
        elif intent == QueryIntent.STATS:
            return "keyword_stats"
        else:
            return "list"
    
    def _detect_author_strategy(self, author_name: str) -> str:
        """
        Detect the best search strategy for an author name.
        
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
    
    def execute_query(self, query_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a query specification.
        
        Args:
            query_spec: Dictionary with function, args, and post-processing info
            
        Returns:
            Dictionary with formatted results
            
        Raises:
            QueryBuilderError: If the query execution fails
        """
        function_name = query_spec["function"]
        args = query_spec["args"]
        post_process = query_spec["post_process"]
        
        # Check if function exists and is callable
        if not hasattr(self.agent_tools, function_name):
            raise QueryBuilderError(f"Function {function_name} not found in agent tools")
        
        func = getattr(self.agent_tools, function_name)
        if not callable(func):
            raise QueryBuilderError(f"Function {function_name} is not callable")
        
        # Call the function
        es_result = func(**args)
        
        # Post-process the result based on the type
        if post_process == "count":
            return {
                "type": "count",
                "count": es_result["total_results"],
                "query": args.get("author_name", args.get("query", ""))
            }
        
        elif post_process == "list":
            return {
                "type": "list",
                "results": es_result.get("sample_results", []),
                "total": es_result["total_results"],
                "session_id": es_result["session_id"]
            }
        
        elif post_process == "keyword_stats":
            # For keyword stats, we need to make an additional call
            session_id = es_result["session_id"]
            post_process_args = query_spec.get("post_process_args", {})
            limit = post_process_args.get("limit", 10)
            
            # Get keyword statistics
            keyword_stats = self.agent_tools.get_field_statistics(
                session_id=session_id,
                field="Keywords.Value",
                top_n=limit
            )
            
            return {
                "type": "keyword_stats",
                "keywords": keyword_stats.get("values", []),
                "session_id": session_id,
                "group_by": post_process_args.get("group_by")
            }
        
        else:
            raise QueryBuilderError(f"Unknown post-processing type: {post_process}")