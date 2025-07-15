"""
Response formatter for converting query results into human-readable chat responses.
"""

from typing import Dict, Any, List, Optional


class ResponseFormatter:
    """Formats query results into human-readable chat responses."""
    
    def __init__(self):
        """Initialize the response formatter."""
        self.max_list_items = 10
        self.max_abstract_length = 200
        self.max_authors_shown = 3
    
    def format_response(self, query_result: Dict[str, Any], 
                       include_suggestions: bool = False,
                       session_context: Optional[Dict[str, Any]] = None,
                       use_markdown: bool = False) -> Dict[str, Any]:
        """
        Format a query result into a human-readable response.
        
        Args:
            query_result: The result from query execution
            include_suggestions: Whether to include follow-up suggestions
            session_context: Context about the current session
            use_markdown: Whether to use markdown formatting
            
        Returns:
            Dictionary with formatted response
            
        Raises:
            ValueError: If the response type is unknown
        """
        response_type = query_result.get("type", "unknown")
        
        if response_type == "count":
            return self._format_count_response(query_result, include_suggestions, use_markdown)
        elif response_type == "list":
            return self._format_list_response(query_result, session_context, use_markdown)
        elif response_type == "keyword_stats":
            return self._format_keyword_stats_response(query_result, use_markdown)
        elif response_type == "error":
            return self._format_error_response(query_result, use_markdown)
        else:
            raise ValueError(f"Unknown response type: {response_type}")
    
    def _format_count_response(self, result: Dict[str, Any], 
                              include_suggestions: bool = False,
                              use_markdown: bool = False) -> Dict[str, Any]:
        """Format a count response."""
        count = result["count"]
        query = result["query"]
        
        # Handle plural/singular
        if count == 0:
            content = f"No publications found for '{query}'."
        elif count == 1:
            content = f"Found 1 publication for '{query}'."
        else:
            content = f"Found {count} publications for '{query}'."
        
        # Apply markdown formatting if requested
        if use_markdown:
            content = content.replace(f"'{query}'", f"**{query}**")
            content = content.replace(f"{count}", f"**{count}**")
        
        response = {
            "type": "text",
            "content": content
        }
        
        # Add suggestions if requested
        if include_suggestions and count > 0:
            response["suggestions"] = self._generate_suggestions(result)
        
        return response
    
    def _format_list_response(self, result: Dict[str, Any], 
                             session_context: Optional[Dict[str, Any]] = None,
                             use_markdown: bool = False) -> Dict[str, Any]:
        """Format a list response."""
        results = result["results"]
        total = result["total"]
        
        # Handle empty results
        if not results:
            return {
                "type": "text",
                "content": "No results found for your query."
            }
        
        # Truncate if too many results
        truncated = len(results) > self.max_list_items
        if truncated:
            results = results[:self.max_list_items]
        
        # Format each item
        items = []
        for item in results:
            formatted_item = {
                "title": item.get("title", "Untitled"),
                "year": item.get("year", "Unknown"),
                "authors": self._format_author_list(item.get("authors", [])),
                "abstract": self._truncate_abstract(item.get("abstract", "No abstract available"))
            }
            items.append(formatted_item)
        
        response = {
            "type": "list",
            "items": items,
            "total": total,
            "session_id": result["session_id"]
        }
        
        if truncated:
            response["truncated"] = True
        
        # Add context and footer if session context provided
        if session_context:
            if session_context.get("previous_query"):
                response["context"] = f"Results for: {session_context['previous_query']}"
            
            if total > len(items):
                response["footer"] = f"Showing {len(items)} of {total} results. Ask for more results to see additional items."
        
        return response
    
    def _format_keyword_stats_response(self, result: Dict[str, Any],
                                      use_markdown: bool = False) -> Dict[str, Any]:
        """Format a keyword statistics response."""
        keywords = result["keywords"]
        
        # Handle empty keywords
        if not keywords:
            return {
                "type": "text",
                "content": "No keywords found for your query."
            }
        
        # Format as table
        headers = ["Keyword", "Count"]
        rows = []
        
        for keyword in keywords:
            rows.append([keyword["value"], str(keyword["count"])])
        
        # Create summary
        top_keyword = keywords[0]
        summary = f"Top keywords found. Most common: '{top_keyword['value']}' with {top_keyword['count']} occurrences."
        
        if use_markdown:
            summary = summary.replace(f"'{top_keyword['value']}'", f"**{top_keyword['value']}**")
        
        return {
            "type": "table",
            "headers": headers,
            "rows": rows,
            "summary": summary
        }
    
    def _format_error_response(self, result: Dict[str, Any],
                              use_markdown: bool = False) -> Dict[str, Any]:
        """Format an error response."""
        error_message = result.get("error", "An error occurred")
        details = result.get("details", "")
        
        content = error_message
        if details:
            content += f". {details}"
        
        return {
            "type": "error",
            "content": content
        }
    
    def _format_author_list(self, authors: List[str]) -> str:
        """Format a list of authors."""
        if not authors:
            return "Unknown"
        
        if len(authors) <= self.max_authors_shown:
            return ", ".join(authors)
        else:
            shown = authors[:self.max_authors_shown]
            return ", ".join(shown) + " et al."
    
    def _truncate_abstract(self, abstract: str, max_length: Optional[int] = None) -> str:
        """Truncate an abstract to a reasonable length."""
        if not abstract:
            return "No abstract available"
        
        max_len = max_length or self.max_abstract_length
        
        if len(abstract) <= max_len:
            return abstract
        
        # Find the last complete sentence within the limit
        truncated = abstract[:max_len]
        last_period = truncated.rfind('.')
        last_space = truncated.rfind(' ')
        
        if last_period > max_len - 50:  # If there's a period near the end
            return abstract[:last_period + 1]
        elif last_space > max_len - 20:  # If there's a space near the end
            return abstract[:last_space] + "..."
        else:
            return abstract[:max_len] + "..."
    
    def _generate_suggestions(self, result: Dict[str, Any]) -> List[str]:
        """Generate follow-up suggestions based on the result."""
        suggestions = []
        
        if result["type"] == "count":
            query = result["query"]
            suggestions.append(f"List publications by {query}")
            suggestions.append(f"Show recent publications by {query}")
            suggestions.append(f"Find {query}'s most cited papers")
        
        return suggestions