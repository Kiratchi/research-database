"""
Test suite for response formatter functionality.
Tests the conversion of query results into human-readable chat responses.
"""

import pytest
from response_formatter import ResponseFormatter


class TestResponseFormatter:
    """Test the ResponseFormatter class that converts query results to chat responses."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ResponseFormatter()
    
    def test_format_count_response(self):
        """Test formatting count query responses."""
        query_result = {
            "type": "count",
            "count": 42,
            "query": "Christian Fager"
        }
        
        response = self.formatter.format_response(query_result)
        
        assert response["type"] == "text"
        assert "42" in response["content"]
        assert "Christian Fager" in response["content"]
        assert "publications" in response["content"].lower()
    
    def test_format_count_response_zero(self):
        """Test formatting count responses with zero results."""
        query_result = {
            "type": "count",
            "count": 0,
            "query": "Nonexistent Author"
        }
        
        response = self.formatter.format_response(query_result)
        
        assert response["type"] == "text"
        assert "0" in response["content"] or "no" in response["content"].lower()
        assert "Nonexistent Author" in response["content"]
    
    def test_format_count_response_singular(self):
        """Test formatting count responses with singular result."""
        query_result = {
            "type": "count",
            "count": 1,
            "query": "Single Author"
        }
        
        response = self.formatter.format_response(query_result)
        
        assert response["type"] == "text"
        assert "1" in response["content"]
        assert "publication" in response["content"].lower()
        assert "publications" not in response["content"].lower()  # Should be singular
    
    def test_format_list_response(self):
        """Test formatting list query responses."""
        query_result = {
            "type": "list",
            "results": [
                {
                    "title": "Machine Learning in Healthcare",
                    "year": 2023,
                    "authors": ["John Doe", "Jane Smith"],
                    "abstract": "This paper explores the application of machine learning..."
                },
                {
                    "title": "Deep Learning Applications",
                    "year": 2022,
                    "authors": ["Alice Johnson"],
                    "abstract": "A comprehensive review of deep learning applications..."
                }
            ],
            "total": 15,
            "session_id": "test123"
        }
        
        response = self.formatter.format_response(query_result)
        
        assert response["type"] == "list"
        assert len(response["items"]) == 2
        
        # Check first item
        first_item = response["items"][0]
        assert first_item["title"] == "Machine Learning in Healthcare"
        assert first_item["year"] == 2023
        assert "John Doe, Jane Smith" in first_item["authors"]
        assert "This paper explores" in first_item["abstract"]
        
        # Check metadata
        assert response["total"] == 15
        assert response["session_id"] == "test123"
    
    def test_format_list_response_empty(self):
        """Test formatting empty list responses."""
        query_result = {
            "type": "list",
            "results": [],
            "total": 0,
            "session_id": "empty123"
        }
        
        response = self.formatter.format_response(query_result)
        
        assert response["type"] == "text"
        assert "no results" in response["content"].lower() or "not found" in response["content"].lower()
    
    def test_format_keyword_stats_response(self):
        """Test formatting keyword statistics responses."""
        query_result = {
            "type": "keyword_stats",
            "keywords": [
                {"value": "machine learning", "count": 150},
                {"value": "artificial intelligence", "count": 120},
                {"value": "deep learning", "count": 85},
                {"value": "neural networks", "count": 70}
            ],
            "session_id": "stats456",
            "group_by": "year"
        }
        
        response = self.formatter.format_response(query_result)
        
        assert response["type"] == "table"
        assert len(response["headers"]) == 2
        assert "keyword" in response["headers"][0].lower()
        assert "count" in response["headers"][1].lower()
        
        # Check data rows
        assert len(response["rows"]) == 4
        assert response["rows"][0] == ["machine learning", "150"]
        assert response["rows"][1] == ["artificial intelligence", "120"]
        
        # Check summary
        assert "top keywords" in response["summary"].lower()
        assert "machine learning" in response["summary"]
    
    def test_format_keyword_stats_response_empty(self):
        """Test formatting empty keyword statistics responses."""
        query_result = {
            "type": "keyword_stats",
            "keywords": [],
            "session_id": "empty_stats",
            "group_by": "year"
        }
        
        response = self.formatter.format_response(query_result)
        
        assert response["type"] == "text"
        assert "no keywords" in response["content"].lower()
    
    def test_format_response_with_truncation(self):
        """Test formatting responses with many results (truncation)."""
        # Create a list with many results
        many_results = []
        for i in range(50):
            many_results.append({
                "title": f"Paper {i+1}",
                "year": 2023,
                "authors": [f"Author {i+1}"],
                "abstract": f"Abstract for paper {i+1}..."
            })
        
        query_result = {
            "type": "list",
            "results": many_results,
            "total": 50,
            "session_id": "big_list"
        }
        
        response = self.formatter.format_response(query_result)
        
        assert response["type"] == "list"
        # Should be truncated to a reasonable number
        assert len(response["items"]) <= 10
        assert response["total"] == 50
        assert response.get("truncated") is True
    
    def test_format_response_with_missing_fields(self):
        """Test formatting responses with missing fields."""
        query_result = {
            "type": "list",
            "results": [
                {
                    "title": "Paper with Missing Fields",
                    # Missing year, authors, abstract
                },
                {
                    "year": 2023,
                    "authors": ["Complete Author"],
                    "abstract": "Complete abstract"
                    # Missing title
                }
            ],
            "total": 2,
            "session_id": "missing_fields"
        }
        
        response = self.formatter.format_response(query_result)
        
        assert response["type"] == "list"
        assert len(response["items"]) == 2
        
        # Check that missing fields are handled gracefully
        first_item = response["items"][0]
        assert first_item["title"] == "Paper with Missing Fields"
        assert first_item["year"] == "Unknown"
        assert first_item["authors"] == "Unknown"
        assert first_item["abstract"] == "No abstract available"
        
        second_item = response["items"][1]
        assert second_item["title"] == "Untitled"
        assert second_item["year"] == 2023
    
    def test_format_unknown_response_type(self):
        """Test handling of unknown response types."""
        query_result = {
            "type": "unknown_type",
            "data": "some data"
        }
        
        with pytest.raises(ValueError) as exc_info:
            self.formatter.format_response(query_result)
        
        assert "Unknown response type" in str(exc_info.value)
    
    def test_format_response_suggestions(self):
        """Test formatting responses with suggestions for follow-up queries."""
        query_result = {
            "type": "count",
            "count": 42,
            "query": "Christian Fager",
            "session_id": "suggest123"
        }
        
        response = self.formatter.format_response(query_result, include_suggestions=True)
        
        assert response["type"] == "text"
        assert "suggestions" in response
        assert len(response["suggestions"]) > 0
        
        # Check that suggestions are relevant
        suggestions = response["suggestions"]
        assert any("list" in s.lower() for s in suggestions)
        assert any("recent" in s.lower() for s in suggestions)
    
    def test_format_response_with_session_context(self):
        """Test formatting responses with session context."""
        query_result = {
            "type": "list",
            "results": [
                {"title": "Paper 1", "year": 2023, "authors": ["Author 1"]},
                {"title": "Paper 2", "year": 2022, "authors": ["Author 2"]}
            ],
            "total": 25,
            "session_id": "context123"
        }
        
        response = self.formatter.format_response(
            query_result, 
            session_context={"previous_query": "Christian Fager", "page": 1}
        )
        
        assert response["type"] == "list"
        assert "more results" in response.get("footer", "").lower()
        assert "Christian Fager" in response.get("context", "")
    
    def test_format_error_response(self):
        """Test formatting error responses."""
        error_result = {
            "type": "error",
            "error": "Session expired",
            "details": "Please start a new search"
        }
        
        response = self.formatter.format_response(error_result)
        
        assert response["type"] == "error"
        assert "Session expired" in response["content"]
        assert "Please start a new search" in response["content"]
    
    def test_truncate_abstract(self):
        """Test abstract truncation helper method."""
        # Test case where period is far from the end
        long_abstract = "This is a very long abstract that should be truncated and this continues for a very long time without any periods so it will be truncated with ellipsis"
        
        truncated = self.formatter._truncate_abstract(long_abstract, max_length=100)
        
        assert len(truncated) <= 103  # 100 + "..."
        assert truncated.endswith("...")
        assert "This is a very long abstract" in truncated
        
        # Test case where period is near the end
        abstract_with_period = "This is a shorter abstract that ends with a period."
        truncated2 = self.formatter._truncate_abstract(abstract_with_period, max_length=100)
        assert truncated2 == abstract_with_period  # Should not be truncated
        
        # Test case where period is exactly at the cutoff
        abstract_with_close_period = "This is an abstract that has a period. And then continues a bit more."
        truncated3 = self.formatter._truncate_abstract(abstract_with_close_period, max_length=50)
        assert truncated3.endswith(".")  # Should end with period, not ellipsis
    
    def test_format_author_list(self):
        """Test author list formatting helper method."""
        # Test with multiple authors
        authors = ["John Doe", "Jane Smith", "Bob Johnson"]
        formatted = self.formatter._format_author_list(authors)
        assert formatted == "John Doe, Jane Smith, Bob Johnson"
        
        # Test with many authors (should truncate)
        many_authors = [f"Author {i}" for i in range(10)]
        formatted = self.formatter._format_author_list(many_authors)
        assert "et al." in formatted
        assert len(formatted.split(", ")) <= 4  # Should truncate to 3 + "et al."
        
        # Test with empty list
        formatted = self.formatter._format_author_list([])
        assert formatted == "Unknown"
    
    def test_format_with_markdown(self):
        """Test formatting responses with markdown formatting."""
        query_result = {
            "type": "count",
            "count": 42,
            "query": "Christian Fager"
        }
        
        response = self.formatter.format_response(query_result, use_markdown=True)
        
        assert response["type"] == "text"
        assert "**" in response["content"]  # Bold formatting
        assert "*" in response["content"]  # Italic formatting