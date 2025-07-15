"""
Test suite for query builder functionality.
Tests the conversion of parsed queries to Elasticsearch function calls.
"""

import pytest
from unittest.mock import Mock, patch
from query_builder import QueryBuilder, QueryBuilderError
from chat_parser import ParsedQuery, QueryIntent


class TestQueryBuilder:
    """Test the QueryBuilder class that converts parsed queries to ES calls."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock the agent tools with specific functions
        self.mock_agent_tools = Mock()
        # Set up the functions that should exist
        self.mock_agent_tools.search_by_author = Mock()
        self.mock_agent_tools.search_publications = Mock()
        self.mock_agent_tools.get_field_statistics = Mock()
        self.builder = QueryBuilder(self.mock_agent_tools)
    
    def test_build_author_count_query(self):
        """Test building author count queries."""
        parsed_query = ParsedQuery(
            intent=QueryIntent.COUNT,
            entity_type="author",
            author_name="Christian Fager",
            filters={}
        )
        
        result = self.builder.build_query(parsed_query)
        
        assert result["function"] == "search_by_author"
        assert result["args"]["author_name"] == "Christian Fager"
        assert result["args"]["strategy"] == "exact"  # Auto-detected for full name
        assert result["post_process"] == "count"
    
    def test_build_author_list_query(self):
        """Test building author list queries."""
        parsed_query = ParsedQuery(
            intent=QueryIntent.LIST,
            entity_type="author",
            author_name="Anna Dubois",
            filters={}
        )
        
        result = self.builder.build_query(parsed_query)
        
        assert result["function"] == "search_by_author"
        assert result["args"]["author_name"] == "Anna Dubois"
        assert result["args"]["strategy"] == "exact"  # Auto-detected for full name
        assert result["post_process"] == "list"
    
    def test_build_author_with_year_filter(self):
        """Test building author queries with year filters."""
        parsed_query = ParsedQuery(
            intent=QueryIntent.LIST,
            entity_type="author",
            author_name="Erik Lind",
            filters={"year_range": {"gte": 2022, "lte": 2024}}
        )
        
        result = self.builder.build_query(parsed_query)
        
        assert result["function"] == "search_by_author"
        assert result["args"]["author_name"] == "Erik Lind"
        assert result["args"]["strategy"] == "exact"  # Auto-detected for full name
        assert result["args"]["year_range"] == {"gte": 2022, "lte": 2024}
        assert result["post_process"] == "list"
    
    def test_build_journal_year_query(self):
        """Test building journal/year queries."""
        parsed_query = ParsedQuery(
            intent=QueryIntent.COUNT,
            entity_type="publication",
            journal_name="Nature",
            filters={"year": 2023}
        )
        
        result = self.builder.build_query(parsed_query)
        
        assert result["function"] == "search_publications"
        assert result["args"] == {
            "filters": {
                "journal_name": "Nature",
                "year": 2023
            }
        }
        assert result["post_process"] == "count"
    
    def test_build_search_query(self):
        """Test building search queries."""
        parsed_query = ParsedQuery(
            intent=QueryIntent.SEARCH,
            entity_type="publication",
            search_terms="machine learning",
            filters={}
        )
        
        result = self.builder.build_query(parsed_query)
        
        assert result["function"] == "search_publications"
        assert result["args"] == {
            "query": "machine learning",
            "filters": {}
        }
        assert result["post_process"] == "list"
    
    def test_build_keywords_stats_query(self):
        """Test building keyword statistics queries."""
        parsed_query = ParsedQuery(
            intent=QueryIntent.STATS,
            entity_type="keywords",
            limit=10,
            group_by="year",
            filters={"year_range": {"gte": 2020, "lte": 2024}}
        )
        
        result = self.builder.build_query(parsed_query)
        
        assert result["function"] == "search_publications"
        assert result["args"] == {
            "filters": {"year_range": {"gte": 2020, "lte": 2024}},
            "size": 0  # We only want aggregations
        }
        assert result["post_process"] == "keyword_stats"
        assert result["post_process_args"] == {
            "limit": 10,
            "group_by": "year"
        }
    
    def test_build_query_with_size_limit(self):
        """Test building queries with size limits."""
        parsed_query = ParsedQuery(
            intent=QueryIntent.LIST,
            entity_type="publication",
            search_terms="AI",
            limit=5,
            filters={}
        )
        
        result = self.builder.build_query(parsed_query)
        
        assert result["function"] == "search_publications"
        assert result["args"] == {
            "query": "AI",
            "filters": {},
            "size": 5
        }
        assert result["post_process"] == "list"
    
    def test_build_unknown_intent_query(self):
        """Test handling of unknown intent queries."""
        parsed_query = ParsedQuery(
            intent=QueryIntent.UNKNOWN,
            entity_type="publication"
        )
        
        with pytest.raises(QueryBuilderError) as exc_info:
            self.builder.build_query(parsed_query)
        
        assert "Unknown intent" in str(exc_info.value)
    
    def test_execute_query_count(self):
        """Test executing a count query."""
        # Mock the search function to return a session
        mock_session = Mock()
        mock_session.total_results = 42
        self.mock_agent_tools.search_by_author.return_value = {
            "session_id": "test123",
            "total_results": 42
        }
        
        query_spec = {
            "function": "search_by_author",
            "args": {"author_name": "Test Author"},
            "post_process": "count"
        }
        
        result = self.builder.execute_query(query_spec)
        
        assert result["type"] == "count"
        assert result["count"] == 42
        assert result["query"] == "Test Author"
        self.mock_agent_tools.search_by_author.assert_called_once_with(
            author_name="Test Author"
        )
    
    def test_execute_query_list(self):
        """Test executing a list query."""
        mock_results = [
            {"title": "Paper 1", "year": 2023},
            {"title": "Paper 2", "year": 2022}
        ]
        
        self.mock_agent_tools.search_publications.return_value = {
            "session_id": "test456",
            "total_results": 2,
            "sample_results": mock_results
        }
        
        query_spec = {
            "function": "search_publications",
            "args": {"query": "test"},
            "post_process": "list"
        }
        
        result = self.builder.execute_query(query_spec)
        
        assert result["type"] == "list"
        assert result["results"] == mock_results
        assert result["total"] == 2
        assert result["session_id"] == "test456"
    
    def test_execute_query_keyword_stats(self):
        """Test executing a keyword statistics query."""
        # This is more complex as it requires multiple calls
        self.mock_agent_tools.search_publications.return_value = {
            "session_id": "test789",
            "total_results": 1000
        }
        
        # Mock the field statistics call
        self.mock_agent_tools.get_field_statistics.return_value = {
            "field": "Keywords.Value",
            "values": [
                {"value": "machine learning", "count": 50},
                {"value": "artificial intelligence", "count": 30}
            ]
        }
        
        query_spec = {
            "function": "search_publications",
            "args": {"filters": {"year_range": {"gte": 2020, "lte": 2024}}},
            "post_process": "keyword_stats",
            "post_process_args": {"limit": 10, "group_by": "year"}
        }
        
        result = self.builder.execute_query(query_spec)
        
        assert result["type"] == "keyword_stats"
        assert result["keywords"] == [
            {"value": "machine learning", "count": 50},
            {"value": "artificial intelligence", "count": 30}
        ]
        assert result["session_id"] == "test789"
    
    def test_execute_query_function_not_found(self):
        """Test handling of non-existent function calls."""
        # Create a real object without the function
        class MockAgentTools:
            def search_by_author(self):
                pass
                
        real_mock = MockAgentTools()
        builder = QueryBuilder(real_mock)
        
        query_spec = {
            "function": "nonexistent_function",
            "args": {},
            "post_process": "count"
        }
        
        # This should raise an exception because the function doesn't exist
        with pytest.raises(QueryBuilderError) as exc_info:
            builder.execute_query(query_spec)
        
        assert "Function nonexistent_function not found" in str(exc_info.value)
    
    def test_build_and_execute_integration(self):
        """Test the full build and execute flow."""
        # Mock a complete search response
        self.mock_agent_tools.search_by_author.return_value = {
            "session_id": "integration_test",
            "total_results": 15,
            "sample_results": [
                {"title": "Test Paper", "year": 2023, "authors": ["Test Author"]}
            ]
        }
        
        parsed_query = ParsedQuery(
            intent=QueryIntent.COUNT,
            entity_type="author",
            author_name="Test Author",
            filters={}
        )
        
        # Build the query
        query_spec = self.builder.build_query(parsed_query)
        
        # Execute the query
        result = self.builder.execute_query(query_spec)
        
        assert result["type"] == "count"
        assert result["count"] == 15
        assert result["query"] == "Test Author"


class TestQueryBuilderError:
    """Test the QueryBuilderError exception."""
    
    def test_query_builder_error_creation(self):
        """Test creating QueryBuilderError."""
        error = QueryBuilderError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)