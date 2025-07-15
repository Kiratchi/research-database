"""
Test suite for LangChain tools.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_tools import (
    create_langchain_tools,
    initialize_langchain_tools,
    SearchPublicationsInput,
    SearchByAuthorInput,
    GetMoreResultsInput,
    GetFieldStatisticsInput,
    GetPublicationDetailsInput,
    _search_publications_wrapper,
    _search_by_author_wrapper,
    _get_more_results_wrapper,
    _get_field_statistics_wrapper,
    _get_statistics_summary_wrapper
)


class TestLangChainTools:
    """Test LangChain tools functionality."""
    
    def test_create_langchain_tools(self):
        """Test that LangChain tools are created with correct properties."""
        tools = create_langchain_tools()
        
        # Check that we have the expected tools
        tool_names = [tool.name for tool in tools]
        expected_tools = [
            "search_publications",
            "search_by_author", 
            "get_more_results",
            "get_field_statistics",
            "get_statistics_summary"
        ]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names, f"Missing tool: {expected_tool}"
        
        # Check that each tool has required properties
        for tool in tools:
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, 'func')
            assert tool.description is not None
            assert len(tool.description) > 0
    
    def test_search_publications_input_schema(self):
        """Test SearchPublicationsInput schema validation."""
        # Test with minimal input
        input_data = SearchPublicationsInput(query="machine learning")
        assert input_data.query == "machine learning"
        assert input_data.filters is None
        assert input_data.size == 20
        
        # Test with full input
        input_data = SearchPublicationsInput(
            query="artificial intelligence",
            filters={"year": 2023},
            size=10
        )
        assert input_data.query == "artificial intelligence"
        assert input_data.filters == {"year": 2023}
        assert input_data.size == 10
    
    def test_search_by_author_input_schema(self):
        """Test SearchByAuthorInput schema validation."""
        # Test with minimal input
        input_data = SearchByAuthorInput(author_name="Christian Fager")
        assert input_data.author_name == "Christian Fager"
        assert input_data.year_range is None
        assert input_data.strategy == "auto"
        
        # Test with full input
        input_data = SearchByAuthorInput(
            author_name="Anna Dubois",
            year_range={"gte": 2020, "lte": 2024},
            strategy="exact"
        )
        assert input_data.author_name == "Anna Dubois"
        assert input_data.year_range == {"gte": 2020, "lte": 2024}
        assert input_data.strategy == "exact"
    
    def test_get_more_results_input_schema(self):
        """Test GetMoreResultsInput schema validation."""
        input_data = GetMoreResultsInput(session_id="test-session-123")
        assert input_data.session_id == "test-session-123"
        assert input_data.page_number == 0
        assert input_data.page_size == 20
        
        # Test with custom values
        input_data = GetMoreResultsInput(
            session_id="test-session-456",
            page_number=2,
            page_size=50
        )
        assert input_data.session_id == "test-session-456"
        assert input_data.page_number == 2
        assert input_data.page_size == 50
    
    def test_get_field_statistics_input_schema(self):
        """Test GetFieldStatisticsInput schema validation."""
        input_data = GetFieldStatisticsInput(
            session_id="test-session-789",
            field="Year"
        )
        assert input_data.session_id == "test-session-789"
        assert input_data.field == "Year"
        assert input_data.top_n == 20
    
    @patch('langchain_tools.search_publications')
    def test_search_publications_wrapper_success(self, mock_search):
        """Test search_publications wrapper with successful result."""
        # Mock the search_publications result
        mock_result = {
            'total_results': 100,
            'session_id': 'test-session-123',
            'sample_results': [
                {
                    '_source': {
                        'Title': 'Machine Learning in Healthcare',
                        'authors': ['John Doe', 'Jane Smith'],
                        'Year': 2023
                    }
                }
            ],
            'aggregations': {'years': [], 'types': []}
        }
        mock_search.return_value = mock_result
        
        result = _search_publications_wrapper(query="machine learning")
        
        # Check that the result is formatted correctly
        assert "Total Results: 100" in result
        assert "Session ID: test-session-123" in result
        assert "Machine Learning in Healthcare" in result
        assert "John Doe, Jane Smith" in result
        assert "2023" in result
        
        # Verify the underlying function was called correctly
        mock_search.assert_called_once_with(query="machine learning", filters=None, size=20)
    
    @patch('langchain_tools.search_publications')
    def test_search_publications_wrapper_error(self, mock_search):
        """Test search_publications wrapper with error."""
        mock_search.side_effect = Exception("Test error")
        
        result = _search_publications_wrapper(query="test")
        
        assert "Error in search_publications: Test error" in result
    
    @patch('langchain_tools.search_by_author')
    def test_search_by_author_wrapper_success(self, mock_search):
        """Test search_by_author wrapper with successful result."""
        mock_result = {
            'total_results': 25,
            'session_id': 'author-session-456',
            'sample_results': [
                {
                    '_source': {
                        'Title': 'Research Paper 1',
                        'Year': 2023
                    }
                }
            ]
        }
        mock_search.return_value = mock_result
        
        result = _search_by_author_wrapper(author_name="Christian Fager")
        
        assert "Author: Christian Fager" in result
        assert "Total Publications: 25" in result
        assert "Session ID: author-session-456" in result
        assert "Research Paper 1" in result
        
        mock_search.assert_called_once_with(author_name="Christian Fager", year_range=None, strategy="auto")
    
    @patch('langchain_tools.get_more_results')
    def test_get_more_results_wrapper_success(self, mock_get_more):
        """Test get_more_results wrapper with successful result."""
        mock_result = {
            'results': [
                {
                    '_source': {
                        'Title': 'Page 2 Paper 1',
                        'authors': ['Author 1'],
                        'Year': 2022
                    }
                }
            ],
            'page': 1,
            'total_pages': 5,
            'has_next': True,
            'has_previous': True
        }
        mock_get_more.return_value = mock_result
        
        result = _get_more_results_wrapper(session_id="test-session", page_number=1)
        
        assert "Page 2 of 5" in result
        assert "Page 2 Paper 1" in result
        assert "More results available" in result
        
        mock_get_more.assert_called_once_with(session_id="test-session", page_number=1, page_size=20)
    
    @patch('langchain_tools.get_more_results')
    def test_get_more_results_wrapper_error(self, mock_get_more):
        """Test get_more_results wrapper with error."""
        mock_result = {
            'error': 'Session not found',
            'suggestion': 'Run a new search'
        }
        mock_get_more.return_value = mock_result
        
        result = _get_more_results_wrapper(session_id="invalid-session")
        
        assert "Error: Session not found" in result
    
    @patch('langchain_tools.get_field_statistics')
    def test_get_field_statistics_wrapper_success(self, mock_get_stats):
        """Test get_field_statistics wrapper with successful result."""
        mock_result = {
            'field': 'Year',
            'values': [
                {'value': '2023', 'count': 50},
                {'value': '2022', 'count': 45},
                {'value': '2021', 'count': 40}
            ],
            'session_id': 'test-session'
        }
        mock_get_stats.return_value = mock_result
        
        result = _get_field_statistics_wrapper(session_id="test-session", field="Year")
        
        assert "Field Statistics for 'Year'" in result
        assert "2023: 50 publications" in result
        assert "2022: 45 publications" in result
        assert "2021: 40 publications" in result
        
        mock_get_stats.assert_called_once_with(session_id="test-session", field="Year", top_n=20)
    
    @patch('langchain_tools.get_statistics_summary')
    def test_get_statistics_summary_wrapper_success(self, mock_get_summary):
        """Test get_statistics_summary wrapper with successful result."""
        mock_result = {
            'total_publications': 10000,
            'years': [
                {'key': '2023', 'doc_count': 1000},
                {'key': '2022', 'doc_count': 950}
            ],
            'publication_types': [
                {'key': 'Journal article', 'doc_count': 8000},
                {'key': 'Conference paper', 'doc_count': 2000}
            ],
            'top_authors': [
                {'key': 'John Doe', 'doc_count': 50},
                {'key': 'Jane Smith', 'doc_count': 45}
            ]
        }
        mock_get_summary.return_value = mock_result
        
        result = _get_statistics_summary_wrapper()
        
        assert "Total Publications: 10000" in result
        assert "2023: 1000 publications" in result
        assert "Journal article: 8000 publications" in result
        assert "John Doe: 50 publications" in result
        
        mock_get_summary.assert_called_once()
    
    @patch('langchain_tools.initialize_tools')
    def test_initialize_langchain_tools(self, mock_initialize):
        """Test initialize_langchain_tools function."""
        mock_es_client = Mock()
        
        tools = initialize_langchain_tools(mock_es_client, "test-index")
        
        # Check that initialization was called
        mock_initialize.assert_called_once_with(mock_es_client, "test-index")
        
        # Check that tools were created
        assert len(tools) == 5  # Expected number of tools
        assert all(hasattr(tool, 'name') for tool in tools)
        assert all(hasattr(tool, 'description') for tool in tools)
        assert all(hasattr(tool, 'func') for tool in tools)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])