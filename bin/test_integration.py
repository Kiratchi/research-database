"""
Integration tests for the complete chat agent flow.
Tests the end-to-end flow from natural language to formatted response.
"""

import pytest
from unittest.mock import Mock
from chat_parser import ChatParser
from query_builder import QueryBuilder
from response_formatter import ResponseFormatter


class TestChatAgentIntegration:
    """Test the complete chat agent flow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = ChatParser()
        
        # Mock the agent tools
        self.mock_agent_tools = Mock()
        self.mock_agent_tools.search_by_author = Mock()
        self.mock_agent_tools.search_publications = Mock()
        self.mock_agent_tools.get_field_statistics = Mock()
        
        self.builder = QueryBuilder(self.mock_agent_tools)
        self.formatter = ResponseFormatter()
    
    def test_complete_author_count_flow(self):
        """Test the complete flow for author count queries."""
        # Step 1: Natural language input
        user_query = "How many articles has Christian Fager published?"
        
        # Step 2: Parse the query
        parsed_query = self.parser.parse(user_query)
        
        # Step 3: Build the query
        query_spec = self.builder.build_query(parsed_query)
        
        # Step 4: Mock the ES response
        self.mock_agent_tools.search_by_author.return_value = {
            "session_id": "test123",
            "total_results": 42,
            "sample_results": []
        }
        
        # Step 5: Execute the query
        query_result = self.builder.execute_query(query_spec)
        
        # Step 6: Format the response
        formatted_response = self.formatter.format_response(query_result)
        
        # Verify the complete flow
        assert formatted_response["type"] == "text"
        assert "42" in formatted_response["content"]
        assert "Christian Fager" in formatted_response["content"]
        assert "publications" in formatted_response["content"].lower()
    
    def test_complete_author_list_flow(self):
        """Test the complete flow for author list queries."""
        # Step 1: Natural language input
        user_query = "List all articles that Anna Dubois has published from 2020 to 2023"
        
        # Step 2: Parse the query
        parsed_query = self.parser.parse(user_query)
        
        # Step 3: Build the query
        query_spec = self.builder.build_query(parsed_query)
        
        # Step 4: Mock the ES response
        self.mock_agent_tools.search_by_author.return_value = {
            "session_id": "test456",
            "total_results": 5,
            "sample_results": [
                {
                    "title": "Supply Chain Innovation",
                    "year": 2022,
                    "authors": ["Anna Dubois", "Lars-Erik Gadde"],
                    "abstract": "This paper explores innovation in supply chains..."
                },
                {
                    "title": "Industrial Networks",
                    "year": 2021,
                    "authors": ["Anna Dubois"],
                    "abstract": "A study of industrial network relationships..."
                }
            ]
        }
        
        # Step 5: Execute the query
        query_result = self.builder.execute_query(query_spec)
        
        # Step 6: Format the response
        formatted_response = self.formatter.format_response(query_result)
        
        # Verify the complete flow
        assert formatted_response["type"] == "list"
        assert len(formatted_response["items"]) == 2
        assert formatted_response["items"][0]["title"] == "Supply Chain Innovation"
        assert formatted_response["items"][0]["year"] == 2022
        assert "Anna Dubois" in formatted_response["items"][0]["authors"]
        assert formatted_response["total"] == 5
    
    def test_complete_keyword_stats_flow(self):
        """Test the complete flow for keyword statistics queries."""
        # Step 1: Natural language input
        user_query = "What are the top 5 keywords on publications per year from 2020 to 2024?"
        
        # Step 2: Parse the query
        parsed_query = self.parser.parse(user_query)
        
        # Step 3: Build the query
        query_spec = self.builder.build_query(parsed_query)
        
        # Step 4: Mock the ES responses
        self.mock_agent_tools.search_publications.return_value = {
            "session_id": "test789",
            "total_results": 1000,
            "sample_results": []
        }
        
        self.mock_agent_tools.get_field_statistics.return_value = {
            "field": "Keywords.Value",
            "values": [
                {"value": "machine learning", "count": 150},
                {"value": "artificial intelligence", "count": 120},
                {"value": "deep learning", "count": 85},
                {"value": "neural networks", "count": 70},
                {"value": "data mining", "count": 60}
            ]
        }
        
        # Step 5: Execute the query
        query_result = self.builder.execute_query(query_spec)
        
        # Step 6: Format the response
        formatted_response = self.formatter.format_response(query_result)
        
        # Verify the complete flow
        assert formatted_response["type"] == "table"
        assert len(formatted_response["rows"]) == 5
        assert formatted_response["rows"][0] == ["machine learning", "150"]
        assert "machine learning" in formatted_response["summary"]
    
    def test_complete_search_flow(self):
        """Test the complete flow for search queries."""
        # Step 1: Natural language input
        user_query = "Find papers about quantum computing from 2023"
        
        # Step 2: Parse the query
        parsed_query = self.parser.parse(user_query)
        
        # Step 3: Build the query
        query_spec = self.builder.build_query(parsed_query)
        
        # Step 4: Mock the ES response
        self.mock_agent_tools.search_publications.return_value = {
            "session_id": "search123",
            "total_results": 8,
            "sample_results": [
                {
                    "title": "Quantum Computing Advances",
                    "year": 2023,
                    "authors": ["Dr. Quantum", "Prof. Computing"],
                    "abstract": "Recent advances in quantum computing technology..."
                }
            ]
        }
        
        # Step 5: Execute the query
        query_result = self.builder.execute_query(query_spec)
        
        # Step 6: Format the response
        formatted_response = self.formatter.format_response(query_result)
        
        # Verify the complete flow
        assert formatted_response["type"] == "list"
        assert len(formatted_response["items"]) == 1
        assert formatted_response["items"][0]["title"] == "Quantum Computing Advances"
        assert formatted_response["total"] == 8
        assert formatted_response["session_id"] == "search123"
    
    def test_error_handling_flow(self):
        """Test error handling in the complete flow."""
        # Step 1: Natural language input
        user_query = "asdfghjkl random nonsense"
        
        # Step 2: Parse the query
        parsed_query = self.parser.parse(user_query)
        
        # Step 3: Try to build the query (should fail for unknown intent)
        try:
            query_spec = self.builder.build_query(parsed_query)
            assert False, "Should have raised an exception"
        except Exception as e:
            # This is expected for unknown intent
            assert "Unknown intent" in str(e)
    
    def test_complete_flow_with_suggestions(self):
        """Test the complete flow with response suggestions."""
        # Step 1: Natural language input
        user_query = "How many papers has Erik Lind published?"
        
        # Step 2: Parse the query
        parsed_query = self.parser.parse(user_query)
        
        # Step 3: Build the query
        query_spec = self.builder.build_query(parsed_query)
        
        # Step 4: Mock the ES response
        self.mock_agent_tools.search_by_author.return_value = {
            "session_id": "suggestions123",
            "total_results": 25,
            "sample_results": []
        }
        
        # Step 5: Execute the query
        query_result = self.builder.execute_query(query_spec)
        
        # Step 6: Format the response with suggestions
        formatted_response = self.formatter.format_response(
            query_result, 
            include_suggestions=True
        )
        
        # Verify the complete flow
        assert formatted_response["type"] == "text"
        assert "25" in formatted_response["content"]
        assert "Erik Lind" in formatted_response["content"]
        assert "suggestions" in formatted_response
        assert len(formatted_response["suggestions"]) > 0
    
    def test_confidence_based_parsing(self):
        """Test that low confidence queries are handled appropriately."""
        # Step 1: Ambiguous query with low confidence
        user_query = "Tell me about quantum"
        
        # Step 2: Parse the query
        parsed_query = self.parser.parse(user_query)
        
        # Verify confidence is low
        assert parsed_query.confidence < 0.7
        
        # Step 3: Build the query (should still work)
        query_spec = self.builder.build_query(parsed_query)
        
        # Step 4: Mock the ES response
        self.mock_agent_tools.search_publications.return_value = {
            "session_id": "ambiguous123",
            "total_results": 150,
            "sample_results": [
                {
                    "title": "Quantum Physics Overview",
                    "year": 2023,
                    "authors": ["Dr. Physics"],
                    "abstract": "An overview of quantum physics concepts..."
                }
            ]
        }
        
        # Step 5: Execute the query
        query_result = self.builder.execute_query(query_spec)
        
        # Step 6: Format the response
        formatted_response = self.formatter.format_response(query_result)
        
        # Verify the complete flow still works
        assert formatted_response["type"] == "list"
        assert len(formatted_response["items"]) == 1
        assert "Quantum Physics Overview" in formatted_response["items"][0]["title"]


def test_chat_agent_components_exist():
    """Test that all required components exist and can be imported."""
    # This test ensures our components are properly structured
    from chat_parser import ChatParser, QueryIntent, ParsedQuery
    from query_builder import QueryBuilder, QueryBuilderError
    from response_formatter import ResponseFormatter
    
    # Test basic instantiation
    parser = ChatParser()
    formatter = ResponseFormatter()
    
    # Test with mock agent tools
    mock_tools = Mock()
    builder = QueryBuilder(mock_tools)
    
    assert parser is not None
    assert builder is not None
    assert formatter is not None