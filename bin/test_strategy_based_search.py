"""
Test suite for strategy-based author search functionality.
"""

import pytest
from unittest.mock import Mock, patch
from agent_tools import _detect_author_strategy, search_by_author
from query_builder import QueryBuilder
from search_session import SearchSession
from chat_parser import ChatParser, QueryIntent
import agent_tools


class TestStrategyDetection:
    """Test the strategy detection logic."""
    
    def test_detect_full_name_strategy(self):
        """Test detection of full names that should use exact strategy."""
        test_cases = [
            ("Christian Fager", "exact"),
            ("Anna Dubois", "exact"),
            ("Erik Lind", "exact"),
            ("Maria Garcia-Lopez", "exact"),
            ("John Smith Jr", "exact"),
        ]
        
        for name, expected_strategy in test_cases:
            strategy = _detect_author_strategy(name)
            assert strategy == expected_strategy, f"Expected {expected_strategy} for '{name}', got {strategy}"
    
    def test_detect_partial_name_strategy(self):
        """Test detection of partial names that should use partial strategy."""
        test_cases = [
            ("Fager", "partial"),
            ("Smith", "partial"),
            ("Anderson", "partial"),
            ("Johnson", "partial"),
        ]
        
        for name, expected_strategy in test_cases:
            strategy = _detect_author_strategy(name)
            assert strategy == expected_strategy, f"Expected {expected_strategy} for '{name}', got {strategy}"
    
    def test_detect_fuzzy_strategy(self):
        """Test detection of names that should use fuzzy strategy."""
        test_cases = [
            ("fager", "fuzzy"),  # lowercase
            ("christian fager", "fuzzy"),  # lowercase
            ("A", "fuzzy"),  # too short
            ("ab", "fuzzy"),  # too short
            ("", "partial"),  # empty defaults to partial
            ("   ", "partial"),  # whitespace defaults to partial
        ]
        
        for name, expected_strategy in test_cases:
            strategy = _detect_author_strategy(name)
            assert strategy == expected_strategy, f"Expected {expected_strategy} for '{name}', got {strategy}"


class TestQueryBuilderStrategy:
    """Test strategy integration in QueryBuilder."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_agent_tools = Mock()
        self.mock_agent_tools.search_by_author = Mock()
        self.mock_agent_tools.search_publications = Mock()
        self.mock_agent_tools.get_field_statistics = Mock()
        self.builder = QueryBuilder(self.mock_agent_tools)
    
    def test_query_builder_strategy_detection(self):
        """Test that QueryBuilder correctly detects and uses strategies."""
        from chat_parser import ParsedQuery, QueryIntent
        
        # Test full name -> exact strategy
        parsed_query = ParsedQuery(
            intent=QueryIntent.COUNT,
            entity_type="author",
            author_name="Christian Fager",
            filters={}
        )
        
        query_spec = self.builder.build_query(parsed_query)
        
        assert query_spec["function"] == "search_by_author"
        assert query_spec["args"]["author_name"] == "Christian Fager"
        assert query_spec["args"]["strategy"] == "exact"
    
    def test_query_builder_single_name_strategy(self):
        """Test strategy detection for single names."""
        from chat_parser import ParsedQuery, QueryIntent
        
        # Test single name -> partial strategy
        parsed_query = ParsedQuery(
            intent=QueryIntent.COUNT,
            entity_type="author",
            author_name="Fager",
            filters={}
        )
        
        query_spec = self.builder.build_query(parsed_query)
        
        assert query_spec["function"] == "search_by_author"
        assert query_spec["args"]["author_name"] == "Fager"
        assert query_spec["args"]["strategy"] == "partial"
    
    def test_query_builder_with_year_filter(self):
        """Test strategy detection with additional filters."""
        from chat_parser import ParsedQuery, QueryIntent
        
        parsed_query = ParsedQuery(
            intent=QueryIntent.LIST,
            entity_type="author",
            author_name="Erik Lind",
            filters={"year_range": {"gte": 2020, "lte": 2024}}
        )
        
        query_spec = self.builder.build_query(parsed_query)
        
        assert query_spec["function"] == "search_by_author"
        assert query_spec["args"]["author_name"] == "Erik Lind"
        assert query_spec["args"]["strategy"] == "exact"
        assert query_spec["args"]["year_range"] == {"gte": 2020, "lte": 2024}


class TestSearchSessionStrategy:
    """Test strategy implementation in SearchSession."""
    
    def test_build_author_query_exact(self):
        """Test exact strategy query building."""
        session = SearchSession(Mock(), "test-index")
        
        query = session._build_author_query("Christian Fager", "exact")
        
        expected = {
            "match_phrase": {
                "Persons.PersonData.DisplayName": "Christian Fager"
            }
        }
        assert query == expected
    
    def test_build_author_query_partial(self):
        """Test partial strategy query building."""
        session = SearchSession(Mock(), "test-index")
        
        query = session._build_author_query("Fager", "partial")
        
        expected = {
            "match": {
                "Persons.PersonData.DisplayName": "Fager"
            }
        }
        assert query == expected
    
    def test_build_author_query_fuzzy(self):
        """Test fuzzy strategy query building."""
        session = SearchSession(Mock(), "test-index")
        
        query = session._build_author_query("fager", "fuzzy")
        
        expected = {
            "fuzzy": {
                "Persons.PersonData.DisplayName": {
                    "value": "fager",
                    "fuzziness": "AUTO"
                }
            }
        }
        assert query == expected
    
    def test_build_author_query_default(self):
        """Test default strategy query building."""
        session = SearchSession(Mock(), "test-index")
        
        query = session._build_author_query("Test Author", "unknown")
        
        expected = {
            "match": {
                "Persons.PersonData.DisplayName": "Test Author"
            }
        }
        assert query == expected


class TestChatParserIntegration:
    """Test integration between ChatParser and strategy-based search."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = ChatParser()
        self.mock_agent_tools = Mock()
        self.mock_agent_tools.search_by_author = Mock()
        self.mock_agent_tools.search_publications = Mock()
        self.mock_agent_tools.get_field_statistics = Mock()
        self.builder = QueryBuilder(self.mock_agent_tools)
    
    def test_full_name_query_uses_exact_strategy(self):
        """Test that full name queries use exact strategy."""
        query = "How many papers has Christian Fager published?"
        
        # Parse the query
        parsed = self.parser.parse(query)
        
        # Build the query spec
        query_spec = self.builder.build_query(parsed)
        
        # Verify it uses exact strategy
        assert query_spec["args"]["strategy"] == "exact"
        assert query_spec["args"]["author_name"] == "Christian Fager"
    
    def test_single_name_query_uses_partial_strategy(self):
        """Test that single name queries use partial strategy."""
        query = "How many papers has Fager published?"
        
        # Parse the query
        parsed = self.parser.parse(query)
        
        # Build the query spec
        query_spec = self.builder.build_query(parsed)
        
        # Verify it uses partial strategy
        assert query_spec["args"]["strategy"] == "partial"
        assert query_spec["args"]["author_name"] == "Fager"
    
    def test_lowercase_name_query_uses_fuzzy_strategy(self):
        """Test that lowercase names use fuzzy strategy."""
        # Create a parsed query directly since ChatParser doesn't extract lowercase names
        from chat_parser import ParsedQuery, QueryIntent
        
        parsed = ParsedQuery(
            intent=QueryIntent.COUNT,
            entity_type="author",
            author_name="fager",  # lowercase name
            filters={}
        )
        
        # Build the query spec
        query_spec = self.builder.build_query(parsed)
        
        # Verify it uses fuzzy strategy
        assert query_spec["args"]["strategy"] == "fuzzy"
        assert query_spec["args"]["author_name"] == "fager"


class TestAgentToolsStrategy:
    """Test strategy parameter in agent_tools functions."""
    
    def test_search_by_author_auto_strategy(self):
        """Test search_by_author with auto strategy detection."""
        with patch('agent_tools.search_publications') as mock_search:
            mock_search.return_value = {"session_id": "test", "total_results": 100}
            
            # Test auto strategy detection
            result = search_by_author("Christian Fager")
            
            # Verify the call included the detected strategy
            mock_search.assert_called_once_with(
                filters={"author": "Christian Fager", "author_strategy": "exact"}
            )
    
    def test_search_by_author_manual_strategy(self):
        """Test search_by_author with manually specified strategy."""
        with patch('agent_tools.search_publications') as mock_search:
            mock_search.return_value = {"session_id": "test", "total_results": 100}
            
            # Test manual strategy override
            result = search_by_author("Christian Fager", strategy="partial")
            
            # Verify the call used the manual strategy
            mock_search.assert_called_once_with(
                filters={"author": "Christian Fager", "author_strategy": "partial"}
            )
    
    def test_search_by_author_with_year_range(self):
        """Test search_by_author with year range and strategy."""
        with patch('agent_tools.search_publications') as mock_search:
            mock_search.return_value = {"session_id": "test", "total_results": 50}
            
            # Test with year range
            result = search_by_author(
                "Fager", 
                year_range={"gte": 2020, "lte": 2024}, 
                strategy="fuzzy"
            )
            
            # Verify the call included both strategy and year range
            mock_search.assert_called_once_with(
                filters={
                    "author": "Fager", 
                    "author_strategy": "fuzzy",
                    "year_range": {"gte": 2020, "lte": 2024}
                }
            )


class TestStrategyEffectiveness:
    """Test that strategies produce the expected behavioral differences."""
    
    def test_strategy_comparison_concept(self):
        """Test the conceptual difference between strategies."""
        # This is a conceptual test - in practice, this would need real ES data
        
        # Exact strategy should be most precise for full names
        # Partial strategy should be broader for single names
        # Fuzzy strategy should be most tolerant of typos
        
        strategies = ["exact", "partial", "fuzzy"]
        
        for strategy in strategies:
            # Test that each strategy produces different query structures
            session = SearchSession(Mock(), "test-index")
            query = session._build_author_query("Test Author", strategy)
            
            # Each strategy should produce a different ES query type
            if strategy == "exact":
                assert "match_phrase" in query
            elif strategy == "partial":
                assert "match" in query
            elif strategy == "fuzzy":
                assert "fuzzy" in query
    
    def test_strategy_parameter_handling(self):
        """Test that strategy parameters are handled correctly throughout the pipeline."""
        # Test that strategy is preserved through the filter chain
        
        # Mock a SearchSession
        session = SearchSession(Mock(), "test-index")
        
        # Test filter processing with author_strategy
        filters = {
            "author": "Test Author",
            "author_strategy": "exact",
            "year": 2023
        }
        
        # This should not raise an error and should skip the author_strategy filter
        # while using it for the author query
        try:
            session._build_query(None, filters)
        except Exception as e:
            pytest.fail(f"Filter processing failed: {e}")


def test_backwards_compatibility():
    """Test that existing code still works without strategy parameter."""
    with patch('agent_tools.search_publications') as mock_search:
        mock_search.return_value = {"session_id": "test", "total_results": 100}
        
        # Test that calling without strategy still works
        result = search_by_author("Christian Fager")
        
        # Should default to auto-detected strategy
        call_args = mock_search.call_args[1]["filters"]
        assert "author_strategy" in call_args
        assert call_args["author_strategy"] == "exact"  # Auto-detected for full name