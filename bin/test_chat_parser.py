"""
Test suite for chat message parsing functionality.
Tests the conversion of natural language queries to structured queries.
"""

import pytest
from chat_parser import ChatParser, QueryIntent, ParsedQuery


class TestChatParser:
    """Test the ChatParser class that converts natural language to structured queries."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = ChatParser()
    
    def test_parse_author_count_query(self):
        """Test parsing 'How many articles has [author] published?' queries."""
        query = "How many articles has Christian Fager published?"
        result = self.parser.parse(query)
        
        assert result.intent == QueryIntent.COUNT
        assert result.entity_type == "author"
        assert result.author_name == "Christian Fager"
        assert result.filters == {}
        assert result.confidence > 0.8
    
    def test_parse_author_list_query(self):
        """Test parsing 'List all articles by [author]' queries."""
        query = "List all articles that Anna Dubois has published"
        result = self.parser.parse(query)
        
        assert result.intent == QueryIntent.LIST
        assert result.entity_type == "author"
        assert result.author_name == "Anna Dubois"
        assert result.filters == {}
    
    def test_parse_journal_year_query(self):
        """Test parsing journal/year combination queries."""
        query = "How many publications have been published in Nature in 2023?"
        result = self.parser.parse(query)
        
        assert result.intent == QueryIntent.COUNT
        assert result.entity_type == "publication"
        assert result.journal_name == "Nature"
        assert result.filters == {"year": 2023}
    
    def test_parse_keywords_yearly_query(self):
        """Test parsing keyword trend queries."""
        query = "What are the top 10 keywords on publications per year from 2020 to 2024?"
        result = self.parser.parse(query)
        
        assert result.intent == QueryIntent.STATS
        assert result.entity_type == "keywords"
        assert result.limit == 10
        assert result.filters == {"year_range": {"gte": 2020, "lte": 2024}}
        assert result.group_by == "year"
    
    def test_parse_author_with_year_filter(self):
        """Test parsing author queries with year filters."""
        query = "List articles by Erik Lind from 2022 to 2024"
        result = self.parser.parse(query)
        
        assert result.intent == QueryIntent.LIST
        assert result.entity_type == "author"
        assert result.author_name == "Erik Lind"
        assert result.filters == {"year_range": {"gte": 2022, "lte": 2024}}
    
    def test_parse_simple_search_query(self):
        """Test parsing simple search queries."""
        query = "Find papers about machine learning"
        result = self.parser.parse(query)
        
        assert result.intent == QueryIntent.SEARCH
        assert result.entity_type == "publication"
        assert result.search_terms == "machine learning"
        assert result.filters == {}
    
    def test_parse_ambiguous_query(self):
        """Test handling of ambiguous queries."""
        query = "Tell me about quantum"
        result = self.parser.parse(query)
        
        assert result.intent == QueryIntent.SEARCH
        assert result.confidence < 0.7  # Low confidence for ambiguous queries
    
    def test_parse_invalid_query(self):
        """Test handling of invalid or nonsensical queries."""
        query = "asdfghjkl random nonsense"
        result = self.parser.parse(query)
        
        assert result.intent == QueryIntent.UNKNOWN
        assert result.confidence < 0.5
    
    def test_extract_year_range(self):
        """Test year range extraction from various formats."""
        test_cases = [
            ("from 2020 to 2023", {"gte": 2020, "lte": 2023}),
            ("between 2021 and 2024", {"gte": 2021, "lte": 2024}),
            ("since 2022", {"gte": 2022}),
            ("before 2020", {"lte": 2020}),
            ("in 2023", {"gte": 2023, "lte": 2023}),
        ]
        
        for query_text, expected_range in test_cases:
            extracted = self.parser._extract_year_range(query_text)
            assert extracted == expected_range
    
    def test_extract_author_name(self):
        """Test author name extraction from various query formats."""
        test_cases = [
            ("How many articles has Christian Fager published?", "Christian Fager"),
            ("List all papers by Anna Dubois", "Anna Dubois"),
            ("Publications from Erik Lind", "Erik Lind"),
            ("Find work by Maria Garcia-Lopez", "Maria Garcia-Lopez"),
        ]
        
        for query, expected_name in test_cases:
            extracted = self.parser._extract_author_name(query)
            assert extracted == expected_name
    
    def test_intent_classification(self):
        """Test intent classification for various query types."""
        test_cases = [
            ("How many", QueryIntent.COUNT),
            ("List all", QueryIntent.LIST),
            ("Show me", QueryIntent.LIST),
            ("What are the top", QueryIntent.STATS),
            ("Find papers", QueryIntent.SEARCH),
            ("Search for", QueryIntent.SEARCH),
        ]
        
        for query_start, expected_intent in test_cases:
            intent = self.parser._classify_intent(query_start + " articles about AI")
            assert intent == expected_intent


class TestQueryIntent:
    """Test the QueryIntent enum."""
    
    def test_query_intent_values(self):
        """Test that all required intent values exist."""
        assert QueryIntent.COUNT
        assert QueryIntent.LIST
        assert QueryIntent.SEARCH
        assert QueryIntent.STATS
        assert QueryIntent.UNKNOWN


class TestParsedQuery:
    """Test the ParsedQuery data structure."""
    
    def test_parsed_query_creation(self):
        """Test creating a ParsedQuery object."""
        query = ParsedQuery(
            intent=QueryIntent.COUNT,
            entity_type="author",
            author_name="Test Author",
            confidence=0.9
        )
        
        assert query.intent == QueryIntent.COUNT
        assert query.entity_type == "author"
        assert query.author_name == "Test Author"
        assert query.confidence == 0.9
        assert query.filters == {}  # Default empty dict
    
    def test_parsed_query_defaults(self):
        """Test ParsedQuery with default values."""
        query = ParsedQuery(intent=QueryIntent.SEARCH)
        
        assert query.intent == QueryIntent.SEARCH
        assert query.entity_type is None
        assert query.filters == {}
        assert query.confidence == 0.5  # Default confidence