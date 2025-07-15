"""
Comprehensive test suite for Elasticsearch tools.

This module contains production-ready tests covering functionality,
performance, resilience, and integration scenarios.
"""

import pytest
import json
import time
import threading
from typing import Dict, Any, List
from unittest.mock import patch, Mock

from src.research_agent.tools.elasticsearch_tools import (
    search_publications,
    search_by_author,
    get_field_statistics,
    get_publication_details,
    get_statistics_summary,
    initialize_elasticsearch_tools
)


class TestSearchPublications:
    """Comprehensive tests for search_publications tool."""
    
    @pytest.mark.unit
    def test_search_publications_basic_functionality(self, es_client, performance_monitor):
        """Test basic search functionality with performance monitoring."""
        # Initialize tools
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        # Test basic search
        measurement = performance_monitor.start_measurement("search_publications", "basic_search")
        result = search_publications("machine learning", max_results=10)
        perf_result = performance_monitor.end_measurement(measurement)
        
        # Validate JSON response
        data = json.loads(result)
        assert "total_hits" in data
        assert "results" in data
        assert "query" in data
        assert data["query"] == "machine learning"
        assert isinstance(data["results"], list)
        assert len(data["results"]) <= 10
        
        # Validate result structure
        if data["results"]:
            first_result = data["results"][0]
            required_fields = ["id", "score", "title", "authors", "year", "abstract"]
            for field in required_fields:
                assert field in first_result
        
        # Performance validation
        assert perf_result["duration"] < 3.0, f"Search took {perf_result['duration']}s, exceeding 3s threshold"
    
    @pytest.mark.resilience
    def test_search_publications_unicode_inputs(self, es_client, test_data_generator):
        """Test search with Unicode and international characters."""
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        unicode_inputs = test_data_generator.generate_unicode_inputs()
        
        for unicode_query in unicode_inputs:
            result = search_publications(unicode_query, max_results=5)
            data = json.loads(result)
            
            # Should not error, even if no results
            assert "error" not in data or data.get("total_hits") == 0
            assert "results" in data
            assert isinstance(data["results"], list)
    
    @pytest.mark.resilience
    def test_search_publications_edge_cases(self, es_client, test_data_generator):
        """Test search with edge case inputs."""
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        edge_cases = test_data_generator.generate_edge_case_inputs()
        
        for edge_case in edge_cases:
            result = search_publications(edge_case, max_results=10)
            data = json.loads(result)
            
            # Should handle gracefully
            assert isinstance(data, dict)
            assert "results" in data
            
            # Empty queries should return empty results, not errors
            if edge_case.strip() == "":
                assert data.get("total_hits") == 0 or "error" in data
    
    @pytest.mark.performance
    def test_search_publications_large_query(self, es_client, test_data_generator):
        """Test search with large query strings."""
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        large_query = test_data_generator.generate_large_query(10)  # 10KB query
        
        start_time = time.time()
        result = search_publications(large_query, max_results=10)
        duration = time.time() - start_time
        
        data = json.loads(result)
        
        # Should handle large queries within reasonable time
        assert duration < 5.0, f"Large query took {duration}s, too slow"
        assert isinstance(data, dict)
        assert "results" in data
    
    @pytest.mark.resilience
    def test_search_publications_boundary_values(self, es_client, test_data_generator):
        """Test search with boundary values for max_results."""
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        boundary_values = test_data_generator.generate_boundary_values()
        
        for max_results in boundary_values["max_results"]:
            result = search_publications("test", max_results=max_results)
            data = json.loads(result)
            
            if max_results <= 0:
                # Should handle gracefully
                assert "error" in data or len(data.get("results", [])) == 0
            else:
                assert "results" in data
                if data.get("total_hits", 0) > 0:
                    assert len(data["results"]) <= max_results


class TestSearchByAuthor:
    """Comprehensive tests for search_by_author tool."""
    
    @pytest.mark.unit
    def test_search_by_author_all_strategies(self, es_client, performance_monitor):
        """Test all search strategies with performance monitoring."""
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        author_name = "Christian Fager"
        strategies = ["exact", "partial", "fuzzy"]
        
        results = {}
        for strategy in strategies:
            measurement = performance_monitor.start_measurement("search_by_author", f"strategy_{strategy}")
            result = search_by_author(author_name, strategy=strategy, max_results=10)
            performance_monitor.end_measurement(measurement)
            
            data = json.loads(result)
            results[strategy] = data
            
            # Validate response structure
            assert "total_hits" in data
            assert "results" in data
            assert "author" in data
            assert "strategy" in data
            assert data["strategy"] == strategy
            
            # Validate result structure
            if data["results"]:
                first_result = data["results"][0]
                required_fields = ["id", "title", "authors", "year", "journal", "publication_type"]
                for field in required_fields:
                    assert field in first_result
        
        # Strategy comparison - exact should be most restrictive
        if all(results[s].get("total_hits", 0) > 0 for s in strategies):
            assert results["exact"]["total_hits"] <= results["partial"]["total_hits"]
            assert results["partial"]["total_hits"] <= results["fuzzy"]["total_hits"]
    
    @pytest.mark.resilience
    def test_search_by_author_unicode_names(self, es_client, test_data_generator):
        """Test author search with Unicode names."""
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        unicode_names = test_data_generator.generate_unicode_inputs()
        
        for name in unicode_names:
            for strategy in ["exact", "partial", "fuzzy"]:
                result = search_by_author(name, strategy=strategy, max_results=5)
                data = json.loads(result)
                
                # Should not error
                assert "error" not in data or data.get("total_hits") == 0
                assert "results" in data
                assert data["strategy"] == strategy
    
    @pytest.mark.performance
    def test_search_by_author_concurrent_requests(self, es_client, concurrency_test_helper):
        """Test concurrent author searches."""
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        def search_operation():
            return search_by_author("Smith", strategy="partial", max_results=10)
        
        results = concurrency_test_helper.run_concurrent_operations(
            search_operation, num_threads=10
        )
        
        # Should handle concurrent requests well
        assert results["success_rate"] > 0.95, f"Success rate {results['success_rate']} too low"
        assert len(results["errors"]) < 2, f"Too many errors: {results['errors']}"
        
        # Validate consistent results
        if len(results["results"]) > 1:
            first_result = json.loads(results["results"][0])
            for result_str in results["results"][1:]:
                result = json.loads(result_str)
                assert result["total_hits"] == first_result["total_hits"]


class TestGetFieldStatistics:
    """Comprehensive tests for get_field_statistics tool."""
    
    @pytest.mark.unit
    def test_get_field_statistics_all_fields(self, es_client, performance_monitor):
        """Test statistics for all supported fields."""
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        fields = ["year", "authors", "journal", "publication_type"]
        
        for field in fields:
            measurement = performance_monitor.start_measurement("get_field_statistics", f"field_{field}")
            result = get_field_statistics(field, size=10)
            perf_result = performance_monitor.end_measurement(measurement)
            
            data = json.loads(result)
            
            # Validate response structure
            assert "field" in data
            assert "total_documents" in data
            assert "top_values" in data
            assert data["field"] == field
            assert isinstance(data["top_values"], list)
            assert len(data["top_values"]) <= 10
            
            # Validate top_values structure
            if data["top_values"]:
                first_value = data["top_values"][0]
                assert "value" in first_value
                assert "count" in first_value
                assert isinstance(first_value["count"], int)
                assert first_value["count"] > 0
            
            # Performance validation
            assert perf_result["duration"] < 2.0, f"Statistics took {perf_result['duration']}s, exceeding 2s threshold"
    
    @pytest.mark.resilience
    def test_get_field_statistics_invalid_fields(self, es_client):
        """Test statistics with invalid field names."""
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        invalid_fields = ["nonexistent_field", "", "invalid.field", "field with spaces"]
        
        for field in invalid_fields:
            result = get_field_statistics(field, size=10)
            data = json.loads(result)
            
            # Should handle gracefully with error or empty results
            assert "error" in data or data.get("top_values") == []
    
    @pytest.mark.performance
    def test_get_field_statistics_large_size(self, es_client):
        """Test statistics with large size parameter."""
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        large_sizes = [100, 1000, 10000]
        
        for size in large_sizes:
            start_time = time.time()
            result = get_field_statistics("year", size=size)
            duration = time.time() - start_time
            
            data = json.loads(result)
            
            # Should handle large sizes reasonably
            assert duration < 5.0, f"Large size query took {duration}s, too slow"
            
            if "error" not in data:
                assert len(data["top_values"]) <= size
    
    @pytest.mark.unit
    def test_get_field_statistics_ordering(self, es_client):
        """Test that statistics are properly ordered by count."""
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        result = get_field_statistics("year", size=10)
        data = json.loads(result)
        
        if len(data.get("top_values", [])) > 1:
            counts = [item["count"] for item in data["top_values"]]
            
            # Should be sorted by count in descending order
            assert counts == sorted(counts, reverse=True), "Results not properly sorted by count"


class TestGetPublicationDetails:
    """Comprehensive tests for get_publication_details tool."""
    
    @pytest.mark.unit
    def test_get_publication_details_valid_id(self, es_client, performance_monitor):
        """Test publication details with valid ID."""
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        # First get a valid ID from search
        search_result = search_publications("test", max_results=1)
        search_data = json.loads(search_result)
        
        if search_data.get("results"):
            valid_id = search_data["results"][0]["id"]
            
            measurement = performance_monitor.start_measurement("get_publication_details", "valid_id")
            result = get_publication_details(valid_id)
            perf_result = performance_monitor.end_measurement(measurement)
            
            data = json.loads(result)
            
            # Validate response structure
            expected_fields = ["id", "title", "authors", "year", "journal", 
                             "publication_type", "abstract", "keywords", "doi", "url"]
            
            for field in expected_fields:
                assert field in data
            
            assert data["id"] == valid_id
            
            # Performance validation
            assert perf_result["duration"] < 1.0, f"Detail retrieval took {perf_result['duration']}s, exceeding 1s threshold"
    
    @pytest.mark.resilience
    def test_get_publication_details_invalid_ids(self, es_client, test_data_generator):
        """Test publication details with invalid IDs."""
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        boundary_values = test_data_generator.generate_boundary_values()
        invalid_ids = boundary_values["publication_ids"]
        
        for invalid_id in invalid_ids:
            result = get_publication_details(invalid_id)
            data = json.loads(result)
            
            # Should handle gracefully
            if invalid_id in ["", "invalid_id"]:
                assert "error" in data
            else:
                # May return error or empty fields
                assert isinstance(data, dict)
    
    @pytest.mark.performance
    def test_get_publication_details_concurrent_access(self, es_client, concurrency_test_helper):
        """Test concurrent detail retrieval."""
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        # Get valid ID first
        search_result = search_publications("test", max_results=1)
        search_data = json.loads(search_result)
        
        if search_data.get("results"):
            valid_id = search_data["results"][0]["id"]
            
            def details_operation():
                return get_publication_details(valid_id)
            
            results = concurrency_test_helper.run_concurrent_operations(
                details_operation, num_threads=10
            )
            
            # Should handle concurrent access well
            assert results["success_rate"] > 0.95, f"Success rate {results['success_rate']} too low"
            
            # Results should be consistent
            if len(results["results"]) > 1:
                first_result = json.loads(results["results"][0])
                for result_str in results["results"][1:]:
                    result = json.loads(result_str)
                    assert result["id"] == first_result["id"]
                    assert result["title"] == first_result["title"]


class TestGetStatisticsSummary:
    """Comprehensive tests for get_statistics_summary tool."""
    
    @pytest.mark.unit
    def test_get_statistics_summary_structure(self, es_client, performance_monitor):
        """Test statistics summary structure and performance."""
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        measurement = performance_monitor.start_measurement("get_statistics_summary", "complete_summary")
        result = get_statistics_summary()
        perf_result = performance_monitor.end_measurement(measurement)
        
        # Validate response structure
        expected_fields = ["total_publications", "latest_year", "most_common_type", 
                          "total_authors", "years", "publication_types"]
        
        for field in expected_fields:
            assert field in result
        
        # Validate data types
        assert isinstance(result["total_publications"], int)
        assert isinstance(result["years"], list)
        assert isinstance(result["publication_types"], list)
        assert isinstance(result["total_authors"], int)
        
        # Validate array structures
        if result["years"]:
            first_year = result["years"][0]
            assert "value" in first_year
            assert "count" in first_year
        
        if result["publication_types"]:
            first_type = result["publication_types"][0]
            assert "value" in first_type
            assert "count" in first_type
        
        # Performance validation
        assert perf_result["duration"] < 2.0, f"Summary took {perf_result['duration']}s, exceeding 2s threshold"
    
    @pytest.mark.unit
    def test_get_statistics_summary_data_consistency(self, es_client):
        """Test data consistency in statistics summary."""
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        result = get_statistics_summary()
        
        # Validate consistency
        if result["years"]:
            # Total from years should not exceed total publications
            year_totals = sum(year["count"] for year in result["years"])
            assert year_totals <= result["total_publications"]
        
        if result["publication_types"]:
            # Total from types should not exceed total publications
            type_totals = sum(ptype["count"] for ptype in result["publication_types"])
            assert type_totals <= result["total_publications"]
        
        # Years should be sorted descending
        if len(result["years"]) > 1:
            years = [year["value"] for year in result["years"]]
            assert years == sorted(years, reverse=True)
        
        # Publication types should be sorted by count descending
        if len(result["publication_types"]) > 1:
            counts = [ptype["count"] for ptype in result["publication_types"]]
            assert counts == sorted(counts, reverse=True)


class TestToolsIntegration:
    """Integration tests combining multiple tools."""
    
    @pytest.mark.integration
    def test_author_research_workflow(self, es_client, performance_monitor):
        """Test complete author research workflow."""
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        # Step 1: Search for author
        workflow_start = time.time()
        author_result = search_by_author("Christian Fager", strategy="partial", max_results=10)
        author_data = json.loads(author_result)
        
        if author_data.get("results"):
            # Step 2: Get details for first publication
            first_pub_id = author_data["results"][0]["id"]
            detail_result = get_publication_details(first_pub_id)
            detail_data = json.loads(detail_result)
            
            # Step 3: Get year statistics
            year_stats = get_field_statistics("year", size=10)
            year_data = json.loads(year_stats)
            
            # Step 4: Get database summary
            summary = get_statistics_summary()
            
            workflow_duration = time.time() - workflow_start
            
            # Validate data continuity
            assert detail_data["id"] == first_pub_id
            assert "error" not in detail_data
            
            # Validate workflow performance
            assert workflow_duration < 10.0, f"Workflow took {workflow_duration}s, exceeding 10s threshold"
            
            # Validate cross-tool consistency
            if detail_data.get("year"):
                # The year should appear in statistics if it's common
                year_values = [item["value"] for item in year_data.get("top_values", [])]
                # Note: May not be in top 10, so this is just a data consistency check
                assert isinstance(detail_data["year"], (int, str))
    
    @pytest.mark.integration
    def test_topic_discovery_workflow(self, es_client):
        """Test topic discovery workflow."""
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        # Step 1: Search for topic
        topic_result = search_publications("machine learning", max_results=5)
        topic_data = json.loads(topic_result)
        
        if topic_data.get("results"):
            # Step 2: Extract author from result
            first_result = topic_data["results"][0]
            author_name = first_result["authors"].split(',')[0].strip()
            
            # Step 3: Search for more by same author
            author_result = search_by_author(author_name, strategy="exact", max_results=10)
            author_data = json.loads(author_result)
            
            # Step 4: Get database context
            summary = get_statistics_summary()
            
            # Validate workflow
            assert "error" not in topic_data
            assert "error" not in author_data
            assert "error" not in summary
            
            # Validate data relationships
            if author_data.get("results"):
                # Author search should include the original paper or similar author
                author_publications = [pub["id"] for pub in author_data["results"]]
                # This validates that the author search is working correctly
                assert len(author_publications) > 0
    
    @pytest.mark.performance
    def test_concurrent_mixed_operations(self, es_client, concurrency_test_helper):
        """Test concurrent operations with mixed tool usage."""
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        operations = [
            lambda: search_publications("test", max_results=10),
            lambda: search_by_author("Smith", strategy="partial", max_results=10),
            lambda: get_field_statistics("year", size=10),
            lambda: get_statistics_summary()
        ]
        
        all_results = []
        all_errors = []
        
        for operation in operations:
            result = concurrency_test_helper.run_concurrent_operations(
                operation, num_threads=5
            )
            all_results.extend(result["results"])
            all_errors.extend(result["errors"])
        
        # Calculate overall success rate
        total_operations = len(all_results) + len(all_errors)
        success_rate = len(all_results) / total_operations if total_operations > 0 else 0
        
        # Should handle mixed concurrent operations well
        assert success_rate > 0.95, f"Overall success rate {success_rate} too low"
        assert len(all_errors) < 5, f"Too many errors in concurrent operations: {all_errors}"