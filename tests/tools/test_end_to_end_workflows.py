"""
End-to-End Workflow Tests for Elasticsearch Tools

This module implements comprehensive end-to-end testing of user workflows
that chain multiple tools together to validate data continuity, performance,
and real-world usage scenarios.
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


class TestEndToEndWorkflows:
    """Test complete user workflows with multiple tool interactions."""
    
    def test_author_research_pipeline(self, es_tools, performance_monitor, performance_thresholds):
        """
        Test complete author research workflow:
        1. Search for publications by author
        2. Get details for top results
        3. Analyze publication patterns
        4. Validate data continuity throughout
        """
        # Start performance measurement
        measurement = performance_monitor.start_measurement("author_research_pipeline", "full_workflow")
        
        try:
            # Step 1: Search for author publications
            author_name = "Christian Fager"
            author_results = es_tools.search_by_author(author_name, "partial")
            
            # Validate author search results
            assert isinstance(author_results, dict)
            assert "total_hits" in author_results
            assert "results" in author_results
            assert author_results["author"] == author_name
            
            if author_results["total_hits"] > 0:
                # Step 2: Get detailed information for first publication
                first_pub_id = author_results["results"][0]["id"]
                pub_details = es_tools.get_publication_details(first_pub_id)
                
                # Validate data continuity
                assert pub_details["id"] == first_pub_id
                assert isinstance(pub_details, dict)
                assert "title" in pub_details
                assert "authors" in pub_details
                
                # Step 3: Analyze publication years for this author
                year_stats = es_tools.get_field_statistics("year", size=20)
                assert isinstance(year_stats, dict)
                assert "top_values" in year_stats
                
                # Step 4: Verify the author's publication year appears in stats
                author_year = pub_details["year"]
                year_values = [int(item["value"]) for item in year_stats["top_values"]]
                assert author_year in year_values, f"Author's publication year {author_year} not found in year statistics"
                
                # Step 5: Cross-validate with database summary
                db_summary = es_tools.get_database_summary()
                assert isinstance(db_summary, dict)
                assert "total_publications" in db_summary
                assert "years" in db_summary
                
                # Validate year consistency
                summary_years = [item["value"] for item in db_summary["years"]]
                assert author_year in summary_years, f"Author year {author_year} not in database summary"
            
            # End measurement
            result = performance_monitor.end_measurement(measurement)
            
            # Validate performance
            assert result["duration"] < performance_thresholds["end_to_end_flow"]["max_duration"]
            assert result["memory_delta"] < performance_thresholds["end_to_end_flow"]["max_memory_delta"]
            
        except Exception as e:
            performance_monitor.end_measurement(measurement)
            raise e
        
        workflow_data["publication_details"] = {
            "duration": step2_duration,
            "publication_id": first_publication_id,
            "has_abstract": bool(detail_data.get("abstract", "").strip()),
            "has_keywords": bool(detail_data.get("keywords", "").strip())
        }
        
        # Validate Step 2 - Data continuity
        assert "error" not in detail_data, f"Publication details failed: {detail_data}"
        assert detail_data["id"] == first_publication_id, "ID mismatch between search and details"
        assert step2_duration < 1.0, f"Publication details took {step2_duration}s, exceeding 1s threshold"
        
        # Step 3: Analyze publication years for context
        print("Step 3: Analyzing publication years...")
        step3_start = time.time()
        year_stats_result = get_field_statistics("year", size=10)
        year_stats_data = json.loads(year_stats_result)
        step3_duration = time.time() - step3_start
        
        workflow_data["year_analysis"] = {
            "duration": step3_duration,
            "top_years_count": len(year_stats_data.get("top_values", [])),
            "total_documents": year_stats_data.get("total_documents", 0)
        }
        
        # Validate Step 3
        assert "error" not in year_stats_data, f"Year statistics failed: {year_stats_data}"
        assert step3_duration < 2.0, f"Year statistics took {step3_duration}s, exceeding 2s threshold"
        
        # Step 4: Get database summary for broader context
        print("Step 4: Getting database summary...")
        step4_start = time.time()
        summary_result = get_statistics_summary()
        step4_duration = time.time() - step4_start
        
        workflow_data["database_context"] = {
            "duration": step4_duration,
            "total_publications": summary_result.get("total_publications", 0),
            "latest_year": summary_result.get("latest_year"),
            "publication_types_count": len(summary_result.get("publication_types", []))
        }
        
        # Validate Step 4
        assert "error" not in summary_result, f"Database summary failed: {summary_result}"
        assert step4_duration < 2.0, f"Database summary took {step4_duration}s, exceeding 2s threshold"
        
        # Overall workflow validation
        total_workflow_duration = time.time() - workflow_start
        workflow_data["total_duration"] = total_workflow_duration
        
        # Performance validation
        assert total_workflow_duration < 10.0, f"Complete workflow took {total_workflow_duration}s, exceeding 10s threshold"
        
        # Data consistency validation
        if detail_data.get("year"):
            # The publication year should be reasonable
            pub_year = int(detail_data["year"])
            assert 1950 <= pub_year <= 2025, f"Publication year {pub_year} seems unreasonable"
            
            # The year should appear in database statistics
            all_years = [item["value"] for item in year_stats_data.get("top_values", [])]
            # Note: May not be in top 10, but we can validate it's a reasonable year
            
        # Cross-validation between tools
        author_total = author_data.get("total_hits", 0)
        db_total = summary_result.get("total_publications", 0)
        assert author_total <= db_total, f"Author publications {author_total} exceeds database total {db_total}"
        
        print(f"✅ Author research workflow completed successfully in {total_workflow_duration:.2f}s")
        print(f"📊 Workflow summary: {json.dumps(workflow_data, indent=2)}")
    
    @pytest.mark.acceptance
    def test_author_comparison_workflow(self, es_client):
        """
        Compare multiple authors workflow.
        
        This test validates:
        1. Comparative analysis capabilities
        2. Data consistency across multiple searches
        3. Performance with multiple operations
        """
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        authors = ["Christian Fager", "Anna Dubois", "Smith"]  # Mix of specific and common names
        author_results = {}
        
        workflow_start = time.time()
        
        for author in authors:
            print(f"Analyzing author: {author}")
            
            # Search for author
            author_result = search_by_author(author, strategy="partial", max_results=20)
            author_data = json.loads(author_result)
            
            if "error" not in author_data:
                author_results[author] = {
                    "total_publications": author_data.get("total_hits", 0),
                    "recent_publications": len([
                        pub for pub in author_data.get("results", [])
                        if pub.get("year") and int(pub["year"]) >= 2020
                    ])
                }
                
                # Get details for first publication if available
                if author_data.get("results"):
                    first_pub_id = author_data["results"][0]["id"]
                    detail_result = get_publication_details(first_pub_id)
                    detail_data = json.loads(detail_result)
                    
                    if "error" not in detail_data:
                        author_results[author]["sample_publication"] = {
                            "title": detail_data.get("title", ""),
                            "year": detail_data.get("year", ""),
                            "journal": detail_data.get("journal", "")
                        }
        
        workflow_duration = time.time() - workflow_start
        
        # Validate comparative workflow
        assert len(author_results) > 0, "No successful author searches"
        assert workflow_duration < 15.0, f"Comparative workflow took {workflow_duration}s, too long"
        
        # Validate data consistency
        for author, data in author_results.items():
            assert data["total_publications"] >= 0
            assert data["recent_publications"] >= 0
            assert data["recent_publications"] <= data["total_publications"]
        
        print(f"✅ Author comparison workflow completed in {workflow_duration:.2f}s")
        print(f"📊 Author comparison results: {json.dumps(author_results, indent=2)}")


class TestTopicDiscoveryWorkflow:
    """Test topic discovery and exploration workflow."""
    
    @pytest.mark.acceptance
    def test_topic_to_expert_discovery(self, es_client):
        """
        Complete workflow: Topic search → Expert identification → Publication analysis
        
        This test validates:
        1. Topic-based discovery
        2. Expert identification from search results
        3. Deep dive into expert's work
        """
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        workflow_start = time.time()
        
        # Step 1: Search for topic
        print("Step 1: Searching for topic...")
        topic_result = search_publications("machine learning", max_results=10)
        topic_data = json.loads(topic_result)
        
        assert "error" not in topic_data, f"Topic search failed: {topic_data}"
        
        if not topic_data.get("results"):
            pytest.skip("No results found for topic search - cannot continue workflow")
        
        # Step 2: Extract expert from top results
        print("Step 2: Identifying experts...")
        expert_candidates = []
        
        for result in topic_data["results"][:5]:  # Top 5 results
            authors = result.get("authors", "")
            if authors:
                # Extract first author
                first_author = authors.split(',')[0].strip()
                if first_author and len(first_author) > 3:  # Reasonable author name
                    expert_candidates.append(first_author)
        
        assert len(expert_candidates) > 0, "No expert candidates found"
        
        # Step 3: Analyze expert's work
        print("Step 3: Analyzing expert's work...")
        expert_name = expert_candidates[0]
        expert_result = search_by_author(expert_name, strategy="exact", max_results=15)
        expert_data = json.loads(expert_result)
        
        if "error" not in expert_data and expert_data.get("results"):
            # Step 4: Get detailed analysis
            print("Step 4: Getting detailed publication analysis...")
            
            # Get publication details for recent work
            recent_publications = [
                pub for pub in expert_data["results"]
                if pub.get("year") and int(pub["year"]) >= 2020
            ]
            
            detailed_analyses = []
            for pub in recent_publications[:3]:  # Top 3 recent publications
                detail_result = get_publication_details(pub["id"])
                detail_data = json.loads(detail_result)
                
                if "error" not in detail_data:
                    detailed_analyses.append({
                        "title": detail_data.get("title", ""),
                        "year": detail_data.get("year", ""),
                        "abstract_length": len(detail_data.get("abstract", "")),
                        "has_keywords": bool(detail_data.get("keywords", "").strip())
                    })
            
            # Step 5: Contextualize with field statistics
            print("Step 5: Getting field context...")
            field_stats = get_field_statistics("publication_type", size=10)
            field_data = json.loads(field_stats)
            
            workflow_duration = time.time() - workflow_start
            
            # Validate workflow
            assert len(detailed_analyses) > 0, "No detailed analyses completed"
            assert workflow_duration < 12.0, f"Topic discovery workflow took {workflow_duration}s, too long"
            
            # Validate data consistency
            for analysis in detailed_analyses:
                if analysis["year"]:
                    year = int(analysis["year"])
                    assert 2020 <= year <= 2025, f"Recent publication year {year} invalid"
            
            workflow_summary = {
                "topic_results": topic_data.get("total_hits", 0),
                "expert_name": expert_name,
                "expert_total_publications": expert_data.get("total_hits", 0),
                "recent_publications_analyzed": len(detailed_analyses),
                "workflow_duration": workflow_duration
            }
            
            print(f"✅ Topic discovery workflow completed in {workflow_duration:.2f}s")
            print(f"📊 Discovery summary: {json.dumps(workflow_summary, indent=2)}")
        
        else:
            pytest.skip(f"Expert analysis failed for {expert_name}")


class TestDeepDiveAnalysisWorkflow:
    """Test comprehensive analysis workflow."""
    
    @pytest.mark.acceptance
    def test_comprehensive_database_analysis(self, es_client):
        """
        Complete database analysis workflow.
        
        This test validates:
        1. Database overview and trends
        2. Detailed field analysis
        3. Sample publication investigation
        4. Cross-validation of statistics
        """
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        workflow_start = time.time()
        analysis_results = {}
        
        # Step 1: Get database overview
        print("Step 1: Getting database overview...")
        summary_result = get_statistics_summary()
        
        assert "error" not in summary_result, f"Database summary failed: {summary_result}"
        
        analysis_results["database_overview"] = {
            "total_publications": summary_result.get("total_publications", 0),
            "latest_year": summary_result.get("latest_year"),
            "most_common_type": summary_result.get("most_common_type"),
            "total_authors": summary_result.get("total_authors", 0)
        }
        
        # Step 2: Analyze publication trends
        print("Step 2: Analyzing publication trends...")
        year_stats = get_field_statistics("year", size=10)
        year_data = json.loads(year_stats)
        
        assert "error" not in year_data, f"Year statistics failed: {year_data}"
        
        analysis_results["publication_trends"] = {
            "top_years": year_data.get("top_values", [])[:5],
            "total_documents": year_data.get("total_documents", 0)
        }
        
        # Step 3: Analyze publication types
        print("Step 3: Analyzing publication types...")
        type_stats = get_field_statistics("publication_type", size=10)
        type_data = json.loads(type_stats)
        
        assert "error" not in type_data, f"Publication type statistics failed: {type_data}"
        
        analysis_results["publication_types"] = {
            "top_types": type_data.get("top_values", [])[:5],
            "type_diversity": len(type_data.get("top_values", []))
        }
        
        # Step 4: Sample publication analysis
        print("Step 4: Analyzing sample publications...")
        if analysis_results["publication_types"]["top_types"]:
            most_common_type = analysis_results["publication_types"]["top_types"][0]["value"]
            
            # Search for examples of most common type
            sample_search = search_publications(most_common_type, max_results=5)
            sample_data = json.loads(sample_search)
            
            if "error" not in sample_data and sample_data.get("results"):
                # Get details for first sample
                sample_id = sample_data["results"][0]["id"]
                sample_detail = get_publication_details(sample_id)
                sample_detail_data = json.loads(sample_detail)
                
                if "error" not in sample_detail_data:
                    analysis_results["sample_analysis"] = {
                        "sample_type": most_common_type,
                        "sample_title": sample_detail_data.get("title", ""),
                        "sample_year": sample_detail_data.get("year", ""),
                        "sample_journal": sample_detail_data.get("journal", ""),
                        "has_abstract": bool(sample_detail_data.get("abstract", "").strip())
                    }
        
        # Step 5: Cross-validation
        print("Step 5: Cross-validating statistics...")
        workflow_duration = time.time() - workflow_start
        
        # Validate consistency
        db_total = analysis_results["database_overview"]["total_publications"]
        year_total = analysis_results["publication_trends"]["total_documents"]
        type_total = analysis_results["publication_types"].get("total_documents", 0)
        
        # All should report similar total document counts
        assert abs(db_total - year_total) / max(db_total, 1) < 0.1, \
            f"Document count mismatch: db={db_total}, year={year_total}"
        
        # Validate year trends
        if analysis_results["publication_trends"]["top_years"]:
            latest_in_trends = analysis_results["publication_trends"]["top_years"][0]["value"]
            latest_in_summary = analysis_results["database_overview"]["latest_year"]
            
            # Should be consistent (allowing for different aggregation methods)
            assert abs(int(latest_in_trends) - int(latest_in_summary)) <= 1, \
                f"Latest year mismatch: trends={latest_in_trends}, summary={latest_in_summary}"
        
        # Performance validation
        assert workflow_duration < 15.0, f"Analysis workflow took {workflow_duration}s, too long"
        
        analysis_results["workflow_performance"] = {
            "total_duration": workflow_duration,
            "steps_completed": 5,
            "cross_validation_passed": True
        }
        
        print(f"✅ Comprehensive analysis workflow completed in {workflow_duration:.2f}s")
        print(f"📊 Analysis results: {json.dumps(analysis_results, indent=2)}")


class TestErrorRecoveryWorkflows:
    """Test workflow resilience and error recovery."""
    
    @pytest.mark.resilience
    def test_workflow_with_partial_failures(self, es_client):
        """
        Test workflow behavior when some operations fail.
        
        This test validates:
        1. Graceful degradation when some tools fail
        2. Workflow continuation with available data
        3. Error reporting and recovery
        """
        initialize_elasticsearch_tools(es_client, "research-publications-static")
        
        workflow_start = time.time()
        successful_operations = 0
        failed_operations = 0
        results = {}
        
        # Step 1: Try author search with potentially invalid name
        try:
            author_result = search_by_author("NonexistentAuthor12345", strategy="exact", max_results=10)
            author_data = json.loads(author_result)
            
            if "error" not in author_data:
                successful_operations += 1
                results["author_search"] = "success"
            else:
                failed_operations += 1
                results["author_search"] = "failed"
        except Exception as e:
            failed_operations += 1
            results["author_search"] = f"exception: {str(e)}"
        
        # Step 2: Try invalid field statistics
        try:
            field_result = get_field_statistics("invalid_field_name", size=10)
            field_data = json.loads(field_result)
            
            if "error" not in field_data:
                successful_operations += 1
                results["field_stats"] = "success"
            else:
                failed_operations += 1
                results["field_stats"] = "failed"
        except Exception as e:
            failed_operations += 1
            results["field_stats"] = f"exception: {str(e)}"
        
        # Step 3: Try invalid publication details
        try:
            detail_result = get_publication_details("invalid_publication_id")
            detail_data = json.loads(detail_result)
            
            if "error" not in detail_data:
                successful_operations += 1
                results["publication_details"] = "success"
            else:
                failed_operations += 1
                results["publication_details"] = "failed"
        except Exception as e:
            failed_operations += 1
            results["publication_details"] = f"exception: {str(e)}"
        
        # Step 4: Try valid operation (should succeed)
        try:
            summary_result = get_statistics_summary()
            
            if "error" not in summary_result:
                successful_operations += 1
                results["database_summary"] = "success"
            else:
                failed_operations += 1
                results["database_summary"] = "failed"
        except Exception as e:
            failed_operations += 1
            results["database_summary"] = f"exception: {str(e)}"
        
        workflow_duration = time.time() - workflow_start
        
        # Validate error recovery
        total_operations = successful_operations + failed_operations
        assert total_operations == 4, f"Expected 4 operations, got {total_operations}"
        
        # Should have at least some successful operations
        assert successful_operations > 0, "No operations succeeded"
        
        # Should handle errors gracefully (no exceptions)
        exception_count = sum(1 for result in results.values() if "exception" in str(result))
        assert exception_count == 0, f"Had {exception_count} exceptions, should handle gracefully"
        
        # Performance should still be reasonable
        assert workflow_duration < 10.0, f"Error recovery workflow took {workflow_duration}s, too long"
        
        print(f"✅ Error recovery workflow completed in {workflow_duration:.2f}s")
        print(f"📊 Recovery results: {json.dumps(results, indent=2)}")
        print(f"🎯 Success rate: {successful_operations}/{total_operations} operations successful")