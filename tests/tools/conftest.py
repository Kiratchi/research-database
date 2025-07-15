"""
Pytest configuration and fixtures for comprehensive tool testing.

This module provides the infrastructure for production-ready testing of
Elasticsearch tools with performance monitoring and resilience validation.
"""

import pytest
import json
import time
import psutil
import threading
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
from unittest.mock import Mock, patch
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import os
import sys
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from research_agent.tools.elasticsearch_tools import ElasticsearchTools
except ImportError:
    # Fallback for different import structures
    from src.research_agent.tools.elasticsearch_tools import (
        initialize_elasticsearch_tools,
        create_elasticsearch_tools,
        get_statistics_summary
    )


class PerformanceMetrics:
    """Collect and analyze performance metrics for tool testing."""
    
    def __init__(self):
        self.metrics = []
        self.start_time = None
        self.process = psutil.Process()
        self.baseline_memory = None
    
    def start_measurement(self, tool_name: str, operation: str):
        """Start measuring performance for a tool operation."""
        self.start_time = time.time()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        return {
            'tool': tool_name,
            'operation': operation,
            'start_time': self.start_time,
            'start_memory': self.baseline_memory
        }
    
    def end_measurement(self, measurement: Dict[str, Any]) -> Dict[str, Any]:
        """End measurement and calculate metrics."""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        result = {
            **measurement,
            'end_time': end_time,
            'duration': end_time - measurement['start_time'],
            'memory_delta': end_memory - measurement['start_memory'],
            'peak_memory': end_memory
        }
        
        self.metrics.append(result)
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics:
            return {}
        
        durations = [m['duration'] for m in self.metrics]
        memory_deltas = [m['memory_delta'] for m in self.metrics]
        
        return {
            'total_operations': len(self.metrics),
            'average_duration': sum(durations) / len(durations),
            'max_duration': max(durations),
            'min_duration': min(durations),
            'average_memory_delta': sum(memory_deltas) / len(memory_deltas),
            'max_memory_delta': max(memory_deltas),
            'operations_over_threshold': len([d for d in durations if d > 3.0])
        }


class TestDataGenerator:
    """Generate test data for various testing scenarios."""
    
    @staticmethod
    def generate_unicode_inputs() -> List[str]:
        """Generate Unicode and international test inputs."""
        return [
            "François Müller",
            "José María García",
            "北京大学",
            "Владимир Путин",
            "محمد عبدالله",
            "Ελληνικά",
            "🔬 Research 📊",
            "naïve Bayes",
            "Schrödinger",
            "Café résumé"
        ]
    
    @staticmethod
    def generate_edge_case_inputs() -> List[str]:
        """Generate edge case inputs for resilience testing."""
        return [
            "",  # Empty string
            "   ",  # Whitespace only
            "a",  # Single character
            "THE AND OR NOT",  # All stop words
            "cHrisTiaN FaGer",  # Mixed case
            "author@domain.com",  # Email format
            "C++ programming",  # Special characters
            "machine-learning",  # Hyphenated
            "AI/ML",  # Slash separator
            "search query with very long text " * 100,  # Long query
            "'; DROP TABLE users; --",  # SQL injection attempt
            "OR 1=1",  # Boolean injection
            "<script>alert('xss')</script>",  # XSS attempt
            "null",  # Null string
            "undefined",  # Undefined string
            "NaN",  # NaN string
        ]
    
    @staticmethod
    def generate_large_query(size_kb: int = 10) -> str:
        """Generate large query string of specified size."""
        base_text = "machine learning artificial intelligence neural networks deep learning "
        target_size = size_kb * 1024
        repetitions = (target_size // len(base_text)) + 1
        return (base_text * repetitions)[:target_size]
    
    @staticmethod
    def generate_boundary_values() -> Dict[str, List[Any]]:
        """Generate boundary values for different parameter types."""
        return {
            'max_results': [0, 1, 10, 100, 1000, 10000],
            'size': [0, 1, 5, 10, 50, 100, 1000],
            'field_names': ['year', 'authors', 'journal', 'publication_type', 'invalid_field'],
            'publication_ids': ['valid_id', 'invalid_id', '', '123', 'very_long_id_' * 20]
        }


@pytest.fixture(scope="session")
def es_client():
    """Create Elasticsearch client for testing."""
    load_dotenv(dotenv_path=".env", override=True)
    
    try:
        client = Elasticsearch(
            hosts=[os.getenv('ES_HOST', 'localhost:9200')],
            http_auth=(os.getenv('ES_USER'), os.getenv('ES_PASS')),
            verify_certs=False,
            timeout=30
        )
        
        # Test connection
        if client.ping():
            yield client
        else:
            pytest.skip("Elasticsearch not available")
            
    except Exception as e:
        pytest.skip(f"Could not connect to Elasticsearch: {e}")


@pytest.fixture(scope="session")
def es_tools(es_client):
    """Initialize Elasticsearch tools for testing."""
    index_name = "research-publications-static"
    initialize_elasticsearch_tools(es_client, index_name)
    return create_elasticsearch_tools()


@pytest.fixture
def performance_monitor():
    """Provide performance monitoring for tests."""
    return PerformanceMetrics()


@pytest.fixture
def test_data_generator():
    """Provide test data generator for tests."""
    return TestDataGenerator()


@pytest.fixture
def mock_es_client():
    """Provide mock Elasticsearch client for unit tests."""
    mock_client = Mock()
    mock_client.ping.return_value = True
    return mock_client


@pytest.fixture
def sample_search_response():
    """Provide sample Elasticsearch search response."""
    return {
        'hits': {
            'total': 42,
            'hits': [
                {
                    '_id': 'doc1',
                    '_score': 1.5,
                    '_source': {
                        'title': 'Machine Learning Applications',
                        'authors': 'John Smith, Jane Doe',
                        'year': 2023,
                        'abstract': 'This paper presents novel applications of machine learning...',
                        'journal': 'AI Research Journal',
                        'publication_type': 'journal-article'
                    }
                },
                {
                    '_id': 'doc2',
                    '_score': 1.2,
                    '_source': {
                        'title': 'Deep Learning Fundamentals',
                        'authors': 'Alice Johnson',
                        'year': 2022,
                        'abstract': 'A comprehensive overview of deep learning techniques...',
                        'journal': 'Neural Networks Today',
                        'publication_type': 'conference-paper'
                    }
                }
            ]
        }
    }


@pytest.fixture
def sample_aggregation_response():
    """Provide sample Elasticsearch aggregation response."""
    return {
        'hits': {'total': 1000},
        'aggregations': {
            'field_stats': {
                'buckets': [
                    {'key': '2023', 'doc_count': 150},
                    {'key': '2022', 'doc_count': 120},
                    {'key': '2021', 'doc_count': 100},
                    {'key': '2020', 'doc_count': 80},
                    {'key': '2019', 'doc_count': 60}
                ]
            }
        }
    }


@pytest.fixture(scope="session")
def database_schema_info(es_client):
    """Get database schema information for validation."""
    try:
        # Get index mapping
        mapping = es_client.indices.get_mapping(index="research-publications-static")
        
        # Get sample document
        sample_doc = es_client.search(
            index="research-publications-static",
            body={"size": 1}
        )
        
        return {
            'mapping': mapping,
            'sample_document': sample_doc['hits']['hits'][0]['_source'] if sample_doc['hits']['hits'] else {},
            'total_documents': es_client.count(index="research-publications-static")['count']
        }
    except Exception as e:
        pytest.skip(f"Could not retrieve schema info: {e}")


@pytest.fixture
def concurrency_test_helper():
    """Helper for concurrent testing scenarios."""
    class ConcurrencyHelper:
        def __init__(self):
            self.results = []
            self.errors = []
            self.lock = threading.Lock()
        
        def run_concurrent_operations(self, operation_func, num_threads=10, *args, **kwargs):
            """Run operation concurrently with multiple threads."""
            threads = []
            
            def worker():
                try:
                    result = operation_func(*args, **kwargs)
                    with self.lock:
                        self.results.append(result)
                except Exception as e:
                    with self.lock:
                        self.errors.append(str(e))
            
            # Start threads
            for _ in range(num_threads):
                thread = threading.Thread(target=worker)
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            return {
                'results': self.results,
                'errors': self.errors,
                'success_rate': len(self.results) / (len(self.results) + len(self.errors))
            }
    
    return ConcurrencyHelper()


# Performance thresholds and constants
PERFORMANCE_THRESHOLDS = {
    'individual_tool_max_duration': 3.0,  # seconds
    'workflow_max_duration': 10.0,  # seconds
    'aggregation_max_duration': 2.0,  # seconds
    'detail_retrieval_max_duration': 1.0,  # seconds
    'memory_leak_threshold': 50.0,  # MB
    'max_acceptable_error_rate': 0.05  # 5%
}


@pytest.fixture
def performance_thresholds():
    """Provide performance thresholds for testing."""
    return PERFORMANCE_THRESHOLDS


# Markers for test categorization
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.resilience = pytest.mark.resilience
pytest.mark.acceptance = pytest.mark.acceptance