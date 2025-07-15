#!/usr/bin/env python3
"""
Comprehensive Test Suite Validation Script

This script validates that our production-ready test suite works correctly
by running key tests and reporting results.
"""

import subprocess
import sys
import time
import json
from pathlib import Path

def run_test_command(cmd, description):
    """Run a test command and capture results."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        duration = time.time() - start_time
        
        print(f"Duration: {duration:.2f}s")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return {
            'success': result.returncode == 0,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Test timed out after 120 seconds")
        return {
            'success': False,
            'duration': 120,
            'stdout': '',
            'stderr': 'Test timed out'
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            'success': False,
            'duration': 0,
            'stdout': '',
            'stderr': str(e)
        }

def main():
    """Run the comprehensive test suite validation."""
    print("ELASTICSEARCH TOOLS COMPREHENSIVE TEST SUITE VALIDATION")
    print("="*60)
    
    # Change to the correct directory
    original_dir = Path.cwd()
    workspace_dir = Path("/Users/filipberntsson/Dev/es_workspace")
    
    if not workspace_dir.exists():
        print(f"ERROR: Workspace directory {workspace_dir} does not exist")
        sys.exit(1)
    
    # Change to workspace directory
    import os
    os.chdir(workspace_dir)
    
    # Test commands to run
    test_commands = [
        {
            'cmd': 'source venv/bin/activate && python -m pytest tests/tools/test_elasticsearch_tools_comprehensive.py::TestSearchPublications::test_search_publications_unicode_inputs -v',
            'description': 'Unicode Input Handling Test'
        },
        {
            'cmd': 'source venv/bin/activate && python -m pytest tests/tools/test_elasticsearch_tools_comprehensive.py::TestSearchPublications::test_search_publications_edge_cases -v',
            'description': 'Edge Case Handling Test'
        },
        {
            'cmd': 'source venv/bin/activate && python -m pytest tests/tools/test_elasticsearch_tools_comprehensive.py::TestSearchByAuthor::test_search_by_author_all_strategies -v',
            'description': 'Author Search Strategy Test'
        },
        {
            'cmd': 'source venv/bin/activate && python -m pytest tests/tools/test_elasticsearch_tools_comprehensive.py::TestGetFieldStatistics::test_get_field_statistics_all_fields -v',
            'description': 'Field Statistics Test'
        },
        {
            'cmd': 'source venv/bin/activate && python -m pytest tests/tools/test_elasticsearch_tools_comprehensive.py::TestGetPublicationDetails::test_get_publication_details_valid_id -v',
            'description': 'Publication Details Test'
        }
    ]
    
    results = []
    
    for test_info in test_commands:
        result = run_test_command(test_info['cmd'], test_info['description'])
        results.append({
            'description': test_info['description'],
            'success': result['success'],
            'duration': result['duration']
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUITE VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - passed_tests
    total_duration = sum(r['duration'] for r in results)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {passed_tests/total_tests:.1%}")
    print(f"Total Duration: {total_duration:.2f}s")
    print(f"Average Duration: {total_duration/total_tests:.2f}s per test")
    
    print(f"\nDETAILED RESULTS:")
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{status} {result['description']} ({result['duration']:.2f}s)")
    
    if failed_tests > 0:
        print(f"\n⚠️  {failed_tests} tests failed. Check the detailed output above.")
        return 1
    else:
        print(f"\n🎉 All {passed_tests} tests passed!")
        
        # Additional validation
        if total_duration > 30:
            print(f"⚠️  Total test duration {total_duration:.2f}s is longer than expected (>30s)")
        
        if any(r['duration'] > 10 for r in results):
            slow_tests = [r for r in results if r['duration'] > 10]
            print(f"⚠️  {len(slow_tests)} tests took longer than 10s:")
            for test in slow_tests:
                print(f"   - {test['description']}: {test['duration']:.2f}s")
        
        return 0

if __name__ == "__main__":
    sys.exit(main())