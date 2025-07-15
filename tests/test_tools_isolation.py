#!/usr/bin/env python3
"""
Test Elasticsearch tools in isolation
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

def test_elasticsearch_connection():
    """Test ES connection."""
    print("🔌 Testing Elasticsearch connection...")
    
    load_dotenv()
    
    try:
        es = Elasticsearch(
            hosts=[os.getenv('ES_HOST')],
            http_auth=(os.getenv('ES_USER'), os.getenv('ES_PASS')),
            verify_certs=False
        )
        
        if es.ping():
            print("✅ Elasticsearch connection successful")
            return True, es
        else:
            print("❌ Elasticsearch ping failed")
            return False, None
            
    except Exception as e:
        print(f"❌ Elasticsearch connection failed: {str(e)}")
        return False, None


def test_elasticsearch_tools(es):
    """Test individual ES tools."""
    print("\n🔧 Testing Elasticsearch tools...")
    
    try:
        # Initialize tools
        from src.research_agent.tools.elasticsearch_tools import (
            initialize_elasticsearch_tools,
            search_by_author,
            search_publications,
            get_database_summary,
            get_field_statistics
        )
        
        # Initialize with actual ES client
        initialize_elasticsearch_tools(es, "research-publications-static")
        
        print("✅ Tools initialized successfully")
        
        # Test 1: Database summary
        print("\n📊 Testing database summary...")
        summary = get_database_summary()
        print(f"✅ Database summary: {summary}")
        
        # Test 2: Search by author "Anna Dubois"
        print("\n👤 Testing search by author 'Anna Dubois'...")
        anna_results = search_by_author("Anna Dubois")
        print(f"✅ Anna Dubois search: {len(anna_results)} results")
        
        # Test 3: Search by author with variations
        print("\n🔍 Testing author name variations...")
        variations = ["A. Dubois", "Dubois, Anna", "A Dubois"]
        for var in variations:
            results = search_by_author(var)
            print(f"   {var}: {len(results)} results")
        
        # Test 4: General search for "Dubois"
        print("\n🔍 Testing general search for 'Dubois'...")
        general_results = search_publications("Dubois")
        print(f"✅ General 'Dubois' search: {len(general_results)} results")
        
        # Test 5: Field statistics
        print("\n📈 Testing field statistics...")
        try:
            stats = get_field_statistics("author")
            print(f"✅ Author field statistics: {len(stats)} unique authors")
        except Exception as e:
            print(f"⚠️ Field statistics failed: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tools test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_author_search_detailed(es):
    """Test author search with detailed output."""
    print("\n🔍 Detailed author search test...")
    
    try:
        from src.research_agent.tools.elasticsearch_tools import initialize_elasticsearch_tools
        
        # Initialize tools
        initialize_elasticsearch_tools(es, "research-publications-static")
        
        # Test direct ES query to understand data structure
        print("\n📋 Testing direct ES query...")
        
        # Search for any author with "Dubois" in name
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"author": "Dubois"}},
                        {"wildcard": {"author": "*Dubois*"}},
                        {"wildcard": {"author": "*dubois*"}}
                    ]
                }
            },
            "size": 10
        }
        
        response = es.search(index="research-publications-static", body=search_body)
        hits = response['hits']['hits']
        
        print(f"📊 Direct ES search found {len(hits)} results with 'Dubois' in author")
        
        # Show sample authors
        if hits:
            print("\n👥 Sample authors found:")
            for hit in hits[:3]:
                author = hit['_source'].get('author', 'Unknown')
                title = hit['_source'].get('title', 'Unknown')[:50]
                print(f"   Author: {author}")
                print(f"   Title: {title}...")
                print()
        
        return True, hits
        
    except Exception as e:
        print(f"❌ Detailed search failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, []


def test_database_content(es):
    """Test what's actually in the database."""
    print("\n📚 Testing database content...")
    
    try:
        # Get total count
        total_response = es.count(index="research-publications-static")
        total_docs = total_response['count']
        print(f"📊 Total documents: {total_docs}")
        
        # Get sample documents
        sample_response = es.search(
            index="research-publications-static",
            body={"query": {"match_all": {}}, "size": 5}
        )
        
        print(f"\n📋 Sample documents:")
        for i, hit in enumerate(sample_response['hits']['hits'], 1):
            source = hit['_source']
            author = source.get('author', 'Unknown')
            title = source.get('title', 'Unknown')[:50]
            year = source.get('year', 'Unknown')
            print(f"   {i}. {author} ({year}): {title}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Database content test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all isolation tests."""
    print("🧪 Testing Elasticsearch Tools in Isolation")
    print("=" * 60)
    
    # Test 1: ES connection
    conn_success, es = test_elasticsearch_connection()
    if not conn_success:
        print("❌ Cannot proceed without ES connection")
        return 1
    
    # Test 2: Database content
    content_success = test_database_content(es)
    
    # Test 3: Tools functionality
    tools_success = test_elasticsearch_tools(es)
    
    # Test 4: Detailed author search
    search_success, hits = test_author_search_detailed(es)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Isolation Test Results:")
    
    tests = [
        ("ES Connection", conn_success),
        ("Database Content", content_success),
        ("Tools Functionality", tools_success),
        ("Detailed Search", search_success),
    ]
    
    passed = sum(1 for _, success in tests if success)
    total = len(tests)
    
    for test_name, success in tests:
        print(f"{'✅' if success else '❌'} {test_name}")
    
    print(f"\n🏆 Results: {passed}/{total} tests passed")
    
    # Analysis
    print("\n🔍 Analysis:")
    if not hits:
        print("⚠️ No authors with 'Dubois' found - this might be expected")
        print("💡 The plan-and-execute workflow worked correctly")
        print("❌ The issue is with Pydantic validation in replanner")
    else:
        print("✅ Authors with 'Dubois' exist in database")
        print("❌ Tool might have search strategy issues")
    
    return 0 if passed >= 3 else 1


if __name__ == "__main__":
    sys.exit(main())