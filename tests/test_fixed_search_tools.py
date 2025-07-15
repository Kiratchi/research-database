#!/usr/bin/env python3
"""
Test the fixed search tools with Anna Dubois
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

def test_fixed_search_tools():
    """Test search tools after fixing Persons field."""
    print("🔧 Testing fixed search tools with Anna Dubois...")
    
    load_dotenv()
    
    try:
        es = Elasticsearch(
            hosts=[os.getenv('ES_HOST')],
            http_auth=(os.getenv('ES_USER'), os.getenv('ES_PASS')),
            verify_certs=False
        )
        
        # Initialize tools
        from src.research_agent.tools.elasticsearch_tools import (
            initialize_elasticsearch_tools,
            search_by_author,
            search_publications
        )
        
        initialize_elasticsearch_tools(es, "research-publications-static")
        
        # Test 1: Search by author
        print("\n👤 Testing search_by_author('Anna Dubois')...")
        anna_results = search_by_author("Anna Dubois")
        anna_data = json.loads(anna_results)
        
        print(f"✅ Total hits: {anna_data.get('total_hits', 0)}")
        print(f"✅ Results: {len(anna_data.get('results', []))}")
        
        if anna_data.get('results'):
            print("📋 Sample results:")
            for i, result in enumerate(anna_data['results'][:3], 1):
                print(f"   {i}. {result['title']}")
                print(f"      Authors: {result['authors']}")
                print(f"      Year: {result['year']}")
                print()
        
        # Test 2: General search
        print("\n🔍 Testing search_publications('Anna Dubois')...")
        general_results = search_publications("Anna Dubois")
        general_data = json.loads(general_results)
        
        print(f"✅ Total hits: {general_data.get('total_hits', 0)}")
        print(f"✅ Results: {len(general_data.get('results', []))}")
        
        return True, anna_data, general_data
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, {}, {}

if __name__ == "__main__":
    success, anna_data, general_data = test_fixed_search_tools()
    
    if success:
        anna_count = anna_data.get('total_hits', 0)
        general_count = general_data.get('total_hits', 0)
        
        print(f"\n🎉 Search tools fixed!")
        print(f"   Anna Dubois author search: {anna_count} results")
        print(f"   Anna Dubois general search: {general_count} results")
        print(f"   Ready to test in Streamlit!")
    else:
        print("\n❌ Search tools still have issues")