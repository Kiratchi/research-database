import sys
import os
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.research_agent.tools.elasticsearch_tools import initialize_elasticsearch_tools, search_by_author
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

def test_offset_functionality():
    """Test if offset parameter works in search_by_author"""
    print("🧪 Testing offset functionality:")
    
    # Test 1: First 10 results (offset=0)
    print("  📥 Testing first 10 results (offset=0)...")
    result1 = search_by_author("Per-Olof Arnäs", max_results=10, offset=0)
    data1 = json.loads(result1)
    print(f"  First 10: Got {len(data1.get('results', []))} results")
    print(f"  Total hits: {data1.get('total_hits', 0)}")
    if data1.get('results'):
        print(f"  First result: {data1['results'][0]['title'][:60]}...")
        print(f"  Last result: {data1['results'][-1]['title'][:60]}...")
    
    # Test 2: Next 10 results (offset=10)  
    print("\n  📥 Testing next 10 results (offset=10)...")
    result2 = search_by_author("Per-Olof Arnäs", max_results=10, offset=10)
    data2 = json.loads(result2)
    print(f"  Next 10: Got {len(data2.get('results', []))} results")
    print(f"  Total hits: {data2.get('total_hits', 0)}")
    if data2.get('results'):
        print(f"  First result: {data2['results'][0]['title'][:60]}...")
        print(f"  Last result: {data2['results'][-1]['title'][:60]}...")
    
    # Test 3: Check pagination info
    print("\n  📊 Checking pagination info...")
    if data1.get('pagination'):
        print(f"  Batch 1 pagination: {data1['pagination']}")
    if data2.get('pagination'):
        print(f"  Batch 2 pagination: {data2['pagination']}")
    
    # Test 4: Check if results are different
    print("\n  🔍 Comparing results...")
    if data1.get('results') and data2.get('results'):
        first_titles = [r['title'] for r in data1['results']]
        second_titles = [r['title'] for r in data2['results']]
        overlap = set(first_titles) & set(second_titles)
        print(f"  Overlap between batches: {len(overlap)} publications")
        
        if len(overlap) == 0:
            print("  ✅ OFFSET WORKS: Different results returned")
            print("  🎯 Offset parameter is functioning correctly!")
        else:
            print("  ❌ OFFSET BROKEN: Same results returned")
            print("  🚫 Offset parameter is NOT working")
            print(f"  Duplicate titles: {list(overlap)[:3]}...")
    
    # Test 5: Test max_results=20 in single call
    print("\n  📥 Testing single call with max_results=20...")
    result3 = search_by_author("Per-Olof Arnäs", max_results=20, offset=0)
    data3 = json.loads(result3)
    print(f"  Single call: Got {len(data3.get('results', []))} results")
    if len(data3.get('results', [])) == 20:
        print("  ✅ Single call with max_results=20 works!")
    else:
        print("  ⚠️ Single call with max_results=20 limited to 10 results")
    
    return data1, data2, data3

if __name__ == "__main__":
    print("🔧 Setting up Elasticsearch connection...")
    
    # Load environment variables
    load_dotenv()
    
    # Connect to Elasticsearch (adjust these to match your app.py config)
    try:
        es_client = Elasticsearch(
            hosts=[os.getenv("ES_HOST")],
            http_auth=(os.getenv("ES_USER"), os.getenv("ES_PASS")),
            verify_certs=False,  # Add if you use this in your app
            timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )
        
        # Test connection
        if es_client.ping():
            print("✅ Elasticsearch connected successfully")
        else:
            print("❌ Elasticsearch connection failed")
            exit(1)
            
    except Exception as e:
        print(f"❌ Elasticsearch connection error: {e}")
        exit(1)
    
    # Initialize tools
    print("🔧 Initializing Elasticsearch tools...")
    initialize_elasticsearch_tools(es_client, "research-publications-static")
    
    # Run the test
    print("\n" + "="*60)
    print("🧪 RUNNING OFFSET FUNCTIONALITY TEST")
    print("="*60)
    
    try:
        test_offset_functionality()
        print("\n" + "="*60)
        print("✅ Test completed successfully!")
        print("="*60)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()