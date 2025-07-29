"""
Integration Test Script - Verify new tools work with your directory structure
Run this from your research_agent directory to test the integration
"""

import os
import sys
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_tools_import():
    """Test that we can import the new tools."""
    try:
        print("🔍 Testing tools import...")
        from tools import get_all_tools
        print("✅ Successfully imported get_all_tools")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_tools_loading():
    """Test that we can load tools without ES client."""
    try:
        print("🔍 Testing tools loading...")
        from tools import get_all_tools
        
        # Test loading without ES client (should use default settings)
        tools = get_all_tools()
        print(f"✅ Loaded {len(tools)} tools:")
        for tool in tools:
            print(f"   - {tool.name}: {tool.description[:60]}...")
        return True, tools
    except Exception as e:
        print(f"❌ Tools loading failed: {e}")
        return False, []

def test_es_connection():
    """Test Elasticsearch connection."""
    try:
        print("🔍 Testing Elasticsearch connection...")
        from elasticsearch import Elasticsearch
        
        es_host = os.getenv("ES_HOST")
        es_user = os.getenv("ES_USER") 
        es_pass = os.getenv("ES_PASS")
        
        if not all([es_host, es_user, es_pass]):
            print("⚠️ Missing ES credentials in environment")
            return False, None
        
        es_client = Elasticsearch(
            [es_host],
            http_auth=(es_user, es_pass),
            timeout=10
        )
        
        if es_client.ping():
            print("✅ Elasticsearch connection successful")
            return True, es_client
        else:
            print("❌ Elasticsearch ping failed")
            return False, None
            
    except Exception as e:
        print(f"❌ Elasticsearch connection error: {e}")
        return False, None

def test_tools_with_es():
    """Test tools with actual ES client."""
    try:
        print("🔍 Testing tools with ES client...")
        from tools import get_all_tools
        
        # Get ES client
        es_success, es_client = test_es_connection()
        if not es_success:
            print("⚠️ Skipping ES-based tools test")
            return False
        
        # Test with ES client
        tools = get_all_tools(es_client=es_client, index_name="research-publications-static")
        print(f"✅ Loaded {len(tools)} tools with ES client")
        
        return True
        
    except Exception as e:
        print(f"❌ Tools with ES failed: {e}")
        return False

def test_tool_parameters():
    """Test tool parameter schemas."""
    try:
        print("🔍 Testing tool parameters...")
        from tools import get_all_tools
        
        tools = get_all_tools()
        
        for tool in tools:
            print(f"\n📋 Tool: {tool.name}")
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema
                print(f"   Has parameter schema: {schema.__name__}")
                
                # Try to get field information
                if hasattr(schema, 'model_fields'):
                    # Pydantic v2
                    fields = list(schema.model_fields.keys())
                    print(f"   Parameters: {fields}")
                elif hasattr(schema, '__fields__'):
                    # Pydantic v1
                    fields = list(schema.__fields__.keys())
                    print(f"   Parameters: {fields}")
            else:
                print("   No parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter testing failed: {e}")
        return False

def test_simple_search():
    """Test a simple search if ES is available."""
    try:
        print("🔍 Testing simple search...")
        from tools import get_all_tools
        
        # Get ES client
        es_success, es_client = test_es_connection()
        if not es_success:
            print("⚠️ Skipping search test - no ES connection")
            return False
        
        # Get tools with ES
        tools = get_all_tools(es_client=es_client, index_name="research-publications-static")
        
        # Find the unified_search tool
        search_tool = None
        for tool in tools:
            if tool.name == "unified_search":
                search_tool = tool
                break
        
        if not search_tool:
            print("❌ unified_search tool not found")
            return False
        
        # Test a simple search
        print("   Running test search for 'machine learning'...")
        result = search_tool._run(
            query="machine learning",
            max_results=5
        )
        
        print(f"✅ Search completed, result length: {len(result)} characters")
        
        # Try to parse result as JSON
        import json
        parsed_result = json.loads(result)
        print(f"   Found {len(parsed_result.get('results', []))} results")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple search failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 Starting New Tools Integration Test")
    print("=" * 50)
    
    # Check current directory
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 paths
    
    # Run tests
    tests = [
        ("Import Test", test_tools_import),
        ("Tools Loading", test_tools_loading),
        ("Tool Parameters", test_tool_parameters),
        ("ES Connection", test_es_connection),
        ("Tools with ES", test_tools_with_es),
        ("Simple Search", test_simple_search),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-' * 30}")
        try:
            if test_func == test_tools_loading:
                success, tools = test_func()
            elif test_func == test_es_connection:
                success, es_client = test_func()
            else:
                success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'=' * 50}")
    print("🏁 INTEGRATION TEST SUMMARY")
    print(f"{'=' * 50}")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Integration is working correctly.")
    elif passed >= len(results) - 2:
        print("✅ Integration is mostly working. Minor issues detected.")
    else:
        print("⚠️ Integration has issues. Check the failed tests above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)