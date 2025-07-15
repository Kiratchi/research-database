#!/usr/bin/env python3
"""
Quick test to verify the debug setup works before using the notebook.
"""

import os
import sys
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# Import our chat agent components
from chat_parser import ChatParser
from query_builder import QueryBuilder
from response_formatter import ResponseFormatter
from agent_tools import initialize_tools
import agent_tools

def test_setup():
    """Test that everything is set up correctly."""
    print("🔧 Testing debug setup...")
    
    # Test imports
    print("✅ All imports successful!")
    
    # Test ES connection
    load_dotenv(dotenv_path=".env", override=True)
    es = Elasticsearch(
        hosts=[os.getenv('ES_HOST')],
        http_auth=(os.getenv('ES_USER'), os.getenv('ES_PASS')),
        verify_certs=False
    )
    
    if es.ping():
        print("✅ ES connection successful!")
    else:
        print("❌ ES connection failed!")
        return False
    
    # Initialize agent tools
    initialize_tools(es)
    print("✅ Agent tools initialized!")
    
    # Test components
    parser = ChatParser()
    builder = QueryBuilder(agent_tools)
    formatter = ResponseFormatter()
    print("✅ Chat components initialized!")
    
    # Test a simple query
    test_query = "How many papers has Fager published?"
    print(f"\n🔍 Testing query: '{test_query}'")
    
    try:
        # Parse
        parsed = parser.parse(test_query)
        print(f"✅ Parsed - Intent: {parsed.intent.value}, Author: {parsed.author_name}")
        
        # Build
        query_spec = builder.build_query(parsed)
        print(f"✅ Built - Function: {query_spec['function']}")
        
        # Execute (this might take a moment)
        print("⏳ Executing query...")
        result = builder.execute_query(query_spec)
        print(f"✅ Executed - Type: {result['type']}, Count: {result.get('count', 'N/A')}")
        
        # Format
        formatted = formatter.format_response(result)
        print(f"✅ Formatted - Type: {formatted['type']}")
        print(f"📄 Response: {formatted['content']}")
        
        print("\n🎉 All tests passed! The debug notebook should work perfectly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)