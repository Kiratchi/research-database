"""
Test script for the Streamlit app functionality
"""

import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from chat_parser import ChatParser
from query_builder import QueryBuilder
from response_formatter import ResponseFormatter
from agent_tools import initialize_tools
import agent_tools


def test_streamlit_app_components():
    """Test that all Streamlit app components work correctly."""
    print("Testing Streamlit app components...")
    
    try:
        # Load environment variables
        load_dotenv(dotenv_path=".env", override=True)
        
        # Initialize Elasticsearch client
        es = Elasticsearch(
            hosts=[os.getenv('ES_HOST')],
            http_auth=(os.getenv('ES_USER'), os.getenv('ES_PASS')),
            verify_certs=False
        )
        
        # Test connection
        if not es.ping():
            print("❌ Could not connect to Elasticsearch")
            return False
        
        print("✅ Elasticsearch connection successful")
        
        # Initialize agent tools
        initialize_tools(es)
        print("✅ Agent tools initialized")
        
        # Create chat components
        parser = ChatParser()
        builder = QueryBuilder(agent_tools)
        formatter = ResponseFormatter()
        print("✅ Chat components created")
        
        # Test query processing
        test_query = "How many papers has Christian Fager published?"
        
        # Step 1: Parse
        parsed_query = parser.parse(test_query)
        print(f"✅ Query parsed: {parsed_query.intent.value}")
        
        # Step 2: Build
        query_spec = builder.build_query(parsed_query)
        print(f"✅ Query built: {query_spec['function']}")
        
        # Step 3: Execute
        query_result = builder.execute_query(query_spec)
        print(f"✅ Query executed: {query_result['type']}")
        
        # Step 4: Format
        formatted_response = formatter.format_response(query_result)
        print(f"✅ Response formatted: {formatted_response['type']}")
        
        # Test database stats
        stats = agent_tools.get_statistics_summary()
        print(f"✅ Database stats: {stats['total_publications']:,} publications")
        
        print("\n🎉 All Streamlit app components working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing components: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_streamlit_app_components()
    exit(0 if success else 1)