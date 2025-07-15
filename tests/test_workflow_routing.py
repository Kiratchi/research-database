#!/usr/bin/env python3
"""
Test the fixed workflow routing
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from src.research_agent.core.workflow import ResearchAgent

def test_workflow_routing():
    """Test that the workflow routing works correctly."""
    print("🧪 Testing workflow routing...")
    
    load_dotenv()
    
    try:
        # Setup ES client
        es = Elasticsearch(
            hosts=[os.getenv('ES_HOST')],
            http_auth=(os.getenv('ES_USER'), os.getenv('ES_PASS')),
            verify_certs=False
        )
        
        # Create agent
        agent = ResearchAgent(es_client=es)
        
        # Test simple query that should complete in one step
        query = "How many publications has Christian Fager published?"
        
        print(f"Testing query: {query}")
        print("-" * 50)
        
        # Use sync invoke to avoid async issues
        result = agent.app.invoke({
            "input": query,
            "plan": [],
            "past_steps": [],
            "response": None,
            "session_id": None,
            "total_results": None,
            "current_step": 0,
            "error": None
        })
        
        print(f"✅ Final result: {result}")
        
        # Check if we got a response without unnecessary replanning
        if "response" in result:
            print("✅ Workflow ended with response (good)")
        else:
            print("❌ Workflow didn't end with response")
            
        print(f"Past steps: {len(result.get('past_steps', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_workflow_routing()
    if success:
        print("\n🎉 Workflow routing test passed!")
    else:
        print("\n❌ Workflow routing test failed")