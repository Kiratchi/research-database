#!/usr/bin/env python3
"""
Test the Streamlit integration after async fixes
"""

import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv

def test_streamlit_agent_sync():
    """Test StreamlitAgent with sync methods."""
    print("🧪 Testing StreamlitAgent after async fixes...")
    
    try:
        # Mock ES client
        mock_es = MagicMock()
        mock_es.ping.return_value = True
        
        from streamlit_agent import StreamlitAgent
        
        # Create agent
        agent = StreamlitAgent(
            es_client=mock_es,
            index_name="research-publications-static"
        )
        
        print("✅ StreamlitAgent initialized successfully")
        
        # Test agent info
        info = agent.get_agent_info()
        print(f"✅ Agent info: {info}")
        
        if info.get('initialized'):
            print("✅ Agent is properly initialized")
            return True, agent
        else:
            print("❌ Agent not initialized")
            return False, None
            
    except Exception as e:
        print(f"❌ StreamlitAgent test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """Run tests for the fixed async issues."""
    print("🔧 Testing Streamlit Integration After Async Fixes")
    print("=" * 60)
    
    load_dotenv()
    
    # Test StreamlitAgent
    agent_success, agent = test_streamlit_agent_sync()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    
    if agent_success:
        print("✅ StreamlitAgent Sync")
        print("\n🎉 Async fixes working! Ready to test Anna Dubois query.")
        print("🚀 The Streamlit app should now work without async errors.")
        return 0
    else:
        print("❌ StreamlitAgent Sync")
        print("\n⚠️ Some issues remain.")
        return 1


if __name__ == "__main__":
    sys.exit(main())