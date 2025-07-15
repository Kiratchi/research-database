#!/usr/bin/env python3
"""
Test the Pydantic model fixes
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.research_agent.core.models import Act, Plan, Response

def test_act_model():
    """Test the Act model with new structure."""
    print("🧪 Testing Act model with new structure...")
    
    # Test response action
    try:
        act_response = Act(
            action_type="response",
            response="Here are the 2023 machine learning papers.",
            steps=None
        )
        print(f"✅ Response action: {act_response.action_type}")
        print(f"   Response: {act_response.response}")
    except Exception as e:
        print(f"❌ Response action failed: {e}")
    
    # Test plan action
    try:
        act_plan = Act(
            action_type="plan",
            response=None,
            steps=["Filter results by year 2023", "Present findings"]
        )
        print(f"✅ Plan action: {act_plan.action_type}")
        print(f"   Steps: {act_plan.steps}")
    except Exception as e:
        print(f"❌ Plan action failed: {e}")

if __name__ == "__main__":
    test_act_model()