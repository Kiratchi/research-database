"""
Tests for core models.
"""

import pytest
from research_agent.core.models import Plan, Response, Act


class TestPlan:
    """Test Plan model."""
    
    def test_plan_creation(self):
        """Test Plan creation with valid data."""
        steps = ["Step 1", "Step 2", "Step 3"]
        plan = Plan(steps=steps)
        
        assert plan.steps == steps
        assert isinstance(plan.steps, list)
        assert len(plan.steps) == 3
    
    def test_plan_empty_steps(self):
        """Test Plan with empty steps."""
        plan = Plan(steps=[])
        assert plan.steps == []
        assert isinstance(plan.steps, list)
    
    def test_plan_validation(self):
        """Test Plan validation."""
        # Valid plan
        plan = Plan(steps=["Search", "Analyze"])
        assert len(plan.steps) == 2
        
        # Test with None should raise validation error
        with pytest.raises(ValueError):
            Plan(steps=None)


class TestResponse:
    """Test Response model."""
    
    def test_response_creation(self):
        """Test Response creation with valid data."""
        response_text = "This is a test response."
        response = Response(response=response_text)
        
        assert response.response == response_text
        assert isinstance(response.response, str)
    
    def test_response_empty_string(self):
        """Test Response with empty string."""
        response = Response(response="")
        assert response.response == ""
        assert isinstance(response.response, str)
    
    def test_response_validation(self):
        """Test Response validation."""
        # Valid response
        response = Response(response="Valid response")
        assert response.response == "Valid response"
        
        # Test with None should raise validation error
        with pytest.raises(ValueError):
            Response(response=None)


class TestAct:
    """Test Act model."""
    
    def test_act_with_plan(self):
        """Test Act with Plan action."""
        plan = Plan(steps=["Step 1", "Step 2"])
        act = Act(action=plan)
        
        assert act.action == plan
        assert isinstance(act.action, Plan)
    
    def test_act_with_response(self):
        """Test Act with Response action."""
        response = Response(response="Test response")
        act = Act(action=response)
        
        assert act.action == response
        assert isinstance(act.action, Response)
    
    def test_act_validation(self):
        """Test Act validation."""
        # Valid Act with Plan
        plan = Plan(steps=["Step 1"])
        act = Act(action=plan)
        assert isinstance(act.action, Plan)
        
        # Valid Act with Response
        response = Response(response="Test")
        act = Act(action=response)
        assert isinstance(act.action, Response)
        
        # Test with None should raise validation error
        with pytest.raises(ValueError):
            Act(action=None)


class TestModelIntegration:
    """Test model integration."""
    
    def test_plan_to_act_conversion(self):
        """Test converting Plan to Act."""
        plan = Plan(steps=["Search", "Analyze"])
        act = Act(action=plan)
        
        assert isinstance(act.action, Plan)
        assert act.action.steps == ["Search", "Analyze"]
    
    def test_response_to_act_conversion(self):
        """Test converting Response to Act."""
        response = Response(response="Final answer")
        act = Act(action=response)
        
        assert isinstance(act.action, Response)
        assert act.action.response == "Final answer"
    
    def test_model_serialization(self):
        """Test model serialization."""
        # Test Plan serialization
        plan = Plan(steps=["Step 1", "Step 2"])
        plan_dict = plan.model_dump()
        
        assert "steps" in plan_dict
        assert plan_dict["steps"] == ["Step 1", "Step 2"]
        
        # Test Response serialization
        response = Response(response="Test response")
        response_dict = response.model_dump()
        
        assert "response" in response_dict
        assert response_dict["response"] == "Test response"
    
    def test_model_deserialization(self):
        """Test model deserialization."""
        # Test Plan deserialization
        plan_data = {"steps": ["Step 1", "Step 2"]}
        plan = Plan(**plan_data)
        
        assert plan.steps == ["Step 1", "Step 2"]
        
        # Test Response deserialization
        response_data = {"response": "Test response"}
        response = Response(**response_data)
        
        assert response.response == "Test response"