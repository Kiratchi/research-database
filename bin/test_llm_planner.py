"""
Test suite for LLM query planner.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from llm_planner import (
    QueryPlanner,
    ExecutionPlan,
    PlanStep,
    PlanStepType,
    create_simple_plans
)


class TestQueryPlanner:
    """Test LLM query planner functionality."""
    
    def test_query_planner_initialization(self):
        """Test QueryPlanner initialization."""
        with patch('llm_planner.ChatOpenAI') as mock_openai:
            with patch('os.getenv') as mock_getenv:
                mock_getenv.return_value = "test-api-key"
                mock_openai.return_value = Mock()
                
                planner = QueryPlanner()
                
                assert planner.model_name == "gpt-3.5-turbo"
                assert planner.temperature == 0.1
                assert planner.tools is not None
                assert len(planner.tools) > 0
    
    def test_query_planner_no_api_key(self):
        """Test QueryPlanner raises error without API key."""
        with patch('os.getenv') as mock_getenv:
            mock_getenv.return_value = None
            
            with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable not set"):
                QueryPlanner()
    
    def test_create_tool_descriptions(self):
        """Test tool descriptions creation."""
        with patch('llm_planner.ChatOpenAI') as mock_openai:
            with patch('os.getenv') as mock_getenv:
                mock_getenv.return_value = "test-api-key"
                mock_openai.return_value = Mock()
                
                planner = QueryPlanner()
                descriptions = planner._create_tool_descriptions()
                
                assert "search_publications" in descriptions
                assert "search_by_author" in descriptions
                assert "get_field_statistics" in descriptions
    
    def test_create_system_message(self):
        """Test system message creation."""
        with patch('llm_planner.ChatOpenAI') as mock_openai:
            with patch('os.getenv') as mock_getenv:
                mock_getenv.return_value = "test-api-key"
                mock_openai.return_value = Mock()
                
                planner = QueryPlanner()
                system_message = planner._create_system_message()
                
                assert "query planning assistant" in system_message
                assert "search_publications" in system_message
                assert "JSON object" in system_message
    
    @patch('llm_planner.ChatOpenAI')
    @patch('os.getenv')
    def test_plan_query_simple_author(self, mock_getenv, mock_openai):
        """Test planning a simple author query."""
        mock_getenv.return_value = "test-api-key"
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = json.dumps({
            "query_analysis": {
                "intent": "Find publications by Christian Fager",
                "complexity": 1,
                "query_type": "author_search"
            },
            "execution_plan": {
                "steps": [
                    {
                        "step_id": "step_1",
                        "tool_name": "search_by_author",
                        "parameters": {"author_name": "Christian Fager"},
                        "description": "Search for all publications by Christian Fager",
                        "depends_on": None,
                        "output_variable": "author_results"
                    }
                ],
                "final_output": "Count of publications by Christian Fager"
            }
        })
        mock_llm.return_value = mock_response
        
        planner = QueryPlanner()
        plan = planner.plan_query("How many papers has Christian Fager published?")
        
        assert isinstance(plan, ExecutionPlan)
        assert plan.query == "How many papers has Christian Fager published?"
        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "search_by_author"
        assert plan.steps[0].parameters["author_name"] == "Christian Fager"
        assert plan.estimated_complexity == 1
    
    @patch('llm_planner.ChatOpenAI')
    @patch('os.getenv')
    def test_plan_query_topic_search(self, mock_getenv, mock_openai):
        """Test planning a topic search query."""
        mock_getenv.return_value = "test-api-key"
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = json.dumps({
            "query_analysis": {
                "intent": "Find machine learning papers from 2023",
                "complexity": 2,
                "query_type": "topic_search"
            },
            "execution_plan": {
                "steps": [
                    {
                        "step_id": "step_1",
                        "tool_name": "search_publications",
                        "parameters": {"query": "machine learning", "filters": {"year": 2023}},
                        "description": "Search for machine learning publications from 2023",
                        "depends_on": None,
                        "output_variable": "ml_results"
                    }
                ],
                "final_output": "List of machine learning papers from 2023"
            }
        })
        mock_llm.return_value = mock_response
        
        planner = QueryPlanner()
        plan = planner.plan_query("Find machine learning papers from 2023")
        
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "search_publications"
        assert plan.steps[0].parameters["query"] == "machine learning"
        assert plan.steps[0].parameters["filters"]["year"] == 2023
        assert plan.estimated_complexity == 2
    
    @patch('llm_planner.ChatOpenAI')
    @patch('os.getenv')
    def test_plan_query_invalid_json(self, mock_getenv, mock_openai):
        """Test error handling for invalid JSON response."""
        mock_getenv.return_value = "test-api-key"
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        # Mock invalid JSON response
        mock_response = Mock()
        mock_response.content = "This is not valid JSON"
        mock_llm.return_value = mock_response
        
        planner = QueryPlanner()
        
        with pytest.raises(ValueError, match="Failed to parse LLM response as JSON"):
            planner.plan_query("Test query")
    
    def test_infer_step_type(self):
        """Test step type inference."""
        with patch('llm_planner.ChatOpenAI') as mock_openai:
            with patch('os.getenv') as mock_getenv:
                mock_getenv.return_value = "test-api-key"
                mock_openai.return_value = Mock()
                
                planner = QueryPlanner()
                
                assert planner._infer_step_type("search_publications") == PlanStepType.SEARCH
                assert planner._infer_step_type("search_by_author") == PlanStepType.SEARCH
                assert planner._infer_step_type("get_field_statistics") == PlanStepType.ANALYZE
                assert planner._infer_step_type("get_statistics_summary") == PlanStepType.AGGREGATE
                assert planner._infer_step_type("unknown_tool") == PlanStepType.SEARCH
    
    def test_validate_plan_success(self):
        """Test plan validation with valid plan."""
        with patch('llm_planner.ChatOpenAI') as mock_openai:
            with patch('os.getenv') as mock_getenv:
                mock_getenv.return_value = "test-api-key"
                mock_openai.return_value = Mock()
                
                planner = QueryPlanner()
                
                # Create a valid plan
                plan = ExecutionPlan(
                    query="Test query",
                    steps=[
                        PlanStep(
                            step_id="step_1",
                            step_type=PlanStepType.SEARCH,
                            tool_name="search_by_author",
                            parameters={"author_name": "Test Author"},
                            description="Search for author"
                        )
                    ],
                    final_output="Test output",
                    estimated_complexity=1
                )
                
                errors = planner.validate_plan(plan)
                assert len(errors) == 0
    
    def test_validate_plan_unknown_tool(self):
        """Test plan validation with unknown tool."""
        with patch('llm_planner.ChatOpenAI') as mock_openai:
            with patch('os.getenv') as mock_getenv:
                mock_getenv.return_value = "test-api-key"
                mock_openai.return_value = Mock()
                
                planner = QueryPlanner()
                
                # Create plan with unknown tool
                plan = ExecutionPlan(
                    query="Test query",
                    steps=[
                        PlanStep(
                            step_id="step_1",
                            step_type=PlanStepType.SEARCH,
                            tool_name="unknown_tool",
                            parameters={},
                            description="Unknown tool"
                        )
                    ],
                    final_output="Test output",
                    estimated_complexity=1
                )
                
                errors = planner.validate_plan(plan)
                assert len(errors) == 1
                assert "Unknown tool: unknown_tool" in errors[0]
    
    def test_validate_plan_missing_dependency(self):
        """Test plan validation with missing dependency."""
        with patch('llm_planner.ChatOpenAI') as mock_openai:
            with patch('os.getenv') as mock_getenv:
                mock_getenv.return_value = "test-api-key"
                mock_openai.return_value = Mock()
                
                planner = QueryPlanner()
                
                # Create plan with missing dependency
                plan = ExecutionPlan(
                    query="Test query",
                    steps=[
                        PlanStep(
                            step_id="step_1",
                            step_type=PlanStepType.SEARCH,
                            tool_name="search_publications",
                            parameters={"query": "test"},
                            description="Search",
                            depends_on=["step_0"]  # Non-existent step
                        )
                    ],
                    final_output="Test output",
                    estimated_complexity=1
                )
                
                errors = planner.validate_plan(plan)
                assert len(errors) == 1
                assert "depends on non-existent step step_0" in errors[0]
    
    def test_validate_plan_missing_parameters(self):
        """Test plan validation with missing required parameters."""
        with patch('llm_planner.ChatOpenAI') as mock_openai:
            with patch('os.getenv') as mock_getenv:
                mock_getenv.return_value = "test-api-key"
                mock_openai.return_value = Mock()
                
                planner = QueryPlanner()
                
                # Create plan with missing required parameter
                plan = ExecutionPlan(
                    query="Test query",
                    steps=[
                        PlanStep(
                            step_id="step_1",
                            step_type=PlanStepType.SEARCH,
                            tool_name="search_by_author",
                            parameters={},  # Missing author_name
                            description="Search for author"
                        )
                    ],
                    final_output="Test output",
                    estimated_complexity=1
                )
                
                errors = planner.validate_plan(plan)
                assert len(errors) == 1
                assert "search_by_author requires author_name parameter" in errors[0]
    
    def test_explain_plan(self):
        """Test plan explanation generation."""
        with patch('llm_planner.ChatOpenAI') as mock_openai:
            with patch('os.getenv') as mock_getenv:
                mock_getenv.return_value = "test-api-key"
                mock_openai.return_value = Mock()
                
                planner = QueryPlanner()
                
                # Create a plan
                plan = ExecutionPlan(
                    query="Test query",
                    steps=[
                        PlanStep(
                            step_id="step_1",
                            step_type=PlanStepType.SEARCH,
                            tool_name="search_by_author",
                            parameters={"author_name": "Test Author"},
                            description="Search for author publications"
                        )
                    ],
                    final_output="List of publications",
                    estimated_complexity=1
                )
                
                explanation = planner.explain_plan(plan)
                
                assert "Query: Test query" in explanation
                assert "Complexity: 1/5" in explanation
                assert "Steps: 1" in explanation
                assert "Search for author publications" in explanation
                assert "Tool: search_by_author" in explanation
                assert "Expected output: List of publications" in explanation
    
    def test_create_simple_plans(self):
        """Test simple plan examples."""
        examples = create_simple_plans()
        
        assert len(examples) == 3
        assert all("query" in example for example in examples)
        assert all("expected_plan" in example for example in examples)
        
        # Check first example
        first_example = examples[0]
        assert "Christian Fager" in first_example["query"]
        assert first_example["expected_plan"]["complexity"] == 1
        assert len(first_example["expected_plan"]["steps"]) == 1


class TestPlanStep:
    """Test PlanStep dataclass."""
    
    def test_plan_step_creation(self):
        """Test PlanStep creation."""
        step = PlanStep(
            step_id="test_step",
            step_type=PlanStepType.SEARCH,
            tool_name="search_publications",
            parameters={"query": "test"},
            description="Test step"
        )
        
        assert step.step_id == "test_step"
        assert step.step_type == PlanStepType.SEARCH
        assert step.tool_name == "search_publications"
        assert step.parameters == {"query": "test"}
        assert step.description == "Test step"
        assert step.depends_on is None
        assert step.output_variable is None


class TestExecutionPlan:
    """Test ExecutionPlan dataclass."""
    
    def test_execution_plan_creation(self):
        """Test ExecutionPlan creation."""
        steps = [
            PlanStep(
                step_id="step_1",
                step_type=PlanStepType.SEARCH,
                tool_name="search_publications",
                parameters={"query": "test"},
                description="Test step"
            )
        ]
        
        plan = ExecutionPlan(
            query="Test query",
            steps=steps,
            final_output="Test output",
            estimated_complexity=2
        )
        
        assert plan.query == "Test query"
        assert len(plan.steps) == 1
        assert plan.steps[0].step_id == "step_1"
        assert plan.final_output == "Test output"
        assert plan.estimated_complexity == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])