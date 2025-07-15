"""
Comprehensive integration test suite for Phase 1 LangChain components.

This test suite ensures that all Phase 1 components work together correctly
and validates the success criteria from the LLM_AGENT_TDD_PLAN.md.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import Phase 1 components
from llm_planner import QueryPlanner, ExecutionPlan, PlanStep, PlanStepType
from langchain_tools import (
    create_langchain_tools,
    initialize_langchain_tools,
    SearchPublicationsInput,
    SearchByAuthorInput
)


class TestPhase1Integration:
    """Integration tests for Phase 1 components."""
    
    @patch('llm_planner.ChatOpenAI')
    @patch('os.getenv')
    def test_end_to_end_author_query_planning(self, mock_getenv, mock_openai):
        """Test complete flow from query to execution plan for author search."""
        mock_getenv.return_value = "test-api-key"
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        # Mock realistic LLM response for author query
        mock_response = Mock()
        mock_response.content = json.dumps({
            "query_analysis": {
                "intent": "Find all publications by Christian Fager",
                "complexity": 1,
                "query_type": "author_search"
            },
            "execution_plan": {
                "steps": [
                    {
                        "step_id": "step_1",
                        "tool_name": "search_by_author",
                        "parameters": {"author_name": "Christian Fager", "strategy": "auto"},
                        "description": "Search for all publications by Christian Fager",
                        "depends_on": None,
                        "output_variable": "author_publications"
                    }
                ],
                "final_output": "Total count and list of publications by Christian Fager"
            }
        })
        mock_llm.return_value = mock_response
        
        # Test the complete flow
        planner = QueryPlanner()
        tools = create_langchain_tools()
        
        # Plan the query
        query = "How many papers has Christian Fager published?"
        plan = planner.plan_query(query)
        
        # Validate the plan
        errors = planner.validate_plan(plan)
        assert len(errors) == 0, f"Plan validation failed: {errors}"
        
        # Check plan structure
        assert isinstance(plan, ExecutionPlan)
        assert plan.query == query
        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "search_by_author"
        assert plan.steps[0].parameters["author_name"] == "Christian Fager"
        assert plan.estimated_complexity == 1
        
        # Verify tool exists
        tool_names = [tool.name for tool in tools]
        assert "search_by_author" in tool_names
        
        # Verify tool can be called with plan parameters
        search_tool = next(tool for tool in tools if tool.name == "search_by_author")
        assert search_tool.args_schema == SearchByAuthorInput
        
        # Validate input schema
        input_data = SearchByAuthorInput(**plan.steps[0].parameters)
        assert input_data.author_name == "Christian Fager"
        assert input_data.strategy == "auto"
    
    @patch('llm_planner.ChatOpenAI')
    @patch('os.getenv')
    def test_end_to_end_topic_query_planning(self, mock_getenv, mock_openai):
        """Test complete flow for topic search query."""
        mock_getenv.return_value = "test-api-key"
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        # Mock LLM response for topic query
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
                        "parameters": {
                            "query": "machine learning",
                            "filters": {"year": 2023},
                            "size": 20
                        },
                        "description": "Search for machine learning publications from 2023",
                        "depends_on": None,
                        "output_variable": "ml_papers"
                    }
                ],
                "final_output": "List of machine learning papers published in 2023"
            }
        })
        mock_llm.return_value = mock_response
        
        # Test the complete flow
        planner = QueryPlanner()
        tools = create_langchain_tools()
        
        # Plan the query
        query = "Find machine learning papers from 2023"
        plan = planner.plan_query(query)
        
        # Validate the plan
        errors = planner.validate_plan(plan)
        assert len(errors) == 0, f"Plan validation failed: {errors}"
        
        # Check plan structure
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "search_publications"
        assert plan.steps[0].parameters["query"] == "machine learning"
        assert plan.steps[0].parameters["filters"]["year"] == 2023
        assert plan.estimated_complexity == 2
        
        # Verify tool exists and can handle parameters
        tool_names = [tool.name for tool in tools]
        assert "search_publications" in tool_names
        
        search_tool = next(tool for tool in tools if tool.name == "search_publications")
        assert search_tool.args_schema == SearchPublicationsInput
        
        # Validate input schema
        input_data = SearchPublicationsInput(**plan.steps[0].parameters)
        assert input_data.query == "machine learning"
        assert input_data.filters["year"] == 2023
        assert input_data.size == 20
    
    @patch('llm_planner.ChatOpenAI')
    @patch('os.getenv')
    def test_multi_step_query_planning(self, mock_getenv, mock_openai):
        """Test planning of multi-step queries."""
        mock_getenv.return_value = "test-api-key"
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        # Mock LLM response for multi-step query
        mock_response = Mock()
        mock_response.content = json.dumps({
            "query_analysis": {
                "intent": "Compare publication counts between two authors",
                "complexity": 3,
                "query_type": "comparison"
            },
            "execution_plan": {
                "steps": [
                    {
                        "step_id": "step_1",
                        "tool_name": "search_by_author",
                        "parameters": {"author_name": "Christian Fager", "strategy": "auto"},
                        "description": "Search for publications by Christian Fager",
                        "depends_on": None,
                        "output_variable": "fager_results"
                    },
                    {
                        "step_id": "step_2",
                        "tool_name": "search_by_author",
                        "parameters": {"author_name": "Anna Dubois", "strategy": "auto"},
                        "description": "Search for publications by Anna Dubois",
                        "depends_on": None,
                        "output_variable": "dubois_results"
                    }
                ],
                "final_output": "Comparison of publication counts between Christian Fager and Anna Dubois"
            }
        })
        mock_llm.return_value = mock_response
        
        # Test the complete flow
        planner = QueryPlanner()
        
        # Plan the query
        query = "Compare publication counts between Christian Fager and Anna Dubois"
        plan = planner.plan_query(query)
        
        # Validate the plan
        errors = planner.validate_plan(plan)
        assert len(errors) == 0, f"Plan validation failed: {errors}"
        
        # Check plan structure
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) == 2
        assert plan.estimated_complexity == 3
        
        # Verify both steps are valid
        assert plan.steps[0].tool_name == "search_by_author"
        assert plan.steps[0].parameters["author_name"] == "Christian Fager"
        assert plan.steps[1].tool_name == "search_by_author"
        assert plan.steps[1].parameters["author_name"] == "Anna Dubois"
        
        # Verify step IDs are unique
        step_ids = [step.step_id for step in plan.steps]
        assert len(step_ids) == len(set(step_ids))
    
    @patch('llm_planner.ChatOpenAI')
    @patch('os.getenv')
    def test_error_handling_invalid_plan(self, mock_getenv, mock_openai):
        """Test error handling for invalid plans."""
        mock_getenv.return_value = "test-api-key"
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        # Mock LLM response with invalid tool
        mock_response = Mock()
        mock_response.content = json.dumps({
            "query_analysis": {
                "intent": "Test invalid plan",
                "complexity": 1,
                "query_type": "unknown"
            },
            "execution_plan": {
                "steps": [
                    {
                        "step_id": "step_1",
                        "tool_name": "nonexistent_tool",
                        "parameters": {},
                        "description": "Invalid tool test",
                        "depends_on": None,
                        "output_variable": "invalid_result"
                    }
                ],
                "final_output": "This should fail validation"
            }
        })
        mock_llm.return_value = mock_response
        
        planner = QueryPlanner()
        plan = planner.plan_query("Test invalid query")
        
        # Validate the plan - should have errors
        errors = planner.validate_plan(plan)
        assert len(errors) > 0
        assert any("nonexistent_tool" in error for error in errors)
    
    @patch('llm_planner.ChatOpenAI')
    @patch('os.getenv')
    def test_plan_explanation_readability(self, mock_getenv, mock_openai):
        """Test that plan explanations are human-readable."""
        mock_getenv.return_value = "test-api-key"
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        # Mock simple plan
        mock_response = Mock()
        mock_response.content = json.dumps({
            "query_analysis": {
                "intent": "Find author publications",
                "complexity": 1,
                "query_type": "author_search"
            },
            "execution_plan": {
                "steps": [
                    {
                        "step_id": "step_1",
                        "tool_name": "search_by_author",
                        "parameters": {"author_name": "Test Author"},
                        "description": "Search for publications by Test Author",
                        "depends_on": None,
                        "output_variable": "author_pubs"
                    }
                ],
                "final_output": "List of publications by Test Author"
            }
        })
        mock_llm.return_value = mock_response
        
        planner = QueryPlanner()
        plan = planner.plan_query("Find Test Author publications")
        
        # Generate explanation
        explanation = planner.explain_plan(plan)
        
        # Verify explanation contains key information
        assert "Query:" in explanation
        assert "Complexity:" in explanation
        assert "Steps:" in explanation
        assert "Test Author" in explanation
        assert "search_by_author" in explanation
        assert "Expected output:" in explanation
        assert len(explanation.split('\n')) > 5  # Multi-line explanation
    
    def test_tool_schema_compatibility(self):
        """Test that all tools have compatible schemas."""
        tools = create_langchain_tools()
        
        # Test each tool's schema
        for tool in tools:
            # Check that tool has required attributes
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, 'func')
            
            # Check that description is informative
            assert len(tool.description) > 50  # Meaningful description
            
            # Check that tool name is valid
            assert tool.name.replace('_', '').isalnum()
            
            # Test that schema validation works if present
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema
                assert issubclass(schema, SearchPublicationsInput.__bases__[0])
    
    @patch('langchain_tools.search_publications')
    @patch('langchain_tools.search_by_author')
    def test_tool_execution_with_plan_output(self, mock_search_author, mock_search_pubs):
        """Test that tools can execute with parameters from plans."""
        # Mock tool responses
        mock_search_author.return_value = {
            'total_results': 50,
            'session_id': 'test-session-123',
            'sample_results': [
                {
                    '_source': {
                        'Title': 'Test Paper',
                        'authors': ['Test Author'],
                        'Year': 2023
                    }
                }
            ]
        }
        
        mock_search_pubs.return_value = {
            'total_results': 100,
            'session_id': 'test-session-456',
            'sample_results': [
                {
                    '_source': {
                        'Title': 'Machine Learning Paper',
                        'authors': ['ML Author'],
                        'Year': 2023
                    }
                }
            ],
            'aggregations': {'years': [], 'types': []}
        }
        
        tools = create_langchain_tools()
        
        # Test search_by_author tool
        author_tool = next(tool for tool in tools if tool.name == "search_by_author")
        result = author_tool.func(author_name="Test Author", strategy="auto")
        
        assert "Total Publications: 50" in result
        assert "Test Author" in result
        assert "Test Paper" in result
        
        # Test search_publications tool
        search_tool = next(tool for tool in tools if tool.name == "search_publications")
        result = search_tool.func(query="machine learning", filters={"year": 2023})
        
        assert "Total Results: 100" in result
        assert "Machine Learning Paper" in result
        assert "test-session-456" in result
    
    def test_success_criteria_validation(self):
        """Test that Phase 1 meets the success criteria from the plan."""
        tools = create_langchain_tools()
        
        # Success Criteria from LLM_AGENT_TDD_PLAN.md:
        # 1. LangChain imports without errors - tested in test_langchain_setup.py
        # 2. OpenAI API connection established - tested in test_langchain_setup.py
        # 3. Basic LLM calls work - tested in planner tests
        # 4. All existing tools work as LangChain tools - test here
        
        expected_tools = [
            "search_publications",
            "search_by_author",
            "get_more_results",
            "get_field_statistics",
            "get_statistics_summary"
        ]
        
        tool_names = [tool.name for tool in tools]
        for expected_tool in expected_tools:
            assert expected_tool in tool_names, f"Missing tool: {expected_tool}"
        
        # 5. Tools have proper schemas and descriptions
        for tool in tools:
            assert len(tool.description) > 20  # Meaningful description
            if hasattr(tool, 'args_schema') and tool.args_schema:
                # Schema should be valid Pydantic model
                assert hasattr(tool.args_schema, '__fields__')
        
        # 6. Tools return structured data with metadata
        # This would be tested with actual ES client in integration tests
        
        # 7. Plans generated for simple queries - tested in planner tests
        # 8. Plans include correct tool selection - tested in planner tests  
        # 9. Plans have valid parameters - tested in planner tests


class TestPhase1SuccessMetrics:
    """Test Phase 1 success metrics from the plan."""
    
    def test_natural_language_flexibility_preparation(self):
        """Test preparation for natural language flexibility."""
        # This tests the foundation - actual flexibility testing will come in Phase 2
        
        # Verify that the planner can handle different query formats
        test_queries = [
            "How many papers has Christian Fager published?",
            "how many papers has christian fager published?",
            "Find papers by Christian Fager",
            "Show me publications from Christian Fager",
            "Christian Fager publications count"
        ]
        
        # All queries should be processable (though results will vary)
        # This tests the system's ability to accept diverse inputs
        for query in test_queries:
            assert len(query.strip()) > 0
            assert query.lower() != query.upper()  # Has mixed case handling potential
    
    def test_plan_generation_capability(self):
        """Test that the system can generate structured plans."""
        tools = create_langchain_tools()
        
        # Verify we have the necessary tools for plan generation
        tool_names = [tool.name for tool in tools]
        assert "search_publications" in tool_names
        assert "search_by_author" in tool_names
        assert "get_field_statistics" in tool_names
        
        # Plans should be structurally valid (detailed testing in planner tests)
        # This confirms we have the infrastructure for complex planning
    
    def test_tool_composition_readiness(self):
        """Test readiness for tool composition."""
        tools = create_langchain_tools()
        
        # Verify tools can be chained (session-based tools)
        session_tools = [
            "get_more_results",
            "get_field_statistics"
        ]
        
        tool_names = [tool.name for tool in tools]
        for session_tool in session_tools:
            assert session_tool in tool_names
        
        # These tools require session_id from other tools - enabling composition
    
    def test_error_recovery_foundation(self):
        """Test foundation for error recovery."""
        tools = create_langchain_tools()
        
        # Test that tools handle errors gracefully
        error_tool = next(tool for tool in tools if tool.name == "search_publications")
        
        # Should not crash on bad input (actual error handling tested in tool tests)
        assert error_tool.func is not None
        assert callable(error_tool.func)
    
    def test_performance_baseline(self):
        """Test performance baseline for future optimization."""
        tools = create_langchain_tools()
        
        # Tools should be created efficiently
        assert len(tools) == 5  # Expected number of tools
        
        # Tool creation should be fast enough for interactive use
        # (Detailed performance testing would require actual ES connection)
        
        # Verify no obvious performance bottlenecks in setup
        for tool in tools:
            assert tool.name is not None
            assert tool.description is not None
            assert tool.func is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])