"""
Test suite for LangChain infrastructure setup.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

load_dotenv()


class TestLangChainSetup:
    """Test LangChain infrastructure setup."""
    
    def test_langchain_imports(self):
        """Test all required LangChain components import correctly."""
        try:
            from langchain.agents import AgentType, initialize_agent
            from langchain.llms import OpenAI
            from langchain.schema import AgentAction, AgentFinish
            from langchain.tools import Tool
            from langchain_openai import ChatOpenAI
            from pydantic import BaseModel, Field
        except ImportError as e:
            pytest.fail(f"LangChain import failed: {e}")
    
    def test_pydantic_imports(self):
        """Test Pydantic imports for tool schemas."""
        try:
            from pydantic import BaseModel, Field
            from typing import Optional, Dict, Any
            
            # Test basic Pydantic model creation
            class TestModel(BaseModel):
                name: str = Field(description="Test name")
                value: Optional[int] = Field(None, description="Test value")
            
            # Test model instantiation
            model = TestModel(name="test")
            assert model.name == "test"
            assert model.value is None
            
        except Exception as e:
            pytest.fail(f"Pydantic setup failed: {e}")
    
    @patch('openai.OpenAI')
    def test_openai_client_creation(self, mock_openai):
        """Test OpenAI client can be created (mocked)."""
        try:
            from langchain_openai import ChatOpenAI
            
            # Mock the OpenAI client
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            # Test ChatOpenAI creation
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                openai_api_key="test-key"
            )
            
            assert llm.model_name == "gpt-3.5-turbo"
            assert llm.temperature == 0.1
            
        except Exception as e:
            pytest.fail(f"OpenAI client creation failed: {e}")
    
    def test_environment_variables(self):
        """Test that required environment variables are accessible."""
        # Check if .env file exists or env vars are set
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            pytest.skip("OPENAI_API_KEY not set - skipping OpenAI connection test")
        
        # Test that we can access other required env vars
        es_host = os.getenv('ES_HOST')
        if not es_host:
            pytest.skip("ES_HOST not set - this is required for full functionality")
    
    def test_tool_schema_creation(self):
        """Test that we can create tool schemas using Pydantic."""
        try:
            from pydantic import BaseModel, Field
            from typing import Optional, Dict
            
            # Test schema for search_by_author tool
            class SearchByAuthorInput(BaseModel):
                author_name: str = Field(description="Name of the author to search for")
                year_range: Optional[Dict[str, int]] = Field(
                    None, 
                    description="Optional year range filter, e.g., {'gte': 2020, 'lte': 2024}"
                )
                strategy: str = Field(
                    "auto",
                    description="Search strategy: 'exact', 'partial', 'fuzzy', or 'auto'"
                )
            
            # Test schema validation
            valid_input = SearchByAuthorInput(author_name="Christian Fager")
            assert valid_input.author_name == "Christian Fager"
            assert valid_input.strategy == "auto"
            
            # Test with year range
            input_with_range = SearchByAuthorInput(
                author_name="Anna Dubois",
                year_range={"gte": 2020, "lte": 2024}
            )
            assert input_with_range.year_range == {"gte": 2020, "lte": 2024}
            
        except Exception as e:
            pytest.fail(f"Tool schema creation failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])