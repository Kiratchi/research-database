"""
Configuration management for the research agent.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class ElasticsearchConfig:
    """Configuration for Elasticsearch connection."""
    host: str
    username: Optional[str] = None
    password: Optional[str] = None
    index_name: str = "research-publications-static"
    timeout: int = 30
    verify_certs: bool = True


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI API."""
    api_key: str
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: Optional[int] = None


@dataclass
class AgentConfig:
    """Configuration for the research agent."""
    recursion_limit: int = 50
    max_steps: int = 10
    timeout: int = 300  # 5 minutes
    stream_results: bool = False
    debug: bool = False


@dataclass
class ResearchAgentConfig:
    """Complete configuration for the research agent."""
    elasticsearch: ElasticsearchConfig
    openai: OpenAIConfig
    agent: AgentConfig


def load_config() -> ResearchAgentConfig:
    """
    Load configuration from environment variables and defaults.
    
    Returns:
        Complete configuration object
    """
    # Load environment variables
    load_dotenv()
    
    # Elasticsearch configuration
    es_config = ElasticsearchConfig(
        host=os.getenv("ES_HOST", "localhost:9200"),
        username=os.getenv("ES_USER"),
        password=os.getenv("ES_PASS"),
        index_name=os.getenv("ES_INDEX", "research-publications-static"),
        timeout=int(os.getenv("ES_TIMEOUT", "30")),
        verify_certs=os.getenv("ES_VERIFY_CERTS", "true").lower() == "true"
    )
    
    # OpenAI configuration
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    openai_config = OpenAIConfig(
        api_key=openai_api_key,
        model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS")) if os.getenv("OPENAI_MAX_TOKENS") else None
    )
    
    # Agent configuration
    agent_config = AgentConfig(
        recursion_limit=int(os.getenv("AGENT_RECURSION_LIMIT", "50")),
        max_steps=int(os.getenv("AGENT_MAX_STEPS", "10")),
        timeout=int(os.getenv("AGENT_TIMEOUT", "300")),
        stream_results=os.getenv("AGENT_STREAM", "false").lower() == "true",
        debug=os.getenv("AGENT_DEBUG", "false").lower() == "true"
    )
    
    return ResearchAgentConfig(
        elasticsearch=es_config,
        openai=openai_config,
        agent=agent_config
    )


def create_elasticsearch_client(config: ElasticsearchConfig):
    """
    Create an Elasticsearch client from configuration.
    
    Args:
        config: Elasticsearch configuration
        
    Returns:
        Elasticsearch client instance
    """
    try:
        from elasticsearch import Elasticsearch
        
        # Build connection parameters
        es_params = {
            "hosts": [config.host],
            "timeout": config.timeout,
            "verify_certs": config.verify_certs
        }
        
        # Add authentication if provided
        if config.username and config.password:
            es_params["basic_auth"] = (config.username, config.password)
        
        # Create client
        es_client = Elasticsearch(**es_params)
        
        # Test connection
        if not es_client.ping():
            raise ConnectionError("Failed to connect to Elasticsearch")
        
        return es_client
    
    except ImportError:
        raise ImportError("elasticsearch package is required. Install with: pip install elasticsearch")
    except Exception as e:
        raise ConnectionError(f"Failed to create Elasticsearch client: {str(e)}")


def get_environment_info() -> Dict[str, Any]:
    """
    Get information about the current environment.
    
    Returns:
        Dictionary with environment information
    """
    return {
        "python_version": os.sys.version,
        "working_directory": os.getcwd(),
        "environment_variables": {
            "ES_HOST": os.getenv("ES_HOST", "Not set"),
            "ES_INDEX": os.getenv("ES_INDEX", "Not set"),
            "OPENAI_API_KEY": "Set" if os.getenv("OPENAI_API_KEY") else "Not set",
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "Not set"),
        }
    }


if __name__ == "__main__":
    # Example usage
    try:
        config = load_config()
        print("Configuration loaded successfully:")
        print(f"ES Host: {config.elasticsearch.host}")
        print(f"ES Index: {config.elasticsearch.index_name}")
        print(f"OpenAI Model: {config.openai.model_name}")
        print(f"Agent Recursion Limit: {config.agent.recursion_limit}")
        
        # Test ES connection
        es_client = create_elasticsearch_client(config.elasticsearch)
        print("Elasticsearch connection successful!")
        
    except Exception as e:
        print(f"Configuration error: {e}")
        print("\nEnvironment info:")
        env_info = get_environment_info()
        for key, value in env_info["environment_variables"].items():
            print(f"  {key}: {value}")