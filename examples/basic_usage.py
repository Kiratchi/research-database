"""
Basic usage example for the research agent.
"""

import asyncio
import os
from dotenv import load_dotenv

# Add src to path for examples
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from research_agent.core.workflow import ResearchAgent
from research_agent.utils.config import load_config, create_elasticsearch_client
from research_agent.utils.logging import setup_logging


async def basic_example():
    """Basic example of using the research agent."""
    # Load environment variables
    load_dotenv()
    
    # Set up logging
    setup_logging(level="INFO")
    
    # Load configuration
    config = load_config()
    
    # Create Elasticsearch client
    es_client = create_elasticsearch_client(config.elasticsearch)
    
    # Create research agent
    agent = ResearchAgent(
        es_client=es_client,
        index_name=config.elasticsearch.index_name,
        recursion_limit=10
    )
    
    # Example queries
    queries = [
        "How many papers has Christian Fager published?",
        "Find machine learning papers from 2023",
        "What are the top 5 publication years in the database?",
        "Compare publication counts between Christian Fager and Anna Dubois"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        try:
            # Run the query
            result = await agent.query(query)
            
            # Display results
            if result.get("error"):
                print(f"❌ Error: {result['error']}")
            elif result.get("response"):
                print(f"✅ Answer: {result['response']}")
            else:
                print("❓ No response generated")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")


async def streaming_example():
    """Example of streaming query execution."""
    load_dotenv()
    setup_logging(level="INFO")
    
    config = load_config()
    es_client = create_elasticsearch_client(config.elasticsearch)
    
    agent = ResearchAgent(
        es_client=es_client,
        index_name=config.elasticsearch.index_name
    )
    
    query = "How many papers has Christian Fager published?"
    
    print(f"Streaming query: {query}")
    print("=" * 60)
    
    async for event in agent.stream_query(query):
        for k, v in event.items():
            if k != "__end__":
                print(f"\n[{k.upper()}]")
                if isinstance(v, dict):
                    if "plan" in v and v["plan"]:
                        print("📝 Plan generated:")
                        for i, step in enumerate(v["plan"], 1):
                            print(f"  {i}. {step}")
                    elif "past_steps" in v and v["past_steps"]:
                        step, result = v["past_steps"][-1]
                        print(f"⚡ Step completed: {step}")
                        print(f"📊 Result: {result[:200]}...")
                    elif "response" in v and v["response"]:
                        print(f"✅ Final answer: {v['response']}")
                print("-" * 30)


async def error_handling_example():
    """Example of error handling."""
    load_dotenv()
    setup_logging(level="DEBUG")
    
    config = load_config()
    es_client = create_elasticsearch_client(config.elasticsearch)
    
    agent = ResearchAgent(
        es_client=es_client,
        index_name=config.elasticsearch.index_name
    )
    
    # Test with an invalid query
    invalid_queries = [
        "",  # Empty query
        "What is the meaning of life?",  # Non-research query
        "Search for papers by invalid_author_name_12345",  # Non-existent author
    ]
    
    for query in invalid_queries:
        print(f"\n{'='*60}")
        print(f"Testing query: {query}")
        print(f"{'='*60}")
        
        try:
            result = await agent.query(query)
            
            if result.get("error"):
                print(f"❌ Expected error: {result['error']}")
            elif result.get("response"):
                print(f"✅ Unexpected success: {result['response']}")
            else:
                print("❓ No response")
                
        except Exception as e:
            print(f"❌ Exception caught: {str(e)}")


def main():
    """Main example runner."""
    print("Research Agent Examples")
    print("=" * 60)
    
    # Check if required environment variables are set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    if not os.getenv("ES_HOST"):
        print("❌ Please set ES_HOST environment variable")
        return
    
    # Run examples
    print("\n1. Basic Usage Example")
    asyncio.run(basic_example())
    
    print("\n2. Streaming Example")
    asyncio.run(streaming_example())
    
    print("\n3. Error Handling Example")
    asyncio.run(error_handling_example())


if __name__ == "__main__":
    main()