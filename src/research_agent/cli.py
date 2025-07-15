"""
Command-line interface for the research agent.
"""

import asyncio
import sys
from typing import Optional
import argparse
from pathlib import Path

from .core.workflow import ResearchAgent
from .utils.config import load_config, create_elasticsearch_client
from .utils.logging import setup_logging, get_logger


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Research Publications Agent - Query research publications using natural language"
    )
    
    # Query options
    parser.add_argument(
        "query",
        nargs="?",
        help="Natural language query about research publications"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--stream", "-s",
        action="store_true",
        help="Stream results as they're generated"
    )
    
    # Configuration options
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)"
    )
    
    parser.add_argument(
        "--recursion-limit", "-r",
        type=int,
        default=50,
        help="Maximum recursion limit for agent execution (default: 50)"
    )
    
    # Elasticsearch options
    parser.add_argument(
        "--es-host",
        type=str,
        help="Elasticsearch host (overrides config)"
    )
    
    parser.add_argument(
        "--es-index",
        type=str,
        help="Elasticsearch index name (overrides config)"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file"
    )
    
    # Utility options
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 0.2.0"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Check configuration and exit"
    )
    
    return parser


async def run_query(agent: ResearchAgent, query: str, stream: bool = False) -> None:
    """Run a single query."""
    logger = get_logger()
    
    print(f"Query: {query}")
    print("=" * 50)
    
    try:
        if stream:
            # Stream the results
            async for event in agent.stream_query(query):
                for k, v in event.items():
                    if k != "__end__":
                        print(f"\n[{k.upper()}]")
                        if isinstance(v, dict):
                            if "plan" in v and v["plan"]:
                                print("Plan generated:")
                                for i, step in enumerate(v["plan"], 1):
                                    print(f"  {i}. {step}")
                            elif "past_steps" in v and v["past_steps"]:
                                step, result = v["past_steps"][-1]
                                print(f"Step completed: {step}")
                                print(f"Result: {result[:200]}...")
                            elif "response" in v and v["response"]:
                                print(f"Final answer: {v['response']}")
                        print("-" * 30)
        else:
            # Run synchronously
            result = await agent.query(query)
            
            if result.get("error"):
                print(f"Error: {result['error']}")
            elif result.get("response"):
                print(f"Answer: {result['response']}")
            else:
                print("No response generated")
                
    except Exception as e:
        logger.log_error(f"Query failed: {str(e)}")
        print(f"Error: {str(e)}")


async def interactive_mode(agent: ResearchAgent, stream: bool = False) -> None:
    """Run in interactive mode."""
    print("Research Agent Interactive Mode")
    print("Type 'quit' or 'exit' to exit")
    print("Type 'help' for available commands")
    print("=" * 50)
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            
            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if query.lower() in ["help", "h"]:
                print("\nAvailable commands:")
                print("  help, h - Show this help message")
                print("  quit, exit, q - Exit the program")
                print("  Any other input will be treated as a query")
                continue
            
            if not query:
                continue
            
            await run_query(agent, query, stream)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


def check_configuration() -> bool:
    """Check configuration and return True if valid."""
    try:
        # Load configuration
        config = load_config()
        print("✓ Configuration loaded successfully")
        
        # Test Elasticsearch connection
        es_client = create_elasticsearch_client(config.elasticsearch)
        print("✓ Elasticsearch connection successful")
        
        # Test OpenAI configuration
        if config.openai.api_key:
            print("✓ OpenAI API key configured")
        else:
            print("✗ OpenAI API key not configured")
            return False
        
        print(f"✓ All configuration checks passed")
        return True
        
    except Exception as e:
        print(f"✗ Configuration check failed: {str(e)}")
        return False


async def main() -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(
        level=args.log_level,
        log_file=args.log_file
    )
    
    # Check configuration if requested
    if args.check_config:
        if check_configuration():
            sys.exit(0)
        else:
            sys.exit(1)
    
    # Load configuration
    try:
        config = load_config()
        
        # Override with command-line arguments
        if args.es_host:
            config.elasticsearch.host = args.es_host
        if args.es_index:
            config.elasticsearch.index_name = args.es_index
        if args.model:
            config.openai.model_name = args.model
        if args.recursion_limit:
            config.agent.recursion_limit = args.recursion_limit
        if args.debug:
            config.agent.debug = True
        
        # Create Elasticsearch client
        es_client = create_elasticsearch_client(config.elasticsearch)
        
        # Create research agent
        agent = ResearchAgent(
            es_client=es_client,
            index_name=config.elasticsearch.index_name,
            recursion_limit=config.agent.recursion_limit
        )
        
        # Run based on mode
        if args.interactive:
            await interactive_mode(agent, args.stream)
        elif args.query:
            await run_query(agent, args.query, args.stream)
        else:
            print("Please provide a query or use --interactive mode")
            parser.print_help()
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


def sync_main() -> None:
    """Synchronous wrapper for main."""
    asyncio.run(main())


if __name__ == "__main__":
    sync_main()