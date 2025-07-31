"""
Research Publications Agent
A LangGraph-based plan-and-execute agent for querying research publications
using Elasticsearch with natural language processing capabilities.

Features:
- Smart methodology learning with fast structured logging
- Clean prompt templates in separate TXT files
- Complete research context preservation
- Session-aware conversation management
- No GeneratorExit errors - graceful stream completion
- Standard LangGraph implementation for reliability
"""

__version__ = "1.0.0"  # Updated version for production-ready implementation
__author__ = "Research Agent Team"

# Import core workflow components (models and state are now embedded in workflow)
from .core.workflow import (
    ResearchAgent, 
    run_research_query, 
    compile_research_agent,
    PlanExecuteState,  # Now imported from workflow where it's defined
    Plan,             # Now imported from workflow where it's defined
    Response,         # Now imported from workflow where it's defined
    Act              # Now imported from workflow where it's defined
)

# Import agent management
from .core.agent_manager import AgentManager, create_agent_manager

# Import memory system
from .core.memory_manager import IntegratedMemoryManager, create_memory_manager

# Import fast methodology logging
from .core.methodology_logger import StandardMethodologyLogger, create_standard_methodology_logger

# Import tools system
from .tools import get_all_tools

# Import prompt system
from .prompts import (
    PLANNING_PROMPT_TEMPLATE,
    EXECUTION_PROMPT_TEMPLATE,
    REPLANNING_PROMPT_TEMPLATE,
    get_prompt_template
)

__all__ = [
    # Core workflow
    "ResearchAgent",
    "run_research_query", 
    "compile_research_agent",
    
    # State and models (now from workflow)
    "PlanExecuteState",
    "Plan",
    "Response", 
    "Act",
    
    # Management systems
    "AgentManager",
    "create_agent_manager",
    "IntegratedMemoryManager",
    "create_memory_manager",
    
    # Fast logging system
    "StandardMethodologyLogger",
    "create_standard_methodology_logger",
    
    # Tools system
    "get_all_tools",
    
    # Prompt system
    "PLANNING_PROMPT_TEMPLATE",
    "EXECUTION_PROMPT_TEMPLATE", 
    "REPLANNING_PROMPT_TEMPLATE",
    "get_prompt_template",
]


# Version info tuple for programmatic access
VERSION_INFO = tuple(map(int, __version__.split('.')))

def get_version_info():
    """Get detailed version information."""
    return {
        "version": __version__,
        "version_info": VERSION_INFO,
        "features": [
            "LangGraph plan-and-execute workflow",
            "Fast structured methodology logging (no LLM overhead)",
            "External TXT prompt templates", 
            "Complete research context preservation",
            "Session-aware conversation management",
            "Graceful async stream completion",
            "Smart memory management without fact extraction bottleneck",
            "Production-ready architecture"
        ],
        "architecture": "standard_fast_logging"
    }

# Convenience imports for common usage patterns
def create_research_system(es_client=None, index_name="research-publications-static"):
    """
    Create a complete research system with all components.
    
    Args:
        es_client: Elasticsearch client (optional)
        index_name: Elasticsearch index name
        
    Returns:
        Configured AgentManager ready for use
    """
    return create_agent_manager(index_name=index_name)

# Print info on import (optional - remove if too verbose)
def _print_startup_info():
    """Print startup information."""
    info = get_version_info()
    print(f"🚀 Research Agent v{info['version']} - {info['status']}")
    print(f"⚡ Architecture: {info['architecture']}")
    print(f"🎯 Key features: Fast logging, Clean prompts, Production-ready")

# Uncomment if you want startup info
# _print_startup_info()