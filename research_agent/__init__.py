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

UPDATED: Fixed imports for Flask app compatibility with absolute imports in workflow.py
"""

__version__ = "1.0.0"  # Updated version for production-ready implementation
__author__ = "Research Agent Team"

# UPDATED: Import core workflow components with error handling for absolute/relative import compatibility
try:
    # Try relative imports first (for internal package use)
    from .core.workflow import (
        ResearchAgent,
        run_research_query,
        compile_research_agent,
        PlanExecuteState,  # Now imported from workflow where it's defined
        Plan,             # Now imported from workflow where it's defined
        Response,         # Now imported from workflow where it's defined
        Act              # Now imported from workflow where it's defined
    )
except ImportError:
    # Fall back to absolute imports (for Flask app compatibility)
    from research_agent.core.workflow import (
        ResearchAgent,
        run_research_query,
        compile_research_agent,
        PlanExecuteState,
        Plan,
        Response,
        Act
    )

# UPDATED: Import agent management with error handling
try:
    from .core.agent_manager import AgentManager, create_agent_manager
except ImportError:
    from research_agent.core.agent_manager import AgentManager, create_agent_manager

# UPDATED: Import memory system with error handling
try:
    from .core.memory_manager import IntegratedMemoryManager, create_memory_manager
except ImportError:
    from research_agent.core.memory_manager import IntegratedMemoryManager, create_memory_manager

# UPDATED: Import fast methodology logging with error handling
try:
    from .core.methodology_logger import StandardMethodologyLogger, create_standard_methodology_logger
except ImportError:
    from research_agent.core.methodology_logger import StandardMethodologyLogger, create_standard_methodology_logger

# UPDATED: Import tools system with error handling
try:
    from .tools import get_all_tools
except ImportError:
    from research_agent.tools import get_all_tools

# UPDATED: Import prompt system with error handling
try:
    from .prompts import (
        PLANNING_PROMPT_TEMPLATE,
        EXECUTION_PROMPT_TEMPLATE,
        REPLANNING_PROMPT_TEMPLATE,
        get_prompt_template
    )
except ImportError:
    from research_agent.prompts import (
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
        "status": "production-ready with Flask compatibility",  # UPDATED status
        "features": [
            "LangGraph plan-and-execute workflow",
            "Fast structured methodology logging (no LLM overhead)",
            "External TXT prompt templates",
            "Complete research context preservation",
            "Session-aware conversation management",
            "Graceful async stream completion",
            "Smart memory management without fact extraction bottleneck",
            "Production-ready architecture",
            "Flask app compatibility"  # ADDED feature
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

# UPDATED: Print info on import with compatibility status
def print_startup_info():
    """Print startup information."""
    info = get_version_info()
    print(f"🚀 Research Agent v{info['version']} - {info['status']}")
    print(f"⚡ Architecture: {info['architecture']}")
    print(f"🎯 Key features: Fast logging, Clean prompts, Flask compatible")
    print(f"🔗 Import mode: Hybrid (supports both relative and absolute imports)")

# UPDATED: Enable startup info to show compatibility status
# print_startup_info()