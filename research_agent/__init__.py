"""
Simplified Research Publications Agent
A streamlined LangGraph-based plan-and-execute agent for querying research publications
using Elasticsearch with natural language processing capabilities.

Simplified Features:
- Uses LangGraph's built-in session management
- Flexible model configuration system
- External prompt templates (preserved)
- Basic conversation memory
- Streamlined error handling
- Clean, maintainable codebase
"""

__version__ = "2.0.0-simplified"
__description__ = "Simplified Research Agent with LangGraph session management"

# Core components with error handling for import compatibility
try:
    # Try relative imports first (for internal package use)
    from .core.agent_manager import SimplifiedAgentManager
    from .core.memory_manager import SimplifiedMemoryManager  
    from .core.workflow import ResearchAgent, create_workflow
except ImportError:
    # Fall back to absolute imports (for Flask app compatibility)
    try:
        from research_agent.core.agent_manager import SimplifiedAgentManager
        from research_agent.core.memory_manager import SimplifiedMemoryManager
        from research_agent.core.workflow import ResearchAgent, create_workflow
    except ImportError:
        print("⚠️ Could not import simplified components - check file structure")
        SimplifiedAgentManager = None
        SimplifiedMemoryManager = None
        ResearchAgent = None
        create_workflow = None

# Import tools system with error handling
try:
    from .tools import get_all_tools
except ImportError:
    try:
        from research_agent.tools import get_all_tools
    except ImportError:
        print("⚠️ Could not import tools system")
        get_all_tools = None

# Import prompt system with error handling (preserved as requested)
try:
    from .prompts import (
        PLANNING_PROMPT_TEMPLATE,
        EXECUTION_PROMPT_TEMPLATE,
        REPLANNING_PROMPT_TEMPLATE
    )
except ImportError:
    try:
        from research_agent.prompts import (
            PLANNING_PROMPT_TEMPLATE,
            EXECUTION_PROMPT_TEMPLATE,
            REPLANNING_PROMPT_TEMPLATE
        )
    except ImportError:
        print("⚠️ Could not import prompt templates")
        PLANNING_PROMPT_TEMPLATE = None
        EXECUTION_PROMPT_TEMPLATE = None
        REPLANNING_PROMPT_TEMPLATE = None

# Helper functions
def create_agent_manager(index_name: str = "research-publications-static"):
    """Create simplified agent manager."""
    if SimplifiedAgentManager:
        return SimplifiedAgentManager(index_name=index_name)
    else:
        raise ImportError("SimplifiedAgentManager not available")

def create_memory_manager():
    """Create simplified memory manager."""
    if SimplifiedMemoryManager:
        return SimplifiedMemoryManager()
    else:
        raise ImportError("SimplifiedMemoryManager not available")

def create_research_system(es_client=None, index_name="research-publications-static"):
    """
    Create a complete simplified research system.
    
    Args:
        es_client: Elasticsearch client (optional)
        index_name: Elasticsearch index name
        
    Returns:
        Configured SimplifiedAgentManager ready for use
    """
    return create_agent_manager(index_name=index_name)

# Export main classes and functions
__all__ = [
    "SimplifiedAgentManager",
    "SimplifiedMemoryManager", 
    "ResearchAgent",
    "create_workflow",
    "create_agent_manager",
    "create_memory_manager",
    "create_research_system",
    "get_all_tools",
    "PLANNING_PROMPT_TEMPLATE",
    "EXECUTION_PROMPT_TEMPLATE", 
    "REPLANNING_PROMPT_TEMPLATE"
]

# Version info tuple for programmatic access
VERSION_INFO = tuple(map(int, __version__.split('.')[0].split('-')[0].split('.')))

def get_version_info():
    """Get detailed version information."""
    return {
        "version": __version__,
        "version_info": VERSION_INFO,
        "status": "simplified-production-ready",
        "features": [
            "Simplified LangGraph plan-and-execute workflow",
            "LangGraph built-in session management", 
            "Flexible model configuration system",
            "External prompt templates (preserved)",
            "Basic conversation memory",
            "Streamlined error handling",
            "Clean, maintainable codebase",
            "Flask app compatibility"
        ],
        "removed_complexity": [
            "Custom session caching (uses LangGraph's built-in)",
            "Methodology logger (use LangSmith instead)",
            "Complex research storage (simplified memory)",
            "Extensive health checks (basic status only)",
            "Verbose error handling (clean and simple)"
        ],
        "architecture": "simplified_langgraph"
    }

def print_startup_info():
    """Print minimal startup information."""
    info = get_version_info()
    print(f"Research Agent v{info['version']} initialized")

# Show minimal startup info on import
print_startup_info()