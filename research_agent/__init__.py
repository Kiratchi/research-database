"""
Simplified Research Publications Agent
A streamlined LangGraph-based plan-and-execute agent for querying research publications.
"""

__version__ = "2.0.0-simplified"

# Simple imports with single fallback strategy
def _safe_import(module_path, items):
    """Safely import items with fallback to None."""
    try:
        module = __import__(module_path, fromlist=items)
        return {item: getattr(module, item, None) for item in items}
    except ImportError:
        return {item: None for item in items}

# Import core components
core_imports = _safe_import('research_agent.core.workflow', ['ResearchAgent', 'create_workflow'])
ResearchAgent = core_imports['ResearchAgent']
create_workflow = core_imports['create_workflow']

# Import memory manager
memory_imports = _safe_import('research_agent.core.memory_manager', ['SimplifiedMemoryManager'])
SimplifiedMemoryManager = memory_imports['SimplifiedMemoryManager']

# Import agent manager  
agent_imports = _safe_import('research_agent.core.agent_manager', ['SimplifiedAgentManager'])
SimplifiedAgentManager = agent_imports['SimplifiedAgentManager']

# Import tools
tools_imports = _safe_import('research_agent.tools', ['get_all_tools'])
get_all_tools = tools_imports['get_all_tools']

# Import prompts directly
prompts_imports = _safe_import('research_agent.prompts', [
    'PLANNING_PROMPT_TEMPLATE',
    'EXECUTION_PROMPT_TEMPLATE', 
    'REPLANNING_PROMPT_TEMPLATE'
])
PLANNING_PROMPT_TEMPLATE = prompts_imports['PLANNING_PROMPT_TEMPLATE']
EXECUTION_PROMPT_TEMPLATE = prompts_imports['EXECUTION_PROMPT_TEMPLATE']
REPLANNING_PROMPT_TEMPLATE = prompts_imports['REPLANNING_PROMPT_TEMPLATE']

# Factory functions
def create_agent_manager(index_name: str = "research-publications-static"):
    """Create simplified agent manager."""
    if SimplifiedAgentManager is None:
        raise ImportError("SimplifiedAgentManager not available")
    return SimplifiedAgentManager(index_name=index_name)

def create_memory_manager():
    """Create simplified memory manager."""
    if SimplifiedMemoryManager is None:
        raise ImportError("SimplifiedMemoryManager not available")
    return SimplifiedMemoryManager()

def create_research_system(es_client=None, index_name="research-publications-static"):
    """Create a complete simplified research system."""
    return create_agent_manager(index_name=index_name)

# Export main components
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

# Simple startup message
print(f"Research Agent v{__version__} ready")