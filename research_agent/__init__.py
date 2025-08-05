"""
Simplified Research Publications Agent - ReAct Architecture
A streamlined LangGraph-based ReAct agent for querying research publications.
"""

__version__ = "2.0.0-react"

# Simple imports with single fallback strategy
def safe_import(module_path, items):
    """Safely import items with fallback to None."""
    try:
        module = __import__(module_path, fromlist=items)
        return {item: getattr(module, item, None) for item in items}
    except ImportError:
        return {item: None for item in items}

# Import core components
core_imports = safe_import('research_agent.core.workflow', ['ResearchAgent', 'create_react_workflow'])
ResearchAgent = core_imports['ResearchAgent']
create_react_workflow = core_imports['create_react_workflow']

# Import memory manager (the working one)
memory_imports = safe_import('research_agent.core.memory_singleton', ['get_global_memory_manager', 'GlobalMemoryManager'])
get_global_memory_manager = memory_imports['get_global_memory_manager']
GlobalMemoryManager = memory_imports['GlobalMemoryManager']

# Import agent manager  
agent_imports = safe_import('research_agent.core.agent_manager', ['AgentManager', 'create_agent_manager'])
AgentManager = agent_imports['AgentManager']
create_agent_manager = agent_imports['create_agent_manager']

# Import tools
tools_imports = safe_import('research_agent.tools', ['get_all_tools'])
get_all_tools = tools_imports['get_all_tools']

# Import ReAct prompt (no more plan-execute prompts)
prompts_imports = safe_import('research_agent.core.workflow', ['REACT_PROMPT_TEMPLATE'])
REACT_PROMPT_TEMPLATE = prompts_imports['REACT_PROMPT_TEMPLATE']

# Factory functions
def create_memory_manager():
    """Create global memory manager."""
    if get_global_memory_manager is None:
        raise ImportError("GlobalMemoryManager not available")
    return get_global_memory_manager()

def create_research_system(es_client=None, index_name="research-publications-static"):
    """Create a complete ReAct research system."""
    if create_agent_manager is None:
        raise ImportError("AgentManager not available")
    return create_agent_manager(index_name=index_name)

# Export main components
__all__ = [
    "AgentManager",
    "GlobalMemoryManager", 
    "ResearchAgent",
    "create_react_workflow",
    "create_agent_manager",
    "create_memory_manager",
    "get_global_memory_manager",
    "create_research_system",
    "get_all_tools",
    "REACT_PROMPT_TEMPLATE"
]

# Simple startup message
print(f"Research Agent v{__version__} ready (ReAct)")