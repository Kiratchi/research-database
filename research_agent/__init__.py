"""
Research Publications Agent
A LangChain-based research agent for querying research publications
using Elasticsearch with natural language processing capabilities.
Now using COMBINED architecture: Plan-Execute workflow with LangChain memory integration.
"""

__version__ = "0.5.0"  # Combined architecture release
__author__ = "Research Agent Team"

# COMBINED ARCHITECTURE IMPORTS - Best of Both Worlds
from .core.workflow import CombinedResearchAgent, create_combined_research_agent
from .core.agent_manager import CombinedAgentManager, create_combined_agent_manager
from .core.memory_manager import SessionMemoryManager

# SMART METHODOLOGY LEARNING
try:
    from .core.methodology_logger import SmartMethodologyLogger
    METHODOLOGY_LOGGER_AVAILABLE = True
except ImportError:
    print("⚠️ SmartMethodologyLogger not available - smart learning features disabled")
    METHODOLOGY_LOGGER_AVAILABLE = False

# TOOLS INTEGRATION
try:
    from .tools import get_all_tools
    TOOLS_AVAILABLE = True
except ImportError:
    print("⚠️ Tools not available - basic functionality only")
    TOOLS_AVAILABLE = False
    def get_all_tools(*args, **kwargs):
        return []

# COMBINED STATE MODELS - For Plan-Execute + Memory Architecture
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict

class PlanExecuteState(TypedDict):
    """State for combined architecture with LangChain memory integration."""
    input: str
    plan: List[str]
    past_steps: List[tuple[str, str]]
    response: Optional[str]
    session_id: str  # Required for memory continuity
    chat_history: Optional[List[Any]]  # LangChain injects this automatically

# Model classes for combined architecture
class Plan:
    """Plan model for plan-execute workflow."""
    def __init__(self, steps: List[str]):
        self.steps = steps
    
    def __repr__(self):
        return f"Plan(steps={len(self.steps)})"

class Response:
    """Response model with metadata."""
    def __init__(self, response: str, metadata: Dict[str, Any] = None):
        self.response = response
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"Response(length={len(self.response)})"

class Act:
    """Action model for replanning decisions."""
    def __init__(self, action_type: str, response: str = None, steps: List[str] = None):
        self.action_type = action_type
        self.response = response
        self.steps = steps
    
    def __repr__(self):
        return f"Act(type={self.action_type})"

# CONFIGURATION HELPERS
def get_version_info() -> Dict[str, Any]:
    """Get detailed version and architecture information."""
    return {
        "version": __version__,
        "architecture": "combined_plan_execute_langchain_memory",
        "workflow_type": "plan_execute_with_smart_methodology",
        "memory_system": "langchain_automatic",
        "context_injection": "automatic",
        "session_continuity": True,
        "smart_methodology_learning": METHODOLOGY_LOGGER_AVAILABLE,
        "tools_available": TOOLS_AVAILABLE
    }

def create_default_research_system(index_name: str = "research-publications-static") -> CombinedAgentManager:
    """
    Create a complete research system with combined architecture.
    
    Returns:
        CombinedAgentManager: Ready-to-use research system with:
        - Plan-Execute workflow for comprehensive research
        - LangChain automatic memory injection
        - Smart methodology learning (if available)
        - Session-based memory continuity
    """
    return create_combined_agent_manager(index_name=index_name)

# MAIN EXPORTS
__all__ = [
    # COMBINED ARCHITECTURE COMPONENTS
    "CombinedResearchAgent",
    "create_combined_research_agent", 
    "CombinedAgentManager",
    "create_combined_agent_manager",
    "SessionMemoryManager",
    
    # STATE MODELS
    "PlanExecuteState",
    "Plan", 
    "Response",
    "Act",
    
    # TOOLS
    "get_all_tools",
    
    # HELPERS
    "get_version_info",
    "create_default_research_system",
]

# CONDITIONAL EXPORTS
if METHODOLOGY_LOGGER_AVAILABLE:
    __all__.append("SmartMethodologyLogger")

# INITIALIZATION MESSAGE
def _print_startup_info():
    """Print startup information about the combined architecture."""
    print("🚀 Research Agent v0.5.0 - COMBINED ARCHITECTURE")
    print("🎯 Plan-Execute workflow + LangChain memory + Smart methodology learning")
    print("🧠 Automatic conversation context injection via {chat_history}")
    print("⚡ Features:")
    print("  - Sophisticated research planning and execution")
    print("  - Intelligent replanning based on research progress")
    print("  - LangChain memory for perfect conversation continuity")
    print("  - Smart methodology learning with LLM analysis")
    print("  - Session-based memory management")
    print("  - Tool effectiveness tracking and optimization")
    
    if not TOOLS_AVAILABLE:
        print("⚠️ Tools not available - import research_agent.tools for full functionality")
    
    if not METHODOLOGY_LOGGER_AVAILABLE:
        print("⚠️ Smart methodology learning not available - basic functionality only")

# Print startup info
_print_startup_info()

# USAGE EXAMPLES (as module docstring)
"""
USAGE EXAMPLES:

# Simple usage - create complete research system
from research_agent import create_default_research_system

manager = create_default_research_system()
result = manager.process_query("Who is Per-Olof Arnäs?", session_id="my_session")

# Advanced usage - custom configuration  
from research_agent import CombinedAgentManager

manager = CombinedAgentManager(index_name="my-publications")
result = manager.process_query("Research query", session_id="session_123")

# Follow-up queries use automatic memory
followup = manager.process_query("What are his research areas?", session_id="session_123")

# Memory and session management
session_info = manager.get_session_info("session_123")
manager.clear_memory("session_123")

# Smart methodology insights (if available)
insights = manager.get_smart_methodology_insights(days=7)

# System status and health
status = manager.get_status()
health = manager.health_check()
"""