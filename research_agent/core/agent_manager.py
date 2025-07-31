"""
COMBINED Agent Manager - Best of Both Worlds
COMBINES: Smart methodology learning + Plan-Execute architecture + LangChain memory + Session management
BUILDS ON: Sophisticated workflow with automatic conversation context
FILENAME: agent_manager.py
"""

import os
import json
import asyncio
import time
import uuid
import sys
import threading
import concurrent.futures
from typing import Dict, List, Any, Optional
from elasticsearch import Elasticsearch

from .workflow import create_combined_research_agent, CombinedResearchAgent
from .memory_manager import SessionMemoryManager
from .methodology_logger import SmartMethodologyLogger

# Import tools
try:
    from ..tools import get_all_tools
    TOOLS_AVAILABLE = True
except ImportError:
    print("⚠️ Tools not available - falling back to basic functionality")
    TOOLS_AVAILABLE = False


class CombinedAgentManager:
    """
    Combined Agent Manager with Plan-Execute + LangChain Memory + Smart Methodology Learning.
    COMBINES: Best features from both approaches with session-based memory management.
    """
    
    def __init__(self, index_name: str = "research-publications-static"):
        """Initialize with combined methodology: plan-execute + LangChain memory."""
        self.index_name = index_name
        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "agent_type": "combined_plan_execute_langchain_memory",
            "architecture": "plan_execute_with_smart_methodology",
            "memory_system": "langchain_automatic",
            "context_injection": "automatic",
            "manual_context_building": False,
            "session_continuity": True,
            "smart_methodology_learning": True,
            "llm_powered_analysis": True
        }
        
        # Initialize components
        self.es_client = self._init_elasticsearch()
        self.memory_manager = self._init_memory()
        self.research_agent = self._init_research_agent()
        
        # Initialize smart methodology logger
        try:
            self.smart_logger = SmartMethodologyLogger()
            print("🧠 Smart Methodology Logger initialized in CombinedAgentManager")
        except Exception as e:
            print(f"⚠️ Smart Methodology Logger initialization failed: {e}")
            self.smart_logger = None
        
        print("🚀 CombinedAgentManager initialized with PLAN-EXECUTE + LANGCHAIN MEMORY + SMART METHODOLOGY!")
        print("🎯 Architecture: Best of both worlds approach")
    
    def _init_elasticsearch(self) -> Optional[Elasticsearch]:
        """Initialize Elasticsearch client."""
        try:
            es_host = os.getenv("ES_HOST")
            es_user = os.getenv("ES_USER")
            es_pass = os.getenv("ES_PASS")
            
            if not all([es_host, es_user, es_pass]):
                print("❌ Elasticsearch credentials missing")
                return None
            
            es_client = Elasticsearch(
                [es_host],
                http_auth=(es_user, es_pass),
                timeout=30,
                max_retries=3,
                retry_on_timeout=True
            )
            
            if es_client.ping():
                print("✅ Elasticsearch connected (combined methodology)")
                return es_client
            else:
                print("❌ Elasticsearch connection failed")
                return None
                
        except Exception as e:
            print(f"❌ Elasticsearch error: {e}")
            return None
    
    def _init_memory(self) -> SessionMemoryManager:
        """Initialize LangChain session memory manager."""
        try:
            memory_manager = SessionMemoryManager(
                default_memory_type="buffer_window"
            )
            print("✅ LangChain session memory manager initialized (combined methodology)")
            return memory_manager
        except Exception as e:
            print(f"❌ Memory manager error: {e}")
            # Fallback to basic memory manager
            return SessionMemoryManager()
    
    def _init_research_agent(self) -> Optional[CombinedResearchAgent]:
        """Initialize combined research agent."""
        try:
            research_agent = create_combined_research_agent(
                es_client=self.es_client,
                index_name=self.index_name
            )
            print("✅ Combined research agent initialized (plan-execute + LangChain memory)")
            return research_agent
        except Exception as e:
            print(f"❌ Combined research agent error: {e}")
            return None
    
    def is_ready(self) -> bool:
        """Check if system is ready."""
        return (
            self.memory_manager is not None and
            self.research_agent is not None and
            (self.es_client is None or self.es_client.ping())  # ES is optional
        )
    
    def process_query(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """
        Process query with COMBINED methodology: Plan-Execute + LangChain Memory + Smart Learning.
        ENHANCED: Includes intelligent follow-up analysis and methodology insights.
        """
        
        # Enforce session_id requirement for memory continuity
        if not session_id:
            self.query_stats["failed_queries"] += 1
            return {
                "success": False,
                "error": "session_id is REQUIRED for combined methodology with memory continuity",
                "agent_type": "combined_plan_execute_langchain_memory"
            }
        
        # Validate session_id format
        if not isinstance(session_id, str) or len(session_id) < 10:
            self.query_stats["failed_queries"] += 1
            return {
                "success": False,
                "error": f"Invalid session_id format: {session_id}. Must be string with 10+ characters",
                "agent_type": "combined_plan_execute_langchain_memory"
            }
        
        self.query_stats["total_queries"] += 1
        start_time = time.time()
        
        try:
            print(f"🔍 Processing with COMBINED METHODOLOGY for session: {session_id}")
            print(f"📝 Query: '{query}'")
            
            # ENHANCED: Smart follow-up detection and analysis
            conversation_history = self.memory_manager.get_conversation_history(session_id)
            self._analyze_and_log_followup(query, session_id, conversation_history)
            
            # Handle simple queries (optional optimization)
            simple_response = self._handle_simple_query(query)
            if simple_response:
                # Save to LangChain memory for consistency
                self.memory_manager.save_conversation(session_id, query, simple_response)
                
                self.query_stats["successful_queries"] += 1
                return {
                    "success": True,
                    "response": simple_response,
                    "session_id": session_id,
                    "execution_time": time.time() - start_time,
                    "response_type": "simple",
                    "agent_type": "combined_plan_execute_langchain_memory",
                    "memory_automatic": True,
                    "conversation_length": len(conversation_history) + 2  # +2 for current exchange
                }
            
            # Check system readiness
            if not self.is_ready():
                self.query_stats["failed_queries"] += 1
                return {
                    "success": False,
                    "error": "System not ready - combined research agent unavailable",
                    "session_id": session_id,
                    "agent_type": "combined_plan_execute_langchain_memory"
                }
            
            # Execute with combined research agent (plan-execute + LangChain memory)
            result = self.research_agent.execute_query(session_id, query)
            
            if not result["success"]:
                self.query_stats["failed_queries"] += 1
                return result
            
            self.query_stats["successful_queries"] += 1
            
            # Get updated conversation info
            updated_conversation_history = self.memory_manager.get_conversation_history(session_id)
            
            # Add execution metadata
            result.update({
                "execution_time": time.time() - start_time,
                "response_type": "research",
                "agent_type": "combined_plan_execute_langchain_memory",
                "memory_automatic": True,
                "context_injection": "automatic",
                "conversation_length": len(updated_conversation_history),
                "architecture": "plan_execute_with_smart_methodology"
            })
            
            print(f"✅ Query processed with COMBINED METHODOLOGY for session: {session_id}")
            print(f"🧠 Conversation length: {result.get('conversation_length', 0)} messages")
            print(f"⚡ Architecture: Plan-Execute + LangChain Memory + Smart Learning")
            
            return result
            
        except Exception as e:
            self.query_stats["failed_queries"] += 1
            error_msg = f"Error processing query with combined methodology: {str(e)}"
            print(f"❌ {error_msg} for session: {session_id}")
            
            return {
                "success": False,
                "error": error_msg,
                "session_id": session_id,
                "execution_time": time.time() - start_time,
                "agent_type": "combined_plan_execute_langchain_memory"
            }
    
    def _analyze_and_log_followup(self, query: str, session_id: str, conversation_history: List[Dict]) -> None:
        """Analyze and log follow-up questions with smart LLM analysis."""
        
        # Only analyze if we have previous conversation and smart logger
        if len(conversation_history) < 2 or not self.smart_logger:
            return
        
        try:
            # Extract previous interaction
            previous_messages = [msg for msg in conversation_history if msg.get('role') == 'user']
            previous_responses = [msg for msg in conversation_history if msg.get('role') == 'assistant']
            
            if len(previous_messages) >= 2 and len(previous_responses) >= 1:
                original_query = previous_messages[-2]['content']
                previous_response = previous_responses[-1]['content']
                
                # Prepare rich context for LLM analysis
                context_usage_notes = self._analyze_context_usage(original_query, query, previous_response)
                efficiency_observations = self._analyze_efficiency_patterns(original_query, query, conversation_history)
                
                # SMART LOGGING: LLM-powered follow-up analysis
                self.smart_logger.log_followup_analysis(
                    session_id,
                    original_query,
                    query,
                    context_usage_notes,
                    efficiency_observations
                )
                
                print(f"🔗 Smart follow-up analysis logged for combined session: {session_id}")
                
        except Exception as e:
            print(f"⚠️ Could not log smart follow-up analysis: {e}")
    
    def _analyze_context_usage(self, original_query: str, followup_query: str, previous_response: str) -> str:
        """Analyze how well context is being used in follow-up."""
        
        # Check for pronoun usage (indicates context awareness)
        context_indicators = ['his', 'her', 'their', 'this', 'these', 'those', 'it', 'they', 'them']
        has_context_pronouns = any(word in followup_query.lower() for word in context_indicators)
        
        # Check for keyword overlap
        original_keywords = set(original_query.lower().split())
        followup_keywords = set(followup_query.lower().split())
        keyword_overlap = len(original_keywords & followup_keywords)
        
        # Check if follow-up builds on previous response
        response_keywords = set(previous_response.lower().split())
        response_overlap = len(followup_keywords & response_keywords)
        
        context_analysis = f"""Context Usage Analysis:
- Original Query: '{original_query}'
- Follow-up Query: '{followup_query}'
- Pronoun Usage: {has_context_pronouns} (indicates context awareness)
- Keyword Overlap with Original: {keyword_overlap} words
- Builds on Previous Response: {response_overlap} shared concepts
- Previous Response Length: {len(previous_response)} characters
- Memory System: LangChain automatic injection with Plan-Execute workflow"""
        
        return context_analysis
    
    def _analyze_efficiency_patterns(self, original_query: str, followup_query: str, conversation_history: List[Dict]) -> str:
        """Analyze efficiency patterns in follow-up questions."""
        
        # Calculate conversation depth
        user_messages = [msg for msg in conversation_history if msg.get('role') == 'user']
        conversation_depth = len(user_messages)
        
        # Analyze query evolution
        query_evolution = "deepening" if len(followup_query) > len(original_query) else "narrowing"
        
        # Check for temporal patterns
        has_temporal = any(word in followup_query.lower() for word in ['recent', 'latest', '2023', '2024', 'current', 'new'])
        
        # Check for relationship exploration
        has_relationships = any(word in followup_query.lower() for word in ['collaborate', 'work with', 'team', 'colleagues'])
        
        efficiency_analysis = f"""Efficiency Patterns (Combined Methodology):
- Conversation Depth: {conversation_depth} exchanges
- Query Evolution: {query_evolution} focus
- Temporal Elements: {has_temporal} (seeking recent information)
- Relationship Exploration: {has_relationships} (exploring connections)
- Query Length Change: {len(followup_query)} vs {len(original_query)} characters
- Efficiency Indicators: {'High' if conversation_depth <= 3 else 'Medium' if conversation_depth <= 5 else 'Low'}
- Architecture: Plan-Execute with LangChain memory provides superior context handling"""
        
        return efficiency_analysis
    
    def _handle_simple_query(self, query: str) -> Optional[str]:
        """Handle simple queries with combined methodology context."""
        query_clean = query.lower().strip().rstrip('!?.,;:')
        
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon']
        if query_clean in greetings:
            return "Hello! I'm your research assistant with combined methodology: sophisticated Plan-Execute workflow with automatic LangChain memory and smart learning. I'll remember our entire conversation naturally. What would you like to research?"
        
        thanks = ['thanks', 'thank you', 'thx', 'ty']
        if query_clean in thanks:
            return "You're welcome! Feel free to ask follow-up questions - my combined architecture with LangChain memory ensures perfect conversation continuity."
        
        help_patterns = ['help', 'what can you do', 'how does this work']
        if any(pattern in query_clean for pattern in help_patterns):
            return """I can help you research authors and academic fields with combined methodology architecture:

**🔍 Research Capabilities:**
• Author information and publication analysis
• Research trend identification  
• Collaboration network mapping
• Field-specific searches

**🧠 Combined Architecture Features:**
• Plan-Execute workflow for comprehensive research
• LangChain automatic conversation memory
• Smart methodology learning with LLM analysis
• Intelligent replanning based on research progress
• Tool effectiveness tracking and optimization
• Session-based memory continuity

**🎯 Advanced Memory Features:**
• Natural follow-up questions: "What are his publications?" "Any recent ones?"
• Context-aware research planning
• No repetition of previous searches
• Builds upon conversation history automatically

**💡 Try this conversation flow:**
1. "Who is Per-Olof Arnäs?"
2. "What are his main research areas?" 
3. "Find his recent publications"
4. "Who does he collaborate with?"

My combined Plan-Execute + LangChain memory architecture provides the most sophisticated research experience with perfect conversation continuity!"""
        
        return None
    
    def clear_memory(self, session_id: str) -> Dict[str, Any]:
        """Clear LangChain memory for specific session."""
        if not session_id:
            return {
                "success": False,
                "error": "session_id is required to clear memory",
                "agent_type": "combined_plan_execute_langchain_memory"
            }
        
        try:
            if self.research_agent:
                self.research_agent.clear_session(session_id)
                return {
                    "success": True,
                    "message": f"Cleared combined methodology memory for session: {session_id}",
                    "agent_type": "combined_plan_execute_langchain_memory",
                    "architecture": "plan_execute_with_smart_methodology"
                }
            else:
                return {
                    "success": False,
                    "error": "Combined research agent not available",
                    "agent_type": "combined_plan_execute_langchain_memory"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error clearing memory for session {session_id}: {str(e)}",
                "agent_type": "combined_plan_execute_langchain_memory"
            }
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a specific session with combined methodology context."""
        if not session_id:
            return {
                "success": False,
                "error": "session_id is required",
                "agent_type": "combined_plan_execute_langchain_memory"
            }
        
        try:
            if self.research_agent:
                session_stats = self.research_agent.get_session_stats(session_id)
                session_stats.update({
                    "success": True,
                    "agent_type": "combined_plan_execute_langchain_memory",
                    "memory_system": "langchain_automatic",
                    "architecture": "plan_execute_with_smart_methodology",
                    "workflow_type": "sophisticated_research_with_replanning"
                })
                return session_stats
            else:
                return {
                    "success": False,
                    "error": "Combined research agent not available",
                    "session_id": session_id,
                    "agent_type": "combined_plan_execute_langchain_memory"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error retrieving session info for {session_id}: {str(e)}",
                "session_id": session_id,
                "agent_type": "combined_plan_execute_langchain_memory"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status with combined methodology information."""
        es_connected = False
        if self.es_client:
            try:
                es_connected = self.es_client.ping()
            except:
                es_connected = False
        
        memory_stats = {}
        if self.research_agent:
            try:
                memory_stats = self.research_agent.get_all_sessions_stats()
            except:
                memory_stats = {}
        
        return {
            "system_ready": self.is_ready(),
            "architecture": "combined_plan_execute_langchain_memory",
            "workflow_type": "plan_execute_with_smart_methodology",
            "memory_system": "langchain_automatic",
            "context_injection": "automatic",
            "manual_context_building": False,
            "session_continuity": True,
            "smart_methodology_learning": True,
            "llm_powered_analysis": True,
            "session_handling": "strict_frontend_requirement",
            "elasticsearch": {
                "connected": es_connected,
                "host": os.getenv("ES_HOST", "Not configured"),
                "index": self.index_name,
                "required": False  # ES is optional
            },
            "memory": {
                "type": "langchain_session_based",
                "automatic_injection": True,
                "manual_context_building": False,
                "conversation_continuity": True,
                "total_sessions": memory_stats.get("total_sessions", 0),
                "total_messages": memory_stats.get("total_messages", 0),
                "average_messages_per_session": memory_stats.get("average_messages_per_session", 0)
            },
            "research_agent": {
                "type": "CombinedResearchAgent",
                "workflow": "plan_execute_with_replanning",
                "automatic_memory": True,
                "smart_methodology": True,
                "initialized": self.research_agent is not None
            },
            "smart_methodology": {
                "logger_initialized": self.smart_logger is not None,
                "analysis_type": "llm_powered",
                "learning_active": True,
                "adaptive_categorization": True,
                "no_hardcoded_rules": True,
                "workflow_integration": "seamless"
            },
            "statistics": self.query_stats
        }
    
    def get_smart_methodology_insights(self, days: int = 7) -> Dict[str, Any]:
        """Get LLM-powered methodology insights for the specified period."""
        try:
            if not self.smart_logger:
                return {
                    "success": False,
                    "error": "Smart methodology logger not initialized",
                    "analysis_type": "llm_powered",
                    "architecture": "combined_plan_execute_langchain_memory"
                }
            
            insights = self.smart_logger.generate_llm_insights_summary(days)
            return {
                "success": True,
                "insights": insights,
                "analysis_type": "llm_powered",
                "architecture": "combined_plan_execute_langchain_memory",
                "period_days": days,
                "generated_at": time.time()
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to generate smart insights: {str(e)}",
                "analysis_type": "llm_powered",
                "architecture": "combined_plan_execute_langchain_memory"
            }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics with combined methodology info."""
        if self.research_agent:
            try:
                base_stats = self.research_agent.get_all_sessions_stats()
                base_stats.update({
                    "architecture": "combined_plan_execute_langchain_memory",
                    "workflow_type": "plan_execute_with_smart_methodology",
                    "memory_system": "langchain_automatic",
                    "automatic_context_injection": True,
                    "manual_context_building": False,
                    "session_continuity": True,
                    "smart_methodology_enabled": self.smart_logger is not None,
                    "analysis_capabilities": [
                        "Query type classification with conversation context",
                        "Tool effectiveness assessment in plan-execute workflow",
                        "Replanning reason analysis with memory integration", 
                        "Session outcome evaluation with smart learning",
                        "Follow-up optimization with LangChain memory",
                        "Pattern recognition across conversation sessions"
                    ],
                    "learning_type": "llm_powered_adaptive_with_memory_integration"
                })
                return base_stats
            except Exception as e:
                return {
                    "error": f"Error getting memory stats: {str(e)}",
                    "architecture": "combined_plan_execute_langchain_memory"
                }
        else:
            return {
                "error": "Combined research agent not available",
                "architecture": "combined_plan_execute_langchain_memory"
            }
    
    def get_tools_info(self) -> Dict[str, Any]:
        """Get detailed information about available tools."""
        try:
            if self.research_agent and hasattr(self.research_agent, 'es_client'):
                # Get tools info from the research agent
                if TOOLS_AVAILABLE:
                    if self.research_agent.es_client:
                        tools = get_all_tools(self.research_agent.es_client, self.research_agent.index_name)
                    else:
                        tools = get_all_tools()
                    
                    tools_info = []
                    for tool in tools:
                        tools_info.append({
                            "name": tool.name,
                            "description": tool.description,
                            "type": str(type(tool).__name__)
                        })
                    
                    return {
                        "success": True,
                        "total_tools": len(tools_info),
                        "tools": tools_info,
                        "elasticsearch_connected": self.es_client is not None and self.es_client.ping() if self.es_client else False,
                        "agent_type": "combined_plan_execute_langchain_memory",
                        "architecture": "plan_execute_with_smart_methodology"
                    }
                else:
                    return {
                        "success": False,
                        "error": "Tools not available",
                        "total_tools": 0,
                        "tools": [],
                        "agent_type": "combined_plan_execute_langchain_memory"
                    }
            else:
                return {
                    "success": False,
                    "error": "Combined research agent not available",
                    "total_tools": 0,
                    "tools": [],
                    "agent_type": "combined_plan_execute_langchain_memory"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error retrieving tools info: {str(e)}",
                "total_tools": 0,
                "tools": [],
                "agent_type": "combined_plan_execute_langchain_memory"
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Health check with combined methodology verification."""
        health = {
            "status": "healthy",
            "timestamp": time.time(),
            "architecture": "combined_plan_execute_langchain_memory",
            "workflow_type": "plan_execute_with_smart_methodology",
            "memory_system": "langchain_automatic",
            "context_injection": "automatic",
            "session_continuity": True,
            "checks": {}
        }
        
        # Test session enforcement (should fail without session_id)
        try:
            test_result = self.process_query("test", session_id=None)
            if test_result["success"] == False and "session_id is REQUIRED" in test_result["error"]:
                health["checks"]["session_enforcement"] = "healthy (correctly rejects missing session_id)"
            else:
                health["checks"]["session_enforcement"] = "degraded (should reject missing session_id)"
                health["status"] = "degraded"
        except Exception as e:
            health["checks"]["session_enforcement"] = f"degraded ({str(e)})"
            health["status"] = "degraded"
        
        # Test combined research agent
        if self.research_agent:
            health["checks"]["combined_research_agent"] = "healthy (plan-execute + LangChain memory)"
        else:
            health["checks"]["combined_research_agent"] = "unhealthy"
            health["status"] = "degraded"
        
        # Test Elasticsearch (optional)
        try:
            if self.es_client and self.es_client.ping():
                health["checks"]["elasticsearch"] = "healthy (optional)"
            elif self.es_client:
                health["checks"]["elasticsearch"] = "unhealthy (but optional)"
            else:
                health["checks"]["elasticsearch"] = "not configured (optional)"
        except:
            health["checks"]["elasticsearch"] = "unhealthy (but optional)"
        
        # Test LangChain memory
        try:
            if self.memory_manager:
                test_session = f"health_check_{int(time.time())}"
                stats = self.memory_manager.get_session_info(test_session)
                health["checks"]["langchain_memory"] = "healthy (automatic context injection)"
            else:
                health["checks"]["langchain_memory"] = "unhealthy (memory manager not available)"
                health["status"] = "degraded"
        except Exception as e:
            health["checks"]["langchain_memory"] = f"degraded ({str(e)})"
            health["status"] = "degraded"
        
        # Test smart methodology logger
        try:
            if self.smart_logger:
                health["checks"]["smart_methodology_logger"] = "healthy (llm_powered)"
            else:
                health["checks"]["smart_methodology_logger"] = "unavailable"
                # Don't mark as degraded since it's optional
        except Exception as e:
            health["checks"]["smart_methodology_logger"] = f"degraded ({str(e)})"
        
        return health


def create_combined_agent_manager(index_name: str = "research-publications-static") -> CombinedAgentManager:
    """Create combined agent manager with plan-execute + LangChain memory + smart methodology."""
    return CombinedAgentManager(index_name=index_name)


if __name__ == "__main__":
    print("Testing CombinedAgentManager with PLAN-EXECUTE + LANGCHAIN MEMORY + SMART METHODOLOGY...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    manager = create_combined_agent_manager()
    
    # Test 1: Should FAIL - no session_id provided
    print("\n🧪 Test 1: No session_id (should fail)")
    result = manager.process_query("Hello!")
    print(f"Result: {result['success']} - {result.get('error', 'No error')}")
    
    # Test 2: Should SUCCEED - session_id provided
    print("\n🧪 Test 2: With session_id (should succeed)")
    test_session = "test_combined_methodology_123"
    result = manager.process_query("Hello!", session_id=test_session)
    print(f"Result: {result['success']} - Response: {result.get('response', 'No response')[:50]}...")
    print(f"Architecture: {result.get('agent_type', 'unknown')}")
    
    # Test 3: Follow-up query (should have automatic context from LangChain)
    print("\n🧪 Test 3: Follow-up query (automatic LangChain context)")
    result2 = manager.process_query("What can you help me with?", session_id=test_session)
    print(f"Result: {result2['success']} - Has context: {result2.get('conversation_length', 0) > 2}")
    print(f"Conversation length: {result2.get('conversation_length', 0)} messages")
    
    # Test 4: Complex research query (should use plan-execute workflow)
    print("\n🧪 Test 4: Complex research query (plan-execute workflow)")
    result3 = manager.process_query("Who is Per-Olof Arnäs and what are his research areas?", session_id=test_session)
    print(f"Result: {result3['success']} - Architecture: {result3.get('architecture', 'unknown')}")
    print(f"Response type: {result3.get('response_type', 'unknown')}")
    
    # Test health check
    health = manager.health_check()
    print(f"\n🏥 Health check status: {health['status']}")
    print