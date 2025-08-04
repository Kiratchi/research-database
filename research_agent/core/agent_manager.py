"""
Balanced Simplified Agent Manager - Removes bloat but keeps safety features
Kept: Timeout protection, thread isolation, graceful cleanup
Removed: Verbose logging, complex status, extensive error handling
FIXED: Memory manager persistence issue
"""

import os
import asyncio
import time
import uuid
import concurrent.futures
from typing import Dict, Any, Optional
from elasticsearch import Elasticsearch

from .memory_singleton import get_global_memory_manager
from .workflow import ResearchAgent

# Import tools
try:
    from ..tools import get_all_tools
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False


class AgentManager:
    """Simplified agent manager - removes bloat but keeps safety features."""
    
    def __init__(self, index_name: str = "research-publications-static"):
        self.index_name = index_name
        self.query_stats = {"total": 0, "success": 0, "failed": 0}
        
        self.es_client = self._init_elasticsearch()
        
        # ✅ FIXED: Create ONE memory manager instance that persists
        self.memory_manager = self.memory_manager = get_global_memory_manager()
        
        print("AgentManager initialized with global memory")
    
    def _init_elasticsearch(self) -> Optional[Elasticsearch]:
        """Initialize Elasticsearch client."""
        try:
            es_host = os.getenv("ES_HOST")
            es_user = os.getenv("ES_USER") 
            es_pass = os.getenv("ES_PASS")
            
            if not all([es_host, es_user, es_pass]):
                return None
            
            client = Elasticsearch([es_host], http_auth=(es_user, es_pass), timeout=30)
            return client if client.ping() else None
            
        except Exception:
            return None
    
    def is_ready(self) -> bool:
        """Check if system is ready."""
        return self.es_client is not None and self.es_client.ping()
    
    def process_query(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Process query with safety features preserved."""
        
        session_id = session_id or f'session_{int(time.time())}_{str(uuid.uuid4())[:8]}'
        self.query_stats["total"] += 1
        start_time = time.time()
        
        try:
            print(f"Processing query for session: {session_id}")
            
            # Handle simple queries
            simple_response = self._handle_simple_query(query)
            if simple_response:
                # ✅ FIXED: Use the persistent memory manager
                self.memory_manager.save_conversation(session_id, query, simple_response)
                self.query_stats["success"] += 1
                return {
                    "success": True,
                    "response": simple_response,
                    "session_id": session_id,
                    "execution_time": time.time() - start_time,
                    "response_type": "simple"
                }
            
            # Check system readiness
            if not self.is_ready():
                self.query_stats["failed"] += 1
                return {
                    "success": False,
                    "error": "System not ready - Elasticsearch required",
                    "session_id": session_id
                }
            
            # Execute research workflow with safety features
            # ✅ FIXED: Use the persistent memory manager
            conversation_history = self.memory_manager.get_conversation_history_for_state(session_id)
            response_content = self._execute_research_safely(query, conversation_history, session_id)
            
            if response_content:
                # ✅ FIXED: Use the persistent memory manager
                self.memory_manager.save_conversation(session_id, query, response_content)
                self.query_stats["success"] += 1
                print(f"Research completed in {time.time() - start_time:.1f}s")
                return {
                    "success": True,
                    "response": response_content,
                    "session_id": session_id,
                    "execution_time": time.time() - start_time,
                    "response_type": "research"
                }
            else:
                self.query_stats["failed"] += 1
                return {
                    "success": False,
                    "error": "No response generated",
                    "session_id": session_id
                }
            
        except Exception as e:
            self.query_stats["failed"] += 1
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    def _handle_simple_query(self, query: str) -> Optional[str]:
        """Handle simple queries."""
        query_lower = query.lower().strip()
        
        if query_lower in ["hello", "hi", "hey"]:
            return "Hello! I'm your research assistant. What would you like to research?"
        
        if "help" in query_lower or "what can you do" in query_lower:
            return """I can help you research authors and academic fields:
• Author information and publication analysis
• Research trend identification  
• Collaboration network mapping
• Field-specific searches

Just ask me about any researcher or academic field!"""
        
        return None

    def _execute_research_safely(self, query: str, conversation_history, session_id: str) -> str:
        """Execute research workflow with timeout and thread isolation."""
        
        def run_workflow():
            """Run workflow in isolated thread with proper cleanup."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # ✅ FIXED: Create and compile agent with the persistent memory manager
                agent = ResearchAgent(
                    es_client=self.es_client,
                    index_name=self.index_name,
                    recursion_limit=50,
                    memory_manager=self.memory_manager  # ✅ Pass the persistent memory manager
                )
                agent._compile_agent(session_id)
                
                # Run with timeout
                return loop.run_until_complete(
                    asyncio.wait_for(
                        self._collect_stream_result(agent, query, conversation_history, session_id),
                        timeout=600  # 10 minute timeout
                    )
                )
                
            except asyncio.TimeoutError:
                return "Research workflow timed out after 10 minutes."
            except Exception as e:
                return f"Research error: {str(e)}"
            
            finally:
                # Graceful cleanup
                try:
                    loop.run_until_complete(asyncio.sleep(0.1))  # Let streams finish
                    if not loop.is_closed():
                        loop.close()
                except:
                    pass
        
        # Run in thread pool for isolation
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_workflow)
                return future.result(timeout=650)  # Slightly longer than internal timeout
                
        except concurrent.futures.TimeoutError:
            return "Research workflow timed out."
        except Exception as e:
            return f"Research workflow error: {str(e)}"

    async def _collect_stream_result(self, agent: ResearchAgent, query: str, conversation_history, session_id: str) -> str:
        """Collect the final result from streaming workflow."""
        response_content = ""
        
        async for event_data in agent.stream_query_without_recompile(
            query, conversation_history, session_id
        ):
            if isinstance(event_data, dict):
                for node_name, node_data in event_data.items():
                    if node_name in ["__end__", "replan"] and isinstance(node_data, dict):
                        if "response" in node_data:
                            response_content = node_data["response"]
                            break
        
        return response_content or "Research completed successfully."
    
    def get_status(self) -> Dict[str, Any]:
        """Get basic system status."""
        return {
            "system_ready": self.is_ready(),
            "elasticsearch_connected": self.es_client is not None and self.es_client.ping() if self.es_client else False,
            "memory_sessions": len(self.memory_manager.session_memories),
            "query_stats": self.query_stats
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Simple health check."""
        es_healthy = self.es_client is not None and self.es_client.ping() if self.es_client else False
        
        return {
            "status": "healthy" if es_healthy else "degraded",
            "timestamp": time.time(),
            "elasticsearch": "healthy" if es_healthy else "unhealthy",
            "memory": "healthy"
        }
    
    def clear_memory(self, session_id: str) -> Dict[str, Any]:
        """Clear memory for session."""
        try:
            # ✅ FIXED: Use the persistent memory manager
            self.memory_manager.clear_session_memory(session_id)
            return {"success": True, "message": f"Cleared memory for session: {session_id}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get basic session information."""
        try:
            # ✅ FIXED: Use the persistent memory manager
            history = self.memory_manager.get_conversation_history_for_state(session_id)
            return {
                "success": True,
                "session_id": session_id,
                "conversation_messages": len(history),
                "conversation_preview": history[-2:] if len(history) >= 2 else history
            }
        except Exception as e:
            return {"success": False, "error": str(e), "session_id": session_id}
    
    def get_tools_info(self) -> Dict[str, Any]:
        """Get information about available tools."""
        try:
            if not TOOLS_AVAILABLE:
                return {"success": False, "error": "Tools not available", "total_tools": 0, "tools": []}
            
            tools = get_all_tools(self.es_client, self.index_name) if self.es_client else get_all_tools()
            tools_info = [{"name": t.name, "description": t.description, "type": str(type(t).__name__)} for t in tools]
            
            return {
                "success": True,
                "total_tools": len(tools_info),
                "tools": tools_info,
                "elasticsearch_connected": self.es_client is not None
            }
        except Exception as e:
            return {"success": False, "error": str(e), "total_tools": 0, "tools": []}


# Factory function
def create_agent_manager(index_name: str = "research-publications-static") -> AgentManager:
    """Create agent manager."""
    return AgentManager(index_name=index_name)