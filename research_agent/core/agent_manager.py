"""
Agent Manager - Improved version with better async handling and error recovery
"""

import os
import json
import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional
from elasticsearch import Elasticsearch

from .memory_manager import ConversationMemoryManager
from .workflow import ResearchAgent


class AgentManager:
    """
    Simple coordinator for the research agent system.
    
    Manages Elasticsearch, memory, and workflow execution.
    """
    
    def __init__(self, index_name: str = "research-publications-static"):
        """Initialize the agent manager."""
        self.index_name = index_name
        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0
        }
        
        # Initialize components
        self.es_client = self._init_elasticsearch()
        self.memory_manager = self._init_memory()
        self.research_agent = self._init_research_agent()
        
        print("🚀 AgentManager initialized")
    
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
                print("✅ Elasticsearch connected")
                return es_client
            else:
                print("❌ Elasticsearch connection failed")
                return None
                
        except Exception as e:
            print(f"❌ Elasticsearch error: {e}")
            return None
    
    def _init_memory(self) -> ConversationMemoryManager:
        """Initialize memory manager."""
        try:
            memory_manager = ConversationMemoryManager(
                memory_type="buffer_window",
                cleanup_interval=3600
            )
            print("✅ Memory manager initialized")
            return memory_manager
        except Exception as e:
            print(f"❌ Memory error: {e}")
            return ConversationMemoryManager()
    
    def _init_research_agent(self) -> Optional[ResearchAgent]:
        """Initialize research agent."""
        try:
            if not self.es_client:
                print("⚠️ Research agent initialized without Elasticsearch")
                return None
            
            research_agent = ResearchAgent(
                es_client=self.es_client,
                index_name=self.index_name,
                recursion_limit=50
            )
            print("✅ Research agent initialized")
            return research_agent
        except Exception as e:
            print(f"❌ Research agent error: {e}")
            return None
    
    def is_ready(self) -> bool:
        """Check if system is ready."""
        return (
            self.es_client is not None and 
            self.memory_manager is not None and
            self.research_agent is not None and
            self.es_client.ping()
        )
    
    def process_query(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """
        Process a research query with improved async handling.
        """
        # Generate session ID if needed
        if not session_id:
            session_id = f'session_{int(time.time())}_{str(uuid.uuid4())[:8]}'
        
        self.query_stats["total_queries"] += 1
        start_time = time.time()
        
        try:
            print(f"🔍 Processing: '{query}' (session: {session_id})")
            
            # Handle simple queries
            simple_response = self._handle_simple_query(query)
            if simple_response:
                self.memory_manager.save_conversation(session_id, query, simple_response)
                self.query_stats["successful_queries"] += 1
                
                return {
                    "success": True,
                    "response": simple_response,
                    "session_id": session_id,
                    "execution_time": time.time() - start_time,
                    "response_type": "simple"
                }
            
            # Check system readiness
            if not self.is_ready():
                self.query_stats["failed_queries"] += 1
                return {
                    "success": False,
                    "error": "System not ready - Elasticsearch required",
                    "session_id": session_id
                }
            
            # Execute research workflow
            conversation_history = self.memory_manager.get_conversation_history_for_state(session_id)
            response_content = self._execute_workflow_safe(query, conversation_history, session_id)
            
            if not response_content:
                self.query_stats["failed_queries"] += 1
                return {
                    "success": False,
                    "error": "No response generated",
                    "session_id": session_id
                }
            
            # Save to memory
            self.memory_manager.save_conversation(session_id, query, response_content)
            self.query_stats["successful_queries"] += 1
            
            return {
                "success": True,
                "response": response_content,
                "session_id": session_id,
                "execution_time": time.time() - start_time,
                "response_type": "research"
            }
            
        except Exception as e:
            self.query_stats["failed_queries"] += 1
            error_msg = f"Error processing query: {str(e)}"
            print(f"❌ {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "session_id": session_id,
                "execution_time": time.time() - start_time
            }
    
    def _handle_simple_query(self, query: str) -> Optional[str]:
        """Handle simple queries like greetings."""
        query_clean = query.lower().strip().rstrip('!?.,;:')
        
        # Greetings
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon']
        if query_clean in greetings:
            return "Hello! I'm here to help you search and analyze research publications. What would you like to know?"
        
        # Thanks
        thanks = ['thanks', 'thank you', 'thx', 'ty']
        if query_clean in thanks:
            return "You're welcome! Feel free to ask about research publications."
        
        # Help
        help_patterns = ['help', 'what can you do', 'how does this work']
        if any(pattern in query_clean for pattern in help_patterns):
            return ("I can help you search and analyze research publications:\n"
                   "• Find papers by author, title, or topic\n"
                   "• Count publications and analyze trends\n"
                   "• Answer questions about specific papers\n"
                   "• Handle follow-up questions in context\n\n"
                   "Just ask me anything about research publications!")
        
        return None
    
    def _execute_workflow_safe(self, query: str, conversation_history: List, session_id: str) -> str:
        """Execute the research workflow with standard async handling."""
        if not self.research_agent:
            raise Exception("Research agent not available")
        
        print(f"🔬 Executing workflow for: '{query}'")
        print(f"📝 Context: {len(conversation_history)} previous messages")
        
        # Create new event loop for clean execution
        loop = None
        try:
            # Try to get existing loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # No loop exists, create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def run_workflow_streaming():
            """Process streaming events naturally without fighting GeneratorExit."""
            response_content = ""
            event_count = 0
            
            # STANDARD PATTERN - Let GeneratorExit propagate naturally
            async for event_data in self.research_agent.stream_query(query, conversation_history):
                event_count += 1
                
                # Process events normally - no timeout or GeneratorExit handling needed
                if isinstance(event_data, dict):
                    # Look for final response in various possible locations
                    for node_name, node_data in event_data.items():
                        if node_name == "__end__" and isinstance(node_data, dict):
                            if "response" in node_data:
                                response_content = node_data["response"]
                                print(f"✅ Found final response in __end__ node")
                                return response_content
                        elif node_name == "complete" and isinstance(node_data, dict):
                            if "response" in node_data:
                                response_content = node_data["response"]
                                print(f"✅ Found final response in complete node")
                                return response_content
                        elif node_name == "replan" and isinstance(node_data, dict):
                            if "final_response" in node_data:
                                response_content = node_data["final_response"]
                                print(f"✅ Found final response in replan node")
                                return response_content
                        elif node_name == "error" and isinstance(node_data, dict):
                            error_msg = node_data.get("error", "Unknown error")
                            print(f"❌ Error event received: {error_msg}")
                            return f"Error during research: {error_msg}"
                
                # Handle string event data (legacy format)
                elif isinstance(event_data, str):
                    try:
                        if event_data.strip():
                            event = json.loads(event_data.strip())
                            if event.get("type") == "final":
                                response_content = event.get("content", {}).get("response", "")
                                if response_content:
                                    print(f"✅ Workflow completed after {event_count} events")
                                    return response_content
                    except json.JSONDecodeError:
                        continue
            
            # If we get here, the stream ended naturally
            if not response_content:
                print(f"⚠️ Stream ended naturally after {event_count} events")
                response_content = "Research completed but no response generated."
            
            return response_content
        
        try:
            # Run the workflow - GeneratorExit will be handled naturally by LangGraph
            result = loop.run_until_complete(run_workflow_streaming())
            return result
            
        except Exception as e:
            print(f"❌ Event loop error: {e}")
            return f"Error running workflow: {str(e)}"
        finally:
            # Clean event loop shutdown
            try:
                if loop and not loop.is_closed():
                    # Wait for any remaining tasks to complete
                    pending = asyncio.all_tasks(loop)
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    loop.close()
                    print("🧹 Event loop properly closed")
            except Exception as e:
                print(f"⚠️ Loop cleanup warning: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        es_connected = False
        if self.es_client:
            try:
                es_connected = self.es_client.ping()
            except:
                es_connected = False
        
        memory_stats = self.memory_manager.get_memory_stats() if self.memory_manager else {}
        
        return {
            "system_ready": self.is_ready(),
            "elasticsearch": {
                "connected": es_connected,
                "host": os.getenv("ES_HOST", "Not configured"),
                "index": self.index_name
            },
            "memory": {
                "initialized": self.memory_manager is not None,
                "total_sessions": memory_stats.get("total_sessions", 0),
                "type": memory_stats.get("memory_type", "unknown")
            },
            "research_agent": {
                "initialized": self.research_agent is not None
            },
            "statistics": self.query_stats
        }
    
    def clear_memory(self, session_id: str) -> Dict[str, Any]:
        """Clear memory for a session."""
        try:
            self.memory_manager.clear_session_memory(session_id)
            return {
                "success": True,
                "message": f"Memory cleared for session: {session_id}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error clearing memory: {str(e)}"
            }
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information."""
        try:
            conversation_history = self.memory_manager.get_conversation_history_for_state(session_id)
            return {
                "success": True,
                "session_id": session_id,
                "message_count": len(conversation_history),
                "conversation_history": conversation_history
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error getting session info: {str(e)}"
            }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            return self.memory_manager.get_memory_stats()
        except Exception as e:
            return {"error": f"Error getting memory stats: {str(e)}"}
    
    def health_check(self) -> Dict[str, Any]:
        """Simple health check."""
        health = {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {}
        }
        
        # Check Elasticsearch
        try:
            if self.es_client and self.es_client.ping():
                health["checks"]["elasticsearch"] = "healthy"
            else:
                health["checks"]["elasticsearch"] = "unhealthy"
                health["status"] = "degraded"
        except:
            health["checks"]["elasticsearch"] = "unhealthy"
            health["status"] = "degraded"
        
        # Check memory
        if self.memory_manager:
            health["checks"]["memory"] = "healthy"
        else:
            health["checks"]["memory"] = "unhealthy"
            health["status"] = "degraded"
        
        # Check research agent
        if self.research_agent:
            health["checks"]["research_agent"] = "healthy"
        else:
            health["checks"]["research_agent"] = "unhealthy"
            health["status"] = "degraded"
        
        return health


def create_agent_manager(index_name: str = "research-publications-static") -> AgentManager:
    """Create an agent manager."""
    return AgentManager(index_name=index_name)


if __name__ == "__main__":
    # Test the agent manager
    print("Testing AgentManager...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Create manager
    manager = create_agent_manager()
    
    # Test status
    status = manager.get_status()
    print(f"Status: {status}")
    
    # Test simple query
    result = manager.process_query("Hello!")
    print(f"Simple query: {result}")
    
    print("✅ AgentManager test completed!")