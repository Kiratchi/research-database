"""
SIMPLIFIED Agent Manager - Clean Asyncio Context for Streaming
Key Fix: Use simple callback pattern instead of nested asyncio
"""

import os
import asyncio
import time
import uuid
import json
from typing import Dict, Any, Optional, AsyncGenerator, List
from elasticsearch import Elasticsearch

from .memory_singleton import get_global_memory_manager
from .workflow import ResearchAgent, StreamingEventEmitter, set_global_event_emitter

# Import tools
try:
    from ..tools import get_all_tools
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False


class StreamingAgentManager:
    """Simplified agent manager with clean asyncio context for streaming."""
    
    def __init__(self, index_name: str = "research-publications-static"):
        self.index_name = index_name
        self.query_stats = {"total": 0, "success": 0, "failed": 0, "streaming": 0}
        
        self.es_client = self._init_elasticsearch()
        self.memory_manager = get_global_memory_manager()
        
        print("Enhanced AgentManager initialized with streaming support")
    
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
    
    # BACKWARD COMPATIBILITY: Keep existing methods unchanged
    async def process_query_async(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Process query asynchronously - backward compatible version."""
        session_id = session_id or f'session_{int(time.time())}_{str(uuid.uuid4())[:8]}'
        self.query_stats["total"] += 1
        start_time = time.time()
        
        try:
            if not self.is_ready():
                self.query_stats["failed"] += 1
                return {
                    "success": False,
                    "error": "System not ready - Elasticsearch required",
                    "session_id": session_id
                }
            
            conversation_history = self.memory_manager.get_conversation_history_for_state(session_id)
            
            # Use the enhanced execution with reasoning data collection
            response_content, reasoning_data = await asyncio.wait_for(
                self._execute_research_with_reasoning_collection(query, conversation_history, session_id),
                timeout=600
            )
            
            if response_content:
                self.memory_manager.save_conversation(session_id, query, response_content)
                self.query_stats["success"] += 1
                return {
                    "success": True,
                    "response": response_content,
                    "reasoning_data": reasoning_data,
                    "session_id": session_id,
                    "execution_time": time.time() - start_time,
                    "response_type": "research_with_reasoning"
                }
            else:
                self.query_stats["failed"] += 1
                return {
                    "success": False,
                    "error": "No response generated",
                    "session_id": session_id
                }
            
        except asyncio.TimeoutError:
            self.query_stats["failed"] += 1
            return {
                "success": False,
                "error": "Research workflow timed out after 10 minutes",
                "session_id": session_id
            }
        except Exception as e:
            self.query_stats["failed"] += 1
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    # SIMPLIFIED: Streaming method with clean asyncio context
    async def stream_query_with_reasoning(
        self, 
        query: str, 
        session_id: str = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream query with SIMPLIFIED real-time reasoning steps."""
        
        session_id = session_id or f'session_{int(time.time())}_{str(uuid.uuid4())[:8]}'
        self.query_stats["total"] += 1
        self.query_stats["streaming"] += 1
        start_time = time.time()
        
        print(f"🎯 Starting SIMPLIFIED streaming query for session: {session_id}")
        
        try:
            if not self.is_ready():
                yield {
                    "event": "error",
                    "data": {
                        "error": "System not ready - Elasticsearch required",
                        "session_id": session_id
                    }
                }
                return
            
            # SIMPLIFIED EVENT SYSTEM - Use list to collect events
            collected_events: List[Dict[str, Any]] = []
            
            # Simple event collector callback
            def collect_event(event):
                """Simple synchronous event collector."""
                print(f"📥 Collected event: {event['event']}")
                collected_events.append(event)
            
            # Create event emitter with simple callback
            event_emitter = StreamingEventEmitter()
            event_emitter.subscribe(collect_event)
            set_global_event_emitter(event_emitter)
            
            print(f"🔧 Simplified event system set up")
            
            # Emit initial event
            yield {
                "event": "start",
                "data": {
                    "query": query,
                    "session_id": session_id,
                    "timestamp": time.time()
                }
            }
            
            conversation_history = self.memory_manager.get_conversation_history_for_state(session_id)
            
            # Create agent with event emitter
            agent = ResearchAgent(
                es_client=self.es_client,
                index_name=self.index_name,
                recursion_limit=50,
                memory_manager=self.memory_manager
            )
            agent.set_event_emitter(event_emitter)
            
            print(f"🚀 Starting agent execution...")
            
            # Execute agent and collect results
            final_response = None
            final_reasoning_data = None
            
            try:
                async for event in agent.stream_query(query, conversation_history, session_id):
                    print(f"🔄 Agent stream event: {list(event.keys())}")
                    
                    # Process any collected events first
                    while collected_events:
                        collected_event = collected_events.pop(0)
                        print(f"📤 Streaming collected event: {collected_event['event']}")
                        yield collected_event
                        
                        # Small delay for visual effect
                        await asyncio.sleep(0.1)
                    
                    # Check for final response in agent event
                    if 'react' in event and 'response' in event['react']:
                        final_response = event['react']['response']
                        final_reasoning_data = event['react'].get('_frontend_reasoning_data')
                        print(f"✅ Got final response: {len(final_response)} chars")
                
                # Process any remaining collected events
                while collected_events:
                    collected_event = collected_events.pop(0)
                    print(f"📤 Streaming final collected event: {collected_event['event']}")
                    yield collected_event
                    await asyncio.sleep(0.1)
                
            except Exception as agent_error:
                print(f"❌ Agent execution error: {agent_error}")
                yield {
                    "event": "error",
                    "data": {
                        "error": f"Agent execution error: {str(agent_error)}",
                        "session_id": session_id
                    }
                }
                return
            
            # Save to memory and emit completion
            if final_response:
                self.memory_manager.save_conversation(session_id, query, final_response)
                self.query_stats["success"] += 1
                
                # Emit final completion event
                yield {
                    "event": "complete",
                    "data": {
                        "success": True,
                        "response": final_response,
                        "reasoning_data": final_reasoning_data,
                        "session_id": session_id,
                        "execution_time": time.time() - start_time
                    }
                }
            else:
                self.query_stats["failed"] += 1
                yield {
                    "event": "error", 
                    "data": {
                        "error": "No response generated",
                        "session_id": session_id
                    }
                }
        
        except Exception as e:
            print(f"❌ Critical streaming error: {e}")
            self.query_stats["failed"] += 1
            yield {
                "event": "error",
                "data": {
                    "error": str(e),
                    "session_id": session_id
                }
            }
    
    async def _execute_research_with_reasoning_collection(
        self, 
        query: str, 
        conversation_history, 
        session_id: str
    ) -> tuple[str, Dict[str, Any]]:
        """Execute research and collect reasoning data (non-streaming)."""
        
        try:
            agent = ResearchAgent(
                es_client=self.es_client,
                index_name=self.index_name,
                recursion_limit=50,
                memory_manager=self.memory_manager
            )
            agent._compile_agent(session_id)
            
            response_content = ""
            reasoning_data = None
            
            try:
                async for event_data in agent.stream_query(query, conversation_history, session_id):
                    if isinstance(event_data, dict):
                        for node_name, node_data in event_data.items():
                            if node_name in ["__end__", "react"] and isinstance(node_data, dict):
                                if "response" in node_data:
                                    response_content = node_data["response"]
                                # Get frontend reasoning data
                                if "_frontend_reasoning_data" in node_data:
                                    reasoning_data = node_data["_frontend_reasoning_data"]
                
            except GeneratorExit:
                pass
            except Exception as stream_error:
                return f"Streaming error: {str(stream_error)}", None
            
            return response_content or "Research completed successfully.", reasoning_data
            
        except Exception as e:
            return f"Research error: {str(e)}", None
    
    # UTILITY METHODS (unchanged from original)
    def get_status(self) -> Dict[str, Any]:
        """Get basic system status."""
        return {
            "system_ready": self.is_ready(),
            "elasticsearch_connected": self.es_client is not None and self.es_client.ping() if self.es_client else False,
            "memory_sessions": len(self.memory_manager.session_memories),
            "query_stats": self.query_stats,
            "streaming_enabled": True
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Simple health check."""
        es_healthy = self.es_client is not None and self.es_client.ping() if self.es_client else False
        
        return {
            "status": "healthy" if es_healthy else "degraded",
            "timestamp": time.time(),
            "elasticsearch": "healthy" if es_healthy else "unhealthy",
            "memory": "healthy",
            "streaming": "enabled"
        }
    
    def clear_memory(self, session_id: str) -> Dict[str, Any]:
        """Clear memory for session."""
        try:
            self.memory_manager.clear_session_memory(session_id)
            return {"success": True, "message": f"Cleared memory for session: {session_id}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get basic session information."""
        try:
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


# Backward compatibility alias
AgentManager = StreamingAgentManager


def create_agent_manager(index_name: str = "research-publications-static") -> StreamingAgentManager:
    """Create enhanced agent manager with streaming support."""
    return StreamingAgentManager(index_name=index_name)


if __name__ == "__main__":
    print("SIMPLIFIED Agent Manager with clean asyncio context ready!")