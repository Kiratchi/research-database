"""
Pure LangChain Memory Manager - Session-based conversation memory
REPLACES: Complex IntegratedMemoryManager with custom memory system  
USES: LangChain's proven memory systems with automatic context injection
FILENAME: memory_manager.py (maintains existing import structure)
"""

import time
import os
from typing import Dict, List, Optional, Any
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory, ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_litellm import ChatLiteLLM


class SessionMemoryManager:
    """
    Manages LangChain memory instances per session.
    Simple, clean, and leverages LangChain's automatic memory injection.
    """
    
    def __init__(self, default_memory_type: str = "buffer_window"):
        """Initialize session-based LangChain memory manager."""
        self.default_memory_type = default_memory_type
        self.session_memories: Dict[str, Any] = {}
        self.session_metadata: Dict[str, Dict] = {}
        self.cleanup_interval = 3600  # 1 hour
        self.last_cleanup = time.time()
        
        print(f"🧠 SessionMemoryManager initialized with {default_memory_type} memory")
        print("✅ Pure LangChain memory - automatic conversation context injection")
    
    def get_memory_for_session(self, session_id: str) -> Any:
        """
        Get or create LangChain memory for session.
        Returns appropriate LangChain memory instance with automatic context injection.
        """
        
        # Periodic cleanup
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_sessions()
        
        if session_id not in self.session_memories:
            # Create new memory for session
            memory = self._create_memory(session_id)
            self.session_memories[session_id] = memory
            self.session_metadata[session_id] = {
                "created_at": time.time(),
                "last_accessed": time.time(),
                "message_count": 0,
                "memory_type": self.default_memory_type
            }
            print(f"🧠 Created new LangChain memory for session: {session_id}")
        else:
            # Update last accessed time
            self.session_metadata[session_id]["last_accessed"] = time.time()
        
        return self.session_memories[session_id]
    
    def _create_memory(self, session_id: str) -> Any:
        """Create appropriate LangChain memory instance."""
        
        if self.default_memory_type == "buffer_window":
            try:
                memory = ConversationBufferWindowMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    k=10,  # Keep last 10 messages (5 Q&A pairs)
                    input_key="input",
                    output_key="output"
                )
                print(f"✅ Created BufferWindow memory for session {session_id} (k=10)")
                return memory
                
            except Exception as e:
                print(f"❌ Failed to create buffer window memory: {e}")
                return self._create_fallback_memory(session_id)
        
        elif self.default_memory_type == "summary":
            try:
                # Use cheaper model for summarization
                llm = ChatLiteLLM(
                    model="anthropic/claude-haiku-3.5",
                    api_key=os.getenv("LITELLM_API_KEY"),
                    api_base=os.getenv("LITELLM_BASE_URL"),
                    temperature=0
                )
                
                memory = ConversationSummaryMemory(
                    llm=llm,
                    memory_key="chat_history",
                    return_messages=True,
                    max_token_limit=4000,
                    input_key="input",
                    output_key="output"
                )
                print(f"✅ Created Summary memory for session {session_id} (4k tokens)")
                return memory
                
            except Exception as e:
                print(f"❌ Failed to create summary memory: {e}")
                return self._create_fallback_memory(session_id)
        
        elif self.default_memory_type == "buffer":
            try:
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    input_key="input",
                    output_key="output"
                )
                print(f"✅ Created Buffer memory for session {session_id} (unlimited)")
                return memory
                
            except Exception as e:
                print(f"❌ Failed to create buffer memory: {e}")
                return self._create_fallback_memory(session_id)
        
        else:
            print(f"⚠️ Unknown memory type: {self.default_memory_type}, using buffer_window")
            return self._create_fallback_memory(session_id)
    
    def _create_fallback_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """Create fallback memory if other types fail."""
        try:
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=8,  # Slightly smaller for fallback
                input_key="input",
                output_key="output"
            )
            print(f"✅ Created fallback BufferWindow memory for session {session_id}")
            return memory
        except Exception as e:
            print(f"❌ Even fallback memory failed: {e}")
            raise Exception(f"Could not create any memory type for session {session_id}")
    
    def save_conversation(self, session_id: str, user_input: str, ai_response: str):
        """
        Save conversation to memory.
        Note: With AgentExecutor, this happens automatically, but provided for manual use.
        """
        try:
            memory = self.get_memory_for_session(session_id)
            memory.save_context(
                {"input": user_input},
                {"output": ai_response}
            )
            
            # Update metadata
            if session_id in self.session_metadata:
                self.session_metadata[session_id]["message_count"] += 2  # User + AI message
                self.session_metadata[session_id]["last_accessed"] = time.time()
            
            print(f"💾 Saved conversation to LangChain memory for session: {session_id}")
            
        except Exception as e:
            print(f"❌ Error saving conversation for session {session_id}: {e}")
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get conversation history in a readable format.
        Returns: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        try:
            memory = self.get_memory_for_session(session_id)
            messages = memory.chat_memory.messages
            
            conversation_history = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    conversation_history.append({
                        "role": "user",
                        "content": msg.content
                    })
                elif isinstance(msg, AIMessage):
                    conversation_history.append({
                        "role": "assistant",
                        "content": msg.content
                    })
            
            return conversation_history
            
        except Exception as e:
            print(f"❌ Error retrieving conversation history for session {session_id}: {e}")
            return []
    
    def get_memory_variables(self, session_id: str) -> Dict[str, Any]:
        """
        Get memory variables that LangChain uses for prompt injection.
        This shows what gets injected into {chat_history}.
        """
        try:
            memory = self.get_memory_for_session(session_id)
            variables = memory.load_memory_variables({})
            return variables
        except Exception as e:
            print(f"❌ Error getting memory variables for session {session_id}: {e}")
            return {"chat_history": []}
    
    def clear_session_memory(self, session_id: str):
        """Clear memory for a specific session."""
        try:
            if session_id in self.session_memories:
                del self.session_memories[session_id]
                print(f"🗑️ Cleared LangChain memory for session: {session_id}")
            
            if session_id in self.session_metadata:
                del self.session_metadata[session_id]
                print(f"🗑️ Cleared metadata for session: {session_id}")
            
        except Exception as e:
            print(f"❌ Error clearing memory for session {session_id}: {e}")
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a specific session."""
        try:
            if session_id not in self.session_memories:
                return {
                    "exists": False,
                    "session_id": session_id,
                    "error": "Session not found"
                }
            
            metadata = self.session_metadata.get(session_id, {})
            conversation_history = self.get_conversation_history(session_id)
            
            current_time = time.time()
            
            return {
                "exists": True,
                "session_id": session_id,
                "memory_type": metadata.get("memory_type", "unknown"),
                "message_count": len(conversation_history),
                "created_at": metadata.get("created_at", 0),
                "last_accessed": metadata.get("last_accessed", 0),
                "age_minutes": (current_time - metadata.get("created_at", current_time)) / 60,
                "inactive_minutes": (current_time - metadata.get("last_accessed", current_time)) / 60,
                "conversation_preview": conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history
            }
            
        except Exception as e:
            return {
                "exists": False,
                "session_id": session_id,
                "error": f"Error getting session info: {str(e)}"
            }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about all sessions."""
        current_time = time.time()
        
        stats = {
            "total_sessions": len(self.session_memories),
            "memory_type": self.default_memory_type,
            "architecture": "pure_langchain",
            "automatic_context_injection": True,
            "manual_context_building": False,
            "sessions": []
        }
        
        total_messages = 0
        
        for session_id, metadata in self.session_metadata.items():
            conversation_history = self.get_conversation_history(session_id)
            message_count = len(conversation_history)
            total_messages += message_count
            
            session_stats = {
                "session_id": session_id,
                "memory_type": metadata.get("memory_type", "unknown"),
                "message_count": message_count,
                "created_at": metadata.get("created_at", 0),
                "last_accessed": metadata.get("last_accessed", 0),
                "age_minutes": (current_time - metadata.get("created_at", current_time)) / 60,
                "inactive_minutes": (current_time - metadata.get("last_accessed", current_time)) / 60
            }
            stats["sessions"].append(session_stats)
        
        stats["total_messages"] = total_messages
        stats["average_messages_per_session"] = total_messages / len(self.session_memories) if self.session_memories else 0
        
        return stats
    
    def _cleanup_old_sessions(self):
        """Remove sessions that haven't been accessed recently."""
        current_time = time.time()
        old_sessions = []
        
        for session_id, metadata in self.session_metadata.items():
            # Remove sessions older than 2 hours
            if current_time - metadata.get("last_accessed", current_time) > 7200:
                old_sessions.append(session_id)
        
        for session_id in old_sessions:
            self.clear_session_memory(session_id)
        
        self.last_cleanup = current_time
        
        if old_sessions:
            print(f"🧹 Cleaned up {len(old_sessions)} old LangChain memory sessions")
    
    def switch_memory_type(self, session_id: str, new_memory_type: str):
        """
        Switch memory type for a session (advanced feature).
        Useful for adapting to long conversations.
        """
        try:
            if session_id not in self.session_memories:
                print(f"⚠️ Session {session_id} not found for memory type switch")
                return False
            
            # Get current conversation history
            current_history = self.get_conversation_history(session_id)
            
            # Clear current memory
            del self.session_memories[session_id]
            
            # Create new memory type
            old_type = self.default_memory_type
            self.default_memory_type = new_memory_type
            new_memory = self._create_memory(session_id)
            self.default_memory_type = old_type  # Reset default
            
            # Restore conversation history to new memory
            for i in range(0, len(current_history), 2):
                if i + 1 < len(current_history):
                    user_msg = current_history[i]
                    ai_msg = current_history[i + 1]
                    if user_msg["role"] == "user" and ai_msg["role"] == "assistant":
                        new_memory.save_context(
                            {"input": user_msg["content"]},
                            {"output": ai_msg["content"]}
                        )
            
            # Update session
            self.session_memories[session_id] = new_memory
            self.session_metadata[session_id]["memory_type"] = new_memory_type
            self.session_metadata[session_id]["last_accessed"] = time.time()
            
            print(f"🔄 Switched session {session_id} from {old_type} to {new_memory_type} memory")
            print(f"📚 Restored {len(current_history)} messages to new memory")
            
            return True
            
        except Exception as e:
            print(f"❌ Error switching memory type for session {session_id}: {e}")
            return False


def create_session_memory_manager(memory_type: str = "buffer_window") -> SessionMemoryManager:
    """Factory function to create session memory manager."""
    return SessionMemoryManager(default_memory_type=memory_type)


if __name__ == "__main__":
    print("Testing Pure LangChain SessionMemoryManager...")
    
    # Create memory manager
    memory_manager = create_session_memory_manager("buffer_window")
    
    # Test session 1
    session1 = "test_session_001"
    
    # Test memory creation
    memory1 = memory_manager.get_memory_for_session(session1)
    print(f"Memory for {session1}: {type(memory1).__name__}")
    
    # Test conversation saving
    memory_manager.save_conversation(session1, "Who is Per-Olof Arnäs?", "He's a logistics researcher at Chalmers University...")
    memory_manager.save_conversation(session1, "What are his research areas?", "His main areas include freight transportation and digital transformation...")
    
    # Test conversation retrieval
    history = memory_manager.get_conversation_history(session1)
    print(f"Conversation history: {len(history)} messages")
    for msg in history:
        print(f"  {msg['role']}: {msg['content'][:50]}...")
    
    # Test memory variables (what gets injected into prompts)
    variables = memory_manager.get_memory_variables(session1)
    print(f"Memory variables: {list(variables.keys())}")
    
    # Test session info
    info = memory_manager.get_session_info(session1)
    print(f"Session info: {info['message_count']} messages, age: {info['age_minutes']:.1f} min")
    
    # Test memory stats
    stats = memory_manager.get_memory_stats()
    print(f"Total sessions: {stats['total_sessions']}, total messages: {stats['total_messages']}")
    
    # Test memory type switching
    print("\nTesting memory type switch...")
    success = memory_manager.switch_memory_type(session1, "summary")
    if success:
        switched_memory = memory_manager.get_memory_for_session(session1)
        print(f"Switched to: {type(switched_memory).__name__}")
    
    # Test cleanup
    memory_manager.clear_session_memory(session1)
    final_stats = memory_manager.get_memory_stats()
    print(f"After cleanup: {final_stats['total_sessions']} sessions")
    
    print("\n✅ Pure LangChain SessionMemoryManager test completed!")
    print("🎯 Key features:")
    print("  - Automatic conversation context injection via {chat_history}")
    print("  - Multiple LangChain memory types supported")
    print("  - Session-based memory management")
    print("  - Memory type switching for long conversations")
    print("  - No manual context building required")