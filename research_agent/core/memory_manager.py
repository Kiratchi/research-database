"""
Trust LangChain Memory Manager - Simplified by trusting LangChain
Removed: SimpleMemory fallback, complex error handling, detailed stats
Kept: Smart cleanup, per-session isolation, core functionality
"""

import time
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, AIMessage

# Trust LangChain memory - use the best available
try:
    from langchain.memory import ConversationBufferWindowMemory
except ImportError:
    from langchain_community.memory import ConversationBufferWindowMemory


class MemoryManager:
    """Memory manager that trusts LangChain with smart cleanup."""
    
    def __init__(self, cleanup_interval: int = 3600, max_inactive_time: int = 3600):
        self.cleanup_interval = cleanup_interval  # How often to run cleanup (1 hour)
        self.max_inactive_time = max_inactive_time  # When to consider session old (1 hour)
        self.last_cleanup = time.time()
        self.session_memories = {}
        
        print("MemoryManager initialized")
    
    def get_memory_for_session(self, session_id: str):
        """Get or create memory for a specific session."""
        
        # Smart cleanup: run periodically
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self._smart_cleanup()
        
        if session_id not in self.session_memories:
            # Create LangChain memory - trust it to work
            memory = ConversationBufferWindowMemory(
                k=10,  # Keep last 10 messages (5 Q&A pairs)
                return_messages=True,
                memory_key="chat_history"
            )
            
            self.session_memories[session_id] = {
                "memory": memory,
                "created_at": time.time(),
                "last_accessed": time.time()
            }
            # Don't log memory creation for every session
        else:
            # Update access time for smart cleanup
            self.session_memories[session_id]["last_accessed"] = time.time()
        
        return self.session_memories[session_id]["memory"]
    
    def save_conversation(self, session_id: str, query: str, response: str):
        """Save a Q&A pair to conversation memory."""
        memory = self.get_memory_for_session(session_id)
        memory.save_context({"input": query}, {"output": response})
    
    def get_conversation_history_for_state(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history in format expected by workflow."""
        memory = self.get_memory_for_session(session_id)
        messages = memory.chat_memory.messages
        
        conversation_history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                conversation_history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                conversation_history.append({"role": "assistant", "content": msg.content})
        
        return conversation_history
    
    def clear_session_memory(self, session_id: str):
        """Clear memory for a session."""
        if session_id in self.session_memories:
            del self.session_memories[session_id]
            print(f"🗑️ Cleared memory for session: {session_id}")
    
    def _smart_cleanup(self):
        """Smart cleanup: remove sessions inactive for more than max_inactive_time."""
        current_time = time.time()
        old_sessions = []
        
        for session_id, session_data in self.session_memories.items():
            # Remove sessions that haven't been accessed recently
            if current_time - session_data["last_accessed"] > self.max_inactive_time:
                old_sessions.append(session_id)
        
        # Remove old sessions
        for session_id in old_sessions:
            del self.session_memories[session_id]
        
        self.last_cleanup = current_time
        
        if old_sessions:
            print(f"Cleaned up {len(old_sessions)} inactive sessions")
        
        # Additional smart cleanup: if we have too many sessions, remove oldest
        if len(self.session_memories) > 100:  # Max 100 active sessions
            # Sort by last_accessed and remove oldest
            sessions_by_access = sorted(
                self.session_memories.items(),
                key=lambda x: x[1]["last_accessed"]
            )
            
            # Remove oldest 20 sessions to get back to 80
            sessions_to_remove = sessions_by_access[:20]
            for session_id, _ in sessions_to_remove:
                del self.session_memories[session_id]
            
            print(f"Cleaned up {len(sessions_to_remove)} oldest sessions")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get basic memory statistics."""
        current_time = time.time()
        total_messages = 0
        active_sessions = 0
        
        for session_data in self.session_memories.values():
            memory = session_data["memory"]
            total_messages += len(memory.chat_memory.messages)
            
            # Count as active if accessed in last hour
            if current_time - session_data["last_accessed"] < 3600:
                active_sessions += 1
        
        return {
            "total_sessions": len(self.session_memories),
            "active_sessions": active_sessions,
            "total_messages": total_messages,
            "cleanup_info": {
                "last_cleanup": self.last_cleanup,
                "cleanup_interval": self.cleanup_interval,
                "max_inactive_time": self.max_inactive_time
            }
        }


def create_memory_manager() -> MemoryManager:
    """Factory function to create memory manager."""
    return MemoryManager()


if __name__ == "__main__":
    # Test the trust LangChain memory manager
    print("Testing MemoryManager with LangChain trust...")
    
    manager = create_memory_manager()
    
    # Test conversation
    session_id = "test_session"
    manager.save_conversation(session_id, "Hello", "Hi there!")
    manager.save_conversation(session_id, "How are you?", "I'm doing well!")
    
    # Test history retrieval
    history = manager.get_conversation_history_for_state(session_id)
    print(f"Retrieved {len(history)} messages")
    
    # Test stats
    stats = manager.get_memory_stats()
    print(f"Memory stats: {stats}")
    
    # Test cleanup (won't remove anything since session is new)
    manager._smart_cleanup()
    
    print("✅ MemoryManager test completed!")