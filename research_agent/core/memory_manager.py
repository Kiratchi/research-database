"""
Fixed Conversation Memory Manager - addresses LangChain deprecation warnings
"""

import time
import os
import warnings
from typing import Dict, List, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_litellm import ChatLiteLLM

# Suppress the specific deprecation warning for now
warnings.filterwarnings("ignore", message="Please see the migration guide at: https://python.langchain.com/docs/versions/migrating_memory/")

# Use the new memory imports if available, fall back to legacy
try:
    from langchain.memory import (
        ConversationBufferWindowMemory, 
        ConversationSummaryMemory,
        ConversationBufferMemory
    )
except ImportError:
    # Fallback for newer versions
    try:
        from langchain_community.memory import (
            ConversationBufferWindowMemory,
            ConversationSummaryMemory, 
            ConversationBufferMemory
        )
    except ImportError:
        print("⚠️ LangChain memory classes not available, using basic implementation")
        ConversationBufferWindowMemory = None
        ConversationSummaryMemory = None
        ConversationBufferMemory = None


class SimpleMemory:
    """Fallback memory implementation if LangChain memory is not available."""
    
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.messages = []
    
    def save_context(self, inputs: dict, outputs: dict):
        """Save conversation context."""
        self.messages.append(HumanMessage(content=inputs.get("input", "")))
        self.messages.append(AIMessage(content=outputs.get("output", "")))
        
        # Keep only last N messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    @property
    def chat_memory(self):
        """Return a simple object with messages attribute."""
        class ChatMemory:
            def __init__(self, messages):
                self.messages = messages
        return ChatMemory(self.messages)


class ConversationMemoryManager:
    """
    Manages conversation memory with fallback for LangChain compatibility.
    """
    
    def __init__(self, memory_type: str = "buffer_window", cleanup_interval: int = 3600):
        """Initialize the memory manager."""
        self.memory_type = memory_type
        self.session_memories = {}
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
        
        print(f"🧠 ConversationMemoryManager initialized with {memory_type} memory")
    
    def get_memory_for_session(self, session_id: str):
        """Get or create memory for a specific session."""
        
        # Periodic cleanup
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_sessions()
        
        if session_id not in self.session_memories:
            self.session_memories[session_id] = {
                "memory": self._create_memory(),
                "created_at": time.time(),
                "last_accessed": time.time()
            }
            print(f"📝 Created new memory for session: {session_id}")
        else:
            # Update last accessed time
            self.session_memories[session_id]["last_accessed"] = time.time()
        
        return self.session_memories[session_id]["memory"]
    
    def _create_memory(self):
        """Create a new memory instance based on memory type."""
        
        if self.memory_type == "buffer_window" and ConversationBufferWindowMemory:
            try:
                return ConversationBufferWindowMemory(
                    k=10,  # Keep last 10 messages (5 Q&A pairs)
                    return_messages=True,
                    memory_key="chat_history"
                )
            except Exception as e:
                print(f"⚠️ Failed to create buffer window memory: {e}")
                return SimpleMemory(max_messages=10)
        
        elif self.memory_type == "summary" and ConversationSummaryMemory:
            # Use cheaper model for summarization
            try:
                llm = ChatLiteLLM(
                    model="anthropic/claude-haiku-3",
                    api_key=os.getenv("LITELLM_API_KEY"),
                    api_base=os.getenv("LITELLM_BASE_URL"),
                    temperature=0
                )
                
                return ConversationSummaryMemory(
                    llm=llm,
                    return_messages=True,
                    memory_key="chat_history",
                    max_token_limit=2000
                )
            except Exception as e:
                print(f"⚠️ Failed to create summary memory, falling back to simple: {e}")
                return SimpleMemory(max_messages=10)
        
        elif self.memory_type == "buffer" and ConversationBufferMemory:
            try:
                return ConversationBufferMemory(
                    return_messages=True,
                    memory_key="chat_history"
                )
            except Exception as e:
                print(f"⚠️ Failed to create buffer memory: {e}")
                return SimpleMemory(max_messages=20)
        
        else:
            print(f"⚠️ Unknown memory type or LangChain not available: {self.memory_type}, using simple memory")
            return SimpleMemory(max_messages=10)
    
    def save_conversation(self, session_id: str, query: str, response: str):
        """Save a Q&A pair to conversation memory."""
        try:
            memory = self.get_memory_for_session(session_id)
            memory.save_context(
                {"input": query},
                {"output": response}
            )
            print(f"💾 Saved conversation to memory for session: {session_id}")
            
        except Exception as e:
            print(f"❌ Error saving conversation for session {session_id}: {e}")
    
    def get_conversation_history_for_state(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get conversation history in PlanExecuteState format.
        
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
    
    def get_conversation_context_string(self, session_id: str, max_messages: int = 6) -> str:
        """Get formatted conversation context for prompts."""
        conversation_history = self.get_conversation_history_for_state(session_id)
        
        if not conversation_history:
            return "No previous conversation history."
        
        # Take the most recent messages
        recent_messages = conversation_history[-max_messages:] if len(conversation_history) > max_messages else conversation_history
        
        context_parts = []
        for msg in recent_messages:
            role = msg["role"].title()
            content = msg["content"]
            
            # Truncate very long messages
            if len(content) > 300:
                content = content[:300] + "..."
            
            context_parts.append(f"- {role}: {content}")
        
        return "\n".join(context_parts)
    
    def clear_session_memory(self, session_id: str):
        """Clear memory for a specific session."""
        if session_id in self.session_memories:
            del self.session_memories[session_id]
            print(f"🗑️ Cleared memory for session: {session_id}")
    
    def _cleanup_old_sessions(self):
        """Remove sessions that haven't been accessed recently."""
        current_time = time.time()
        old_sessions = []
        
        for session_id, session_data in self.session_memories.items():
            # Remove sessions older than 1 hour
            if current_time - session_data["last_accessed"] > 3600:
                old_sessions.append(session_id)
        
        for session_id in old_sessions:
            del self.session_memories[session_id]
        
        self.last_cleanup = current_time
        
        if old_sessions:
            print(f"🧹 Cleaned up {len(old_sessions)} old memory sessions")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about current memory usage."""
        stats = {
            "total_sessions": len(self.session_memories),
            "memory_type": self.memory_type,
            "sessions": []
        }
        
        current_time = time.time()
        for session_id, session_data in self.session_memories.items():
            memory = session_data["memory"]
            
            # Handle both LangChain and simple memory
            try:
                if hasattr(memory, 'chat_memory') and hasattr(memory.chat_memory, 'messages'):
                    message_count = len(memory.chat_memory.messages)
                else:
                    message_count = 0
            except:
                message_count = 0
            
            session_stats = {
                "session_id": session_id,
                "message_count": message_count,
                "created_at": session_data["created_at"],
                "last_accessed": session_data["last_accessed"],
                "age_minutes": (current_time - session_data["created_at"]) / 60,
                "inactive_minutes": (current_time - session_data["last_accessed"]) / 60
            }
            stats["sessions"].append(session_stats)
        
        return stats


def create_memory_manager(memory_type: str = "buffer_window") -> ConversationMemoryManager:
    """
    Factory function to create a memory manager.
    """
    return ConversationMemoryManager(memory_type=memory_type)


if __name__ == "__main__":
    # Test the memory manager
    print("Testing ConversationMemoryManager...")
    
    # Create manager
    manager = create_memory_manager("buffer_window")
    
    # Test conversation
    session_id = "test_session_1"
    manager.save_conversation(session_id, "Who is Per-Olof Arnäs?", "He's a logistics researcher...")
    manager.save_conversation(session_id, "What are his main research areas?", "Intermodal transport and digitalization...")
    
    # Get history in your existing format
    history = manager.get_conversation_history_for_state(session_id)
    print(f"History (your format): {history}")
    
    # Get context string for prompts
    context = manager.get_conversation_context_string(session_id)
    print(f"Context string: {context}")
    
    # Get stats
    stats = manager.get_memory_stats()
    print(f"Memory stats: {stats}")
    
    print("✅ Memory manager test completed!")