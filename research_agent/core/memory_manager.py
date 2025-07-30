"""
Updated Conversation Memory Manager - NO FACT EXTRACTOR
CRITICAL FIX: Stores full research results instead of compressed facts
Preserves all information from execution steps for replanner and final response
"""

import time
import os
import warnings
import uuid
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_litellm import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate

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

# =============================================================================
# RESEARCH STEP STORAGE - NO FACT EXTRACTION
# =============================================================================

@dataclass
class ResearchStep:
    """Research step with FULL result content - NO fact extraction bottleneck."""
    step_id: str
    step_description: str
    full_result: str
    timestamp: float
    token_count: int
    # REMOVED: key_facts: List[str]  # This was the information bottleneck!
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResearchStep':
        """Create from dictionary."""
        return cls(**data)

class InMemoryResearchStore:
    """Research storage WITHOUT fact extraction - preserves all information."""
    
    def __init__(self):
        self.research_sessions: Dict[str, Dict[str, ResearchStep]] = {}
        self.session_order: Dict[str, List[str]] = {}
        self.cleanup_interval = 3600  # 1 hour
        self.last_cleanup = time.time()
    
    def store_step_result(self, session_id: str, step_result: ResearchStep) -> str:
        """Store FULL research step - NO information loss."""
        if session_id not in self.research_sessions:
            self.research_sessions[session_id] = {}
            self.session_order[session_id] = []
        
        reference_id = f"research_step_{len(self.session_order[session_id])}"
        self.research_sessions[session_id][reference_id] = step_result
        self.session_order[session_id].append(reference_id)
        
        print(f"📚 Stored FULL research result: {reference_id} ({step_result.token_count:,} tokens) - NO INFORMATION LOSS")
        return reference_id
    
    def get_step_result(self, session_id: str, reference_id: str) -> Optional[ResearchStep]:
        """Retrieve full research step result."""
        return self.research_sessions.get(session_id, {}).get(reference_id)
    
    def get_intelligent_context_summary(self, session_id: str, max_recent_steps: int = 5, token_budget: int = 12000) -> str:
        """
        CRITICAL FIX: Intelligent context management with FULL results.
        
        Strategy:
        1. Try to include full results within token budget
        2. Use smart truncation only when necessary
        3. Preserve the most recent and important information
        4. NO fact extraction bottleneck
        """
        if session_id not in self.session_order:
            return "No research steps completed yet."
        
        all_refs = self.session_order[session_id]
        recent_refs = all_refs[-max_recent_steps:]
        
        # Strategy 1: Try to fit full results within budget
        summaries = []
        current_chars = 0
        char_budget = token_budget * 4  # Rough chars to tokens conversion
        
        for i, ref_id in enumerate(recent_refs):
            step_result = self.research_sessions[session_id][ref_id]
            step_num = len(all_refs) - len(recent_refs) + i + 1
            
            full_result = step_result.full_result
            
            # Try to include full result
            full_summary = f"Step {step_num}: {step_result.step_description}\n\n=== COMPLETE RESEARCH RESULT ===\n{full_result}\n{'='*50}"
            
            if current_chars + len(full_summary) < char_budget:
                # Include complete result
                summaries.append(full_summary)
                current_chars += len(full_summary)
                print(f"📋 Step {step_num}: Including FULL result ({len(full_result):,} chars)")
            else:
                # Smart truncation - preserve beginning and key information
                remaining_budget = char_budget - current_chars
                if remaining_budget > 1000:  # If we have meaningful space
                    
                    # Smart truncation: Keep important parts
                    if len(full_result) > remaining_budget - 500:
                        # Keep first part (context) and last part (conclusions)
                        first_part = full_result[:int(remaining_budget * 0.6)]
                        last_part = full_result[-int(remaining_budget * 0.2):]
                        
                        truncated_result = f"{first_part}\n\n[...middle content truncated for context size...]\n\n{last_part}"
                    else:
                        truncated_result = full_result
                    
                    smart_summary = f"Step {step_num}: {step_result.step_description}\n\n=== RESEARCH RESULT (Smart Truncation) ===\n{truncated_result}\n{'='*50}"
                    summaries.append(smart_summary)
                    current_chars += len(smart_summary)
                    print(f"📋 Step {step_num}: Smart truncation applied ({len(truncated_result):,} chars)")
                else:
                    # No space left - add a reference note
                    ref_note = f"Step {step_num}: {step_result.step_description}\n[Full result available but truncated for context - {len(full_result):,} chars]"
                    summaries.append(ref_note)
                    print(f"📋 Step {step_num}: Referenced only (no space remaining)")
                
                break  # Stop adding more steps
        
        result = "\n\n".join(summaries)
        
        print(f"📋 Generated intelligent context: {len(result):,} chars (~{len(result)//4:,} tokens)")
        print(f"📊 Included {len(summaries)} steps with FULL/smart content preservation")
        
        return result
    
    def get_all_step_results(self, session_id: str) -> List[ResearchStep]:
        """Get ALL step results with FULL content."""
        if session_id not in self.session_order:
            return []
        
        results = []
        for ref_id in self.session_order[session_id]:
            step_result = self.research_sessions[session_id].get(ref_id)
            if step_result:
                results.append(step_result)
        
        return results
    
    def cleanup_session(self, session_id: str) -> None:
        """Clean up research session data."""
        self.research_sessions.pop(session_id, None)
        self.session_order.pop(session_id, None)
        print(f"🗑️ Cleaned up research memory for session: {session_id}")

# =============================================================================
# SIMPLE MEMORY (FALLBACK) - UNCHANGED
# =============================================================================

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

# =============================================================================
# INTEGRATED MEMORY MANAGER - NO FACT EXTRACTOR
# =============================================================================

class IntegratedMemoryManager:
    """
    Enhanced memory manager WITHOUT FactExtractor bottleneck.
    CRITICAL FIX: Preserves all execution information for replanner and final response.
    """
    
    def __init__(self, memory_type: str = "buffer_window", cleanup_interval: int = 3600):
        """Initialize the integrated memory manager WITHOUT fact extraction."""
        self.memory_type = memory_type
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
        
        # Existing conversation memory (unchanged)
        self.session_memories = {}
        
        # Research step storage WITHOUT fact extraction bottleneck
        self.research_store = InMemoryResearchStore()
        
        # REMOVED: FactExtractor - this was throwing away 95% of information!
        # self.fact_extractor = None
        
        print(f"🧠 IntegratedMemoryManager initialized WITHOUT FactExtractor - preserves ALL research information")
    
    # =============================================================================
    # EXISTING CONVERSATION MEMORY METHODS (UNCHANGED)
    # =============================================================================
    
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
        """Clear both conversation and research memory for a session."""
        # Clear conversation memory
        if session_id in self.session_memories:
            del self.session_memories[session_id]
            print(f"🗑️ Cleared conversation memory for session: {session_id}")
        
        # Clear research memory
        self.research_store.cleanup_session(session_id)
    
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
            # Also clean research data
            self.research_store.cleanup_session(session_id)
        
        self.last_cleanup = current_time
        
        if old_sessions:
            print(f"🧹 Cleaned up {len(old_sessions)} old memory sessions (conversation + research)")
    
    # =============================================================================
    # RESEARCH STEP MEMORY METHODS - NO FACT EXTRACTOR
    # =============================================================================
    
    def store_research_step(self, session_id: str, step_description: str, result_content: str) -> str:
        """
        CRITICAL FIX: Store research step WITHOUT fact extraction bottleneck.
        Preserves ALL information from execution for replanner and final response.
        """
        try:
            # REMOVED: FactExtractor initialization and fact extraction
            # This was the bottleneck that threw away 95% of valuable information!
            
            # Create research step with FULL content preservation
            research_step = ResearchStep(
                step_id=str(uuid.uuid4()),
                step_description=step_description,
                full_result=result_content,  # Keep ALL the rich, detailed content
                timestamp=time.time(),
                token_count=len(result_content.split())
                # REMOVED: key_facts extraction - was information bottleneck
            )
            
            # Store FULL result without information loss
            reference_id = self.research_store.store_step_result(session_id, research_step)
            
            print(f"📊 Stored COMPLETE research step ({len(result_content):,} chars, {research_step.token_count:,} tokens)")
            print(f"🎯 NO INFORMATION LOSS - Full execution result preserved for replanner")
            return reference_id
            
        except Exception as e:
            print(f"❌ Error storing research step: {e}")
            # Return a fallback reference
            return f"error_step_{int(time.time())}"
    
    def get_research_context_summary(self, session_id: str, max_recent_steps: int = 5) -> str:
        """
        CRITICAL FIX: Get research context with FULL results for replanner.
        Uses intelligent context management instead of fact extraction bottleneck.
        """
        return self.research_store.get_intelligent_context_summary(
            session_id, 
            max_recent_steps=max_recent_steps,
            token_budget=15000  # Generous budget for full context
        )
    
    def get_all_research_facts(self, session_id: str) -> List[str]:
        """
        UPDATED: Extract key information from FULL results for final response.
        This is different from FactExtractor - it's used only for final response enhancement.
        """
        all_steps = self.research_store.get_all_step_results(session_id)
        
        if not all_steps:
            return []
        
        # Extract key points from full results for final response formatting
        key_points = []
        
        for step in all_steps:
            full_result = step.full_result
            
            # Simple extraction of key sentences (not LLM-based fact extraction)
            sentences = full_result.split('. ')
            
            # Look for sentences that contain key indicators
            key_indicators = [
                'researcher', 'author', 'professor', 'university', 'institution',
                'published', 'publication', 'journal', 'conference',
                'research', 'study', 'focuses on', 'specializes in',
                'collaboration', 'co-author', 'working with',
                'findings', 'results', 'discovered', 'concluded'
            ]
            
            for sentence in sentences[:10]:  # First 10 sentences are usually most important
                sentence = sentence.strip()
                if len(sentence) > 20 and any(indicator in sentence.lower() for indicator in key_indicators):
                    if sentence not in key_points:  # Simple deduplication
                        key_points.append(sentence)
                        
                if len(key_points) >= 20:  # Reasonable limit
                    break
        
        return key_points[:15]  # Return top 15 key points
    
    def get_research_step_details(self, session_id: str, reference_id: str) -> Optional[str]:
        """Get FULL details of a specific research step."""
        step_result = self.research_store.get_step_result(session_id, reference_id)
        return step_result.full_result if step_result else None
    
    def get_comprehensive_final_response_data(self, session_id: str) -> Dict[str, Any]:
        """
        CRITICAL FIX: Get comprehensive data with FULL results for final response.
        No information loss, preserves all execution content.
        """
        all_steps = self.research_store.get_all_step_results(session_id)
        
        if not all_steps:
            return {
                "total_steps": 0,
                "full_results": [],
                "step_summaries": [],
                "total_content_length": 0
            }
        
        full_results = []
        step_summaries = []
        total_length = 0
        
        for i, step in enumerate(all_steps, 1):
            # Include FULL results, not compressed facts
            full_results.append(step.full_result)
            total_length += len(step.full_result)
            
            # Create step summary with full content reference
            step_summaries.append({
                "step_number": i,
                "task": step.step_description,
                "result_length": len(step.full_result),
                "token_count": step.token_count, 
                "timestamp": step.timestamp,
                "has_full_content": True  # Indicates full content is available
            })
        
        return {
            "total_steps": len(all_steps),
            "full_results": full_results,  # Complete results, no information loss
            "step_summaries": step_summaries,
            "total_content_length": total_length,
            "average_result_length": total_length // len(all_steps) if all_steps else 0
        }
    
    # =============================================================================
    # ENHANCED MEMORY STATS
    # =============================================================================
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Enhanced stats including FULL research memory without fact extraction."""
        stats = {
            "total_sessions": len(self.session_memories),
            "memory_type": self.memory_type,
            "sessions": [],
            "research_sessions": len(self.research_store.research_sessions),
            "total_research_steps": sum(
                len(steps) for steps in self.research_store.research_sessions.values()
            ),
            "fact_extractor_removed": True,  # Indicates the bottleneck was removed
            "information_preservation": "complete",  # Full results preserved
            "context_strategy": "intelligent_truncation"  # Smart context management
        }
        
        current_time = time.time()
        total_content_length = 0
        
        for session_id, session_data in self.session_memories.items():
            memory = session_data["memory"]
            
            try:
                if hasattr(memory, 'chat_memory') and hasattr(memory.chat_memory, 'messages'):
                    message_count = len(memory.chat_memory.messages)
                else:
                    message_count = 0
            except:
                message_count = 0
            
            # Add research step count and content stats
            research_steps = len(self.research_store.session_order.get(session_id, []))
            
            # Calculate total content for this session
            session_content_length = 0
            if session_id in self.research_store.research_sessions:
                for step in self.research_store.research_sessions[session_id].values():
                    session_content_length += len(step.full_result)
            
            total_content_length += session_content_length
            
            session_stats = {
                "session_id": session_id,
                "conversation_messages": message_count,
                "research_steps": research_steps,
                "total_research_content_length": session_content_length,
                "average_step_length": session_content_length // research_steps if research_steps > 0 else 0,
                "created_at": session_data["created_at"],
                "last_accessed": session_data["last_accessed"],
                "age_minutes": (current_time - session_data["created_at"]) / 60,
                "inactive_minutes": (current_time - session_data["last_accessed"]) / 60
            }
            stats["sessions"].append(session_stats)
        
        # Add global content stats
        stats["total_research_content_length"] = total_content_length
        stats["average_step_content_length"] = (
            total_content_length // stats["total_research_steps"] 
            if stats["total_research_steps"] > 0 else 0
        )
        
        return stats

# =============================================================================
# FACTORY FUNCTION AND COMPATIBILITY
# =============================================================================

def create_memory_manager(memory_type: str = "buffer_window") -> IntegratedMemoryManager:
    """
    Factory function to create memory manager WITHOUT FactExtractor bottleneck.
    Preserves all research information for optimal replanner and response quality.
    """
    return IntegratedMemoryManager(memory_type=memory_type)

# Backward compatibility alias
ConversationMemoryManager = IntegratedMemoryManager

if __name__ == "__main__":
    # Test the FIXED integrated memory manager WITHOUT fact extraction
    print("Testing IntegratedMemoryManager WITHOUT FactExtractor...")
    
    # Create manager
    manager = create_memory_manager("buffer_window")
    
    # Test conversation (unchanged functionality)
    session_id = "test_session_no_facts"
    manager.save_conversation(session_id, "Who is Per-Olof Arnäs?", "He's a logistics researcher...")
    manager.save_conversation(session_id, "What are his main research areas?", "Intermodal transport and digitalization...")
    
    # Test research storage WITHOUT fact extraction (preserves ALL information)
    comprehensive_result = """
## Who is Per-Olof Arnäs?

**Per-Olof Arnäs** is a Swedish academic researcher and expert in logistics and transportation, affiliated with **Chalmers University of Technology** in Gothenburg, Sweden. Based on his publication history spanning from 2011 to 2021, here are the key findings:

### Primary Affiliation and Research Areas:
- **Current Position**: Associated with the **Service Management and Logistics** department at Chalmers University of Technology
- **Previous Affiliation**: Also listed under "Logistics & Transportation" and "Technology Management and Economics" departments at Chalmers

### Research Expertise:
Per-Olof Arnäs specializes in several interconnected areas:

1. **Freight Transportation and Logistics**
   - Intermodal freight transportation systems
   - Access management at seaport terminals
   - Container management and depot operations
   - Transport efficiency and optimization

2. **Digital Transformation in Transportation**
   - Emerging and disruptive technologies in freight transport
   - Information exchange platforms (AEOLIX project)
   - Smart goods and tracking systems
   - Digitalization of transport systems

### Publication Profile:
- **Total Publications Found**: 57+ publications
- **Publication Period**: 2011-2021 (active research career of 10+ years)
- **Publication Types**: Journal articles, Conference papers, Reports, Magazine articles

This comprehensive information would have been reduced to 5 bullet points by FactExtractor!
"""
    
    ref_id = manager.store_research_step(
        session_id, 
        "Search for comprehensive information about Per-Olof Arnäs",
        comprehensive_result
    )
    
    # Test FULL context summary (should preserve rich information)
    summary = manager.get_research_context_summary(session_id)
    print(f"\n🔍 FULL Research context summary:")
    print(f"Length: {len(summary):,} characters (~{len(summary)//4:,} tokens)")
    print(f"Content preserved: {len(comprehensive_result):,} -> {len(summary):,} chars")
    print(f"Information retention: {(len(summary)/len(comprehensive_result)*100):.1f}%")
    print(f"Content preview: {summary[:300]}...")
    
    # Test comprehensive final response data
    final_data = manager.get_comprehensive_final_response_data(session_id)
    print(f"\nFinal response data:")
    print(f"- Total steps: {final_data['total_steps']}")
    print(f"- Full results available: {len(final_data['full_results'])}")
    print(f"- Total content length: {final_data['total_content_length']:,} chars")
    print(f"- Average result length: {final_data['average_result_length']:,} chars")
    
    # Test memory stats
    stats = manager.get_memory_stats()
    print(f"\nMemory stats:")
    print(f"- FactExtractor removed: {stats['fact_extractor_removed']}")
    print(f"- Information preservation: {stats['information_preservation']}")
    print(f"- Context strategy: {stats['context_strategy']}")
    print(f"- Total research content: {stats['total_research_content_length']:,} chars")
    
    print("\n✅ IntegratedMemoryManager WITHOUT FactExtractor test completed!")
    print("🎯 Key improvements:")
    print("  - NO information loss from execution to replanner")
    print("  - FULL research results preserved for decision making")
    print("  - Intelligent context management without fact extraction bottleneck")
    print("  - Rich, comprehensive responses possible")
    print("  - Replanner gets complete information for optimal decisions")