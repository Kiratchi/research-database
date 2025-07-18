"""
Fast-path workflow for conversational queries.

This module provides a lightweight, fast response path for queries that
don't require database tools, focusing on conversational interactions.
"""

import time
from typing import Dict, List, Optional, Any, AsyncIterator
from langchain_litellm import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()


class FastPathResponse(BaseModel):
    """Response from fast-path workflow."""
    response: str = Field(description="The conversational response")
    response_time: float = Field(description="Time taken to generate response")
    escalate: bool = Field(description="Whether to escalate to full workflow")
    escalation_reason: str = Field(description="Reason for escalation if needed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConversationalWorkflow:
    """
    Fast-path workflow for conversational queries.
    
    This workflow bypasses the plan-and-execute pattern for simple
    conversational queries that don't require database access.
    """
    
    def __init__(self):
        """Initialize the conversational workflow with LangChain Memory."""
        try:
            self.llm = ChatLiteLLM(
                model="anthropic/claude-sonet-3.7",  # Fast model for conversations
                api_key=os.getenv("LITELLM_API_KEY"),
                api_base=os.getenv("LITELLM_BASE_URL"),
                temperature=0.3  # Slightly more creative for conversations
            )
            
            # Initialize LangChain memory
            self.memory = ConversationBufferMemory(
                memory_key="history",
                return_messages=True,
                output_key="response"
            )
            
            # Create conversation chain with memory
            self.conversation_chain = ConversationChain(
                llm=self.llm,
                memory=self.memory,
                verbose=False
            )
            
            # Override the conversation chain's prompt template
            self.conversation_chain.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful research assistant for a publications database system.

You are currently in conversational mode - responding to queries that don't require database searches.

Your role:
- Provide friendly, helpful conversational responses
- Be concise but warm and professional
- If the user asks about research topics that would require database access, politely indicate that you'd need to search the database
- Keep responses short and to the point
- Maintain context from the conversation history

Important guidelines:
- If a user asks about specific authors, papers, or publications, suggest they ask a research question
- If they ask for data or statistics, indicate you'd need to search the database
- Be helpful but don't make up information about publications or authors
- If you're unsure whether a query needs database access, err on the side of suggesting a database search

If at any point you determine the query actually needs database access, respond with: "ESCALATE: [reason]" at the start of your response."""),
                ("placeholder", "{history}"),
                ("human", "{input}")
            ])
            
        except Exception as e:
            print(f"Warning: Failed to initialize conversational workflow: {e}")
            self.llm = None
            self.conversation_prompt = None
            self.conversation_chain = None
    
    def process_query(self, query: str, conversation_history: Optional[List[Dict]] = None) -> FastPathResponse:
        """
        Process a conversational query using the fast path with LangChain Memory.
        
        Args:
            query: The user query
            conversation_history: Recent conversation history (used for initialization)
            
        Returns:
            FastPathResponse with the conversational response
        """
        start_time = time.time()
        
        # Handle case where LLM is not available
        if not self.conversation_chain:
            return FastPathResponse(
                response="I apologize, but I'm having trouble accessing the conversational system. Let me try to help you with a more detailed search.",
                response_time=time.time() - start_time,
                escalate=True,
                escalation_reason="Conversational system unavailable",
                metadata={
                    "workflow_type": "fast_path",
                    "escalated": True,
                    "error": "LLM not available"
                }
            )
        
        try:
            # Initialize memory with conversation history if provided
            if conversation_history and not self.memory.chat_memory.messages:
                self._initialize_memory_from_history(conversation_history)
            
            # Generate response using conversation chain with memory
            response = self.conversation_chain.invoke({"input": query})
            
            response_time = time.time() - start_time
            
            # Extract response text (LangChain returns dict with 'response' key)
            response_text = response.get('response', '') if isinstance(response, dict) else str(response)
            
            # Check if response indicates escalation is needed
            if response_text.startswith("ESCALATE:"):
                escalation_reason = response_text.split("ESCALATE:")[1].strip()
                actual_response = "I'd be happy to help you with that! Let me search the publications database for you."
                
                return FastPathResponse(
                    response=actual_response,
                    response_time=response_time,
                    escalate=True,
                    escalation_reason=escalation_reason,
                    metadata={
                        "workflow_type": "fast_path",
                        "escalated": True,
                        "original_response": response_text
                    }
                )
            
            return FastPathResponse(
                response=response_text,
                response_time=response_time,
                escalate=False,
                escalation_reason="",
                metadata={
                    "workflow_type": "fast_path",
                    "escalated": False,
                    "memory_messages_count": len(self.memory.chat_memory.messages)
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return FastPathResponse(
                response="I apologize, but I encountered an error. Let me try to help you with a more detailed search.",
                response_time=response_time,
                escalate=True,
                escalation_reason=f"Error in fast path: {str(e)}",
                metadata={
                    "workflow_type": "fast_path",
                    "escalated": True,
                    "error": str(e)
                }
            )
    
    def _initialize_memory_from_history(self, history: List[Dict]):
        """Initialize LangChain memory from conversation history."""
        if not history:
            return
        
        # Take last 10 messages for context (5 turns)
        recent_history = history[-10:]
        
        for msg in recent_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "..."
            
            # Add to memory based on role
            if role == "user":
                self.memory.chat_memory.add_user_message(content)
            elif role == "assistant":
                self.memory.chat_memory.add_ai_message(content)
    
    def get_conversation_memory(self) -> ConversationBufferMemory:
        """Get the current conversation memory instance."""
        return self.memory
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.memory.clear()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of the current memory state."""
        messages = self.memory.chat_memory.messages
        return {
            "total_messages": len(messages),
            "user_messages": len([m for m in messages if isinstance(m, HumanMessage)]),
            "ai_messages": len([m for m in messages if isinstance(m, AIMessage)]),
            "memory_buffer": self.memory.buffer if hasattr(self.memory, 'buffer') else None
        }
    
    async def stream_query(self, query: str, conversation_history: Optional[List[Dict]] = None) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream a conversational query response with LangChain Memory.
        
        Args:
            query: The user query
            conversation_history: Recent conversation history (used for initialization)
            
        Yields:
            Streaming events for the conversational response
        """
        start_time = time.time()
        
        # Yield initial status
        yield {
            "type": "status",
            "content": "💬 Responding...",
            "timestamp": time.time()
        }
        
        # Handle case where LLM is not available
        if not self.conversation_chain:
            yield {
                "type": "escalation",
                "content": {
                    "reason": "Conversational system unavailable",
                    "message": "Let me search the database for you...",
                    "response_time": time.time() - start_time
                },
                "timestamp": time.time()
            }
            return
        
        try:
            # Initialize memory with conversation history if provided
            if conversation_history and not self.memory.chat_memory.messages:
                self._initialize_memory_from_history(conversation_history)
            
            # Stream the response using conversation chain with memory
            response_content = ""
            async for chunk in self.conversation_chain.astream({"input": query}):
                # Handle different chunk formats from LangChain
                chunk_text = ""
                if isinstance(chunk, dict):
                    chunk_text = chunk.get('response', '')
                elif hasattr(chunk, 'content'):
                    chunk_text = chunk.content
                else:
                    chunk_text = str(chunk)
                
                if chunk_text:
                    response_content += chunk_text
                    yield {
                        "type": "response_chunk",
                        "content": chunk_text,
                        "timestamp": time.time()
                    }
            
            response_time = time.time() - start_time
            
            # Check for escalation
            if response_content.startswith("ESCALATE:"):
                escalation_reason = response_content.split("ESCALATE:")[1].strip()
                
                yield {
                    "type": "escalation",
                    "content": {
                        "reason": escalation_reason,
                        "message": "Let me search the database for you...",
                        "response_time": response_time
                    },
                    "timestamp": time.time()
                }
            else:
                yield {
                    "type": "final",
                    "content": {
                        "response": response_content,
                        "response_time": response_time,
                        "escalate": False,
                        "metadata": {
                            "workflow_type": "fast_path",
                            "escalated": False,
                            "memory_messages_count": len(self.memory.chat_memory.messages)
                        }
                    },
                    "timestamp": time.time()
                }
                
        except Exception as e:
            response_time = time.time() - start_time
            yield {
                "type": "error",
                "content": {
                    "error": str(e),
                    "response_time": response_time,
                    "escalate": True,
                    "escalation_reason": f"Error in fast path: {str(e)}"
                },
                "timestamp": time.time()
            }


# Convenience functions
def process_conversational_query(query: str, conversation_history: Optional[List[Dict]] = None) -> FastPathResponse:
    """Process a conversational query using the fast path with LangChain Memory."""
    workflow = ConversationalWorkflow()
    return workflow.process_query(query, conversation_history)


async def stream_conversational_query(query: str, conversation_history: Optional[List[Dict]] = None) -> AsyncIterator[Dict[str, Any]]:
    """Stream a conversational query response with LangChain Memory."""
    workflow = ConversationalWorkflow()
    async for event in workflow.stream_query(query, conversation_history):
        yield event


def create_conversational_workflow() -> ConversationalWorkflow:
    """Create a new conversational workflow instance with fresh memory."""
    return ConversationalWorkflow()


def get_workflow_memory_summary(workflow: ConversationalWorkflow) -> Dict[str, Any]:
    """Get memory summary from a conversational workflow instance."""
    return workflow.get_memory_summary()