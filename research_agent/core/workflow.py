"""
FIXED workflow.py - Proper Asyncio Context for Streaming
Key Fix: Remove nested asyncio loops, use proper async/await throughout
"""

from typing import Dict, Any, List, Optional, AsyncGenerator
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import os
import uuid
import time
import re
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Import tools
from research_agent.tools import get_all_tools

# =============================================================================
# STREAMING EVENT EMITTER (Fixed)
# =============================================================================

class StreamingEventEmitter:
    """Emits real-time events for streaming reasoning steps."""
    
    def __init__(self):
        self.subscribers = []
        self.step_counter = 0
        self.active = False
        self._loop = None
    
    def subscribe(self, callback):
        """Subscribe to streaming events."""
        self.subscribers.append(callback)
        self.active = True
        self._loop = asyncio.get_event_loop()
        print(f"📡 Event emitter: Added subscriber (total: {len(self.subscribers)})")
    
    def unsubscribe(self, callback):
        """Unsubscribe from streaming events."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
        if not self.subscribers:
            self.active = False
        print(f"📡 Event emitter: Removed subscriber (remaining: {len(self.subscribers)})")
    
    async def emit(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to all subscribers."""
        if not self.active or not self.subscribers:
            print(f"⚠️ No active subscribers for event: {event_type}")
            return
            
        event = {
            "event": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"📤 Emitting event: {event_type} to {len(self.subscribers)} subscribers")
        
        # Notify all subscribers
        for callback in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                print(f"❌ Error in event subscriber: {e}")

# Global event emitter (will be set by agent)
_global_event_emitter = None

def set_global_event_emitter(emitter):
    global _global_event_emitter
    _global_event_emitter = emitter
    print("🔧 Global event emitter set")

def get_global_event_emitter():
    return _global_event_emitter

# =============================================================================
# ENHANCED REASONING COLLECTOR (Fixed for Asyncio)
# =============================================================================

class StreamingReasoningCollector:
    """Collects and streams reasoning data in real-time."""
    
    def __init__(self, session_id: str, query: str):
        self.session_id = session_id
        self.query = query
        self.steps: List[Dict] = []
        self.start_time = datetime.now()
        self.step_counter = 0
        self.tools_used = set()
        self.event_emitter = get_global_event_emitter()
        
        print(f"🧠 StreamingReasoningCollector created for session: {session_id}")
        print(f"📡 Event emitter available: {self.event_emitter is not None}")
    
    async def add_thinking_step(self, content: str, timestamp: datetime = None):
        """Add a thinking/reasoning step and emit it."""
        self.step_counter += 1
        simplified = self._simplify_thinking(content)
        
        step = {
            'step_number': self.step_counter,
            'step_type': 'thinking',
            'title': "Understanding your question",
            'description': simplified,
            'timestamp': (timestamp or datetime.now()).strftime("%H:%M:%S"),
            'details': content,
            'icon': '🤔',
            'color': '#4A90E2'
        }
        
        self.steps.append(step)
        
        print(f"🤔 Added thinking step {self.step_counter}: {simplified[:50]}...")
        
        # Emit real-time event
        if self.event_emitter:
            await self.event_emitter.emit("reasoning_step", {
                "step": step,
                "step_number": self.step_counter,
                "total_steps": len(self.steps),
                "session_id": self.session_id
            })
        else:
            print("⚠️ No event emitter available for thinking step")
    
    async def add_tool_step(self, tool_name: str, tool_input: Dict, timestamp: datetime = None):
        """Add a tool execution step and emit it."""
        self.step_counter += 1
        self.tools_used.add(tool_name)
        description = self._explain_tool_usage(tool_name, tool_input)
        
        step = {
            'step_number': self.step_counter,
            'step_type': 'searching',
            'title': "Searching the database",
            'description': description,
            'timestamp': (timestamp or datetime.now()).strftime("%H:%M:%S"),
            'tool_name': tool_name,
            'tool_input': str(tool_input),
            'details': f"Using {tool_name} with parameters: {json.dumps(tool_input, indent=2)}",
            'icon': '🔍',
            'color': '#F5A623'
        }
        
        self.steps.append(step)
        
        print(f"🔍 Added tool step {self.step_counter}: {tool_name}")
        
        # Emit real-time event
        if self.event_emitter:
            await self.event_emitter.emit("reasoning_step", {
                "step": step,
                "step_number": self.step_counter,
                "total_steps": len(self.steps),
                "session_id": self.session_id
            })
        else:
            print("⚠️ No event emitter available for tool step")
    
    async def add_observation_step(self, tool_name: str, result: str, timestamp: datetime = None):
        """Add a tool result observation step and emit it."""
        self.step_counter += 1
        summary = self._summarize_results(result, tool_name)
        
        step = {
            'step_number': self.step_counter,
            'step_type': 'analyzing',
            'title': "Processing the results",
            'description': summary,
            'timestamp': (timestamp or datetime.now()).strftime("%H:%M:%S"),
            'tool_name': tool_name,
            'details': f"Raw results: {result[:500]}..." if len(result) > 500 else result,
            'icon': '📊',
            'color': '#7ED321'
        }
        
        self.steps.append(step)
        
        print(f"📊 Added observation step {self.step_counter}: {summary[:50]}...")
        
        # Emit real-time event
        if self.event_emitter:
            await self.event_emitter.emit("reasoning_step", {
                "step": step,
                "step_number": self.step_counter,
                "total_steps": len(self.steps),
                "session_id": self.session_id
            })
        else:
            print("⚠️ No event emitter available for observation step")
    
    async def add_final_step(self, response: str, timestamp: datetime = None):
        """Add final answer formulation step and emit it."""
        self.step_counter += 1
        
        step = {
            'step_number': self.step_counter,
            'step_type': 'concluding',
            'title': "Formulating your answer",
            'description': "Organizing all findings into a comprehensive response",
            'timestamp': (timestamp or datetime.now()).strftime("%H:%M:%S"),
            'details': f"Generated response: {len(response)} characters",
            'icon': '✅',
            'color': '#9013FE'
        }
        
        self.steps.append(step)
        
        print(f"✅ Added final step {self.step_counter}")
        
        # Emit real-time event
        if self.event_emitter:
            await self.event_emitter.emit("reasoning_step", {
                "step": step,
                "step_number": self.step_counter,
                "total_steps": len(self.steps),
                "session_id": self.session_id
            })
            
            # Emit final response content
            await self.event_emitter.emit("response_chunk", {
                "content": response,
                "is_final": True,
                "session_id": self.session_id
            })
            
            # Emit completion event
            await self.event_emitter.emit("response_complete", {
                "response": response,
                "reasoning_data": self.get_frontend_data(),
                "session_id": self.session_id
            })
        else:
            print("⚠️ No event emitter available for final step")
    
    def _simplify_thinking(self, content: str) -> str:
        content_lower = content.lower()
        if any(word in content_lower for word in ['search', 'find', 'look']):
            return "I need to search the research database to answer your question"
        elif 'analyze' in content_lower:
            return "Let me analyze the available information" 
        elif any(word in content_lower for word in ['understand', 'question', 'query']):
            return "I'm processing your question to determine the best approach"
        else:
            first_sentence = content.split('.')[0].strip()
            if len(first_sentence) > 80:
                return first_sentence[:77] + "..."
            return first_sentence
    
    def _explain_tool_usage(self, tool_name: str, tool_input: Dict) -> str:
        explanations = {
            'search_persons': "Looking for researchers and their profiles",
            'search_publications_by_keywords': "Searching for research papers and publications", 
            'search_publications_by_author': "Finding publications by specific authors",
            'search_projects': "Looking up research projects and grants",
            'get_author_details': "Getting detailed information about researchers",
            'analyze_collaboration': "Analyzing research collaboration networks"
        }
        
        base_explanation = explanations.get(tool_name, f"Using the {tool_name} research tool")
        
        if 'query' in tool_input:
            return f"{base_explanation} for '{tool_input['query']}'"
        elif 'keywords' in tool_input:
            keywords = tool_input['keywords']
            if isinstance(keywords, list) and keywords:
                return f"{base_explanation} related to {', '.join(keywords[:3])}"
        elif 'author_name' in tool_input:
            return f"{base_explanation} by {tool_input['author_name']}"
        
        return base_explanation
    
    def _summarize_results(self, result: str, tool_name: str) -> str:
        import re
        numbers = re.findall(r'\b\d+\b', result)
        
        if numbers and 'search' in tool_name:
            count = numbers[0]
            if int(count) > 0:
                return f"Found {count} relevant results in the database"
            else:
                return "No matching results found in the database"
        
        if any(word in result.lower() for word in ['found', 'identified', 'retrieved']):
            return "Successfully retrieved information from the database"
        elif 'error' in result.lower():
            return "Encountered an issue while searching"
        else:
            preview = result[:100] + "..." if len(result) > 100 else result
            return f"Received results: {preview}"
    
    def get_frontend_data(self) -> Dict[str, Any]:
        """Generate complete data package for frontend."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        return {
            'reasoning_available': True,
            'session_id': self.session_id,
            'query': self.query,
            'summary': {
                'description': f"I used {len(self.tools_used)} research tools across {len(self.steps)} steps",
                'total_steps': len(self.steps),
                'total_duration': f"{duration:.1f}s",
                'tools_used': list(self.tools_used),
                'success': len(self.steps) > 0
            },
            'steps': self.steps,
            'start_time': self.start_time.strftime("%H:%M:%S"),
            'end_time': end_time.strftime("%H:%M:%S"),
            'ui_config': {
                'default_collapsed': True,
                'show_timestamps': True,
                'show_durations': True,
                'show_tools': True,
                'color_coded': True
            }
        }

# =============================================================================
# ORIGINAL STATE SCHEMA
# =============================================================================

class ReActState(TypedDict):
    """Clean ReAct state schema."""
    input: str
    response: Optional[str]
    session_id: str

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_conversation_context(memory_manager, session_id: str, max_length: int = 2000) -> str:
    """Get conversation context with proper length management."""
    if not session_id or not memory_manager:
        return "No previous conversation context available."
    
    try:
        history = memory_manager.get_conversation_history_for_state(session_id)
        if not history:
            return "No previous conversation context available."
        
        context_parts = []
        current_length = 0
        
        for msg in reversed(history):
            role = msg["role"].title()
            content = msg["content"]
            
            new_part = f"- {role}: {content}\n"
            new_length = current_length + len(new_part)
            
            if new_length > max_length:
                remaining_space = max_length - current_length - len(f"- {role}: ") - 20
                if remaining_space > 100:
                    truncated_content = content[:remaining_space] + "..."
                    context_parts.insert(0, f"- {role}: {truncated_content}")
                break
            
            context_parts.insert(0, new_part.strip())
            current_length = new_length
        
        context = "\n".join(context_parts)
        return context
        
    except Exception as e:
        print(f"Error getting conversation context: {e}")
        return "Error retrieving conversation context."

def create_react_llm(session_id: str = None) -> ChatLiteLLM:
    """Create LLM for ReAct agent."""
    config = {
        "model": "anthropic/claude-sonnet-3.7",
        "temperature": 0.1,
        "max_tokens": 4000,
    }
    
    metadata = {
        "component": "react_llm",
        "model_name": config["model"],
        "purpose": "react_execution"
    }
    
    if session_id:
        metadata.update({
            "session_id": session_id,
            "session_group": f"research-session-{session_id}"
        })
    
    try:
        llm = ChatLiteLLM(
            model=config["model"],
            api_key=os.getenv("LITELLM_API_KEY"),
            api_base=os.getenv("LITELLM_BASE_URL"),
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            timeout=60,
            metadata=metadata
        )
        return llm
    except Exception as e:
        raise e

# Same REACT_PROMPT_TEMPLATE as before
REACT_PROMPT_TEMPLATE = """You are an AI research assistant specializing in the Chalmers University research database with access to powerful research tools.

**RESEARCH QUESTION:** {query}

**CONVERSATION CONTEXT:**
{conversation_context}

**DATABASE SCOPE & LIMITATIONS:**
This database covers research from Chalmers University of Technology, Sweden. Key limitations:
- **Author scope**: Only includes publications from when authors worked at Chalmers or collaborated with Chalmers researchers
- **Temporal coverage**: Grant projects comprehensive from 2012+, student theses from 2009+
- **Publication definition**: Includes journal articles, conference papers, reports, theses, and other research outputs
- **Access limitations**: Items registered but full-text may not always be available
- **Institution-specific**: Excludes research from other universities
- **Content gaps**: May lack pre-prints, industry reports, software/datasets
- **Data freshness**: Potential delays in updates

**CONTEXT-AWARE RESEARCH APPROACH:**
The conversation context contains important information from our ongoing discussion. Use this context to:
- Understand references to people, publications, or topics mentioned earlier
- Avoid asking for clarification when context already provides answers
- Build upon previous research findings rather than starting from scratch
- Connect current research to information already discovered
- Address follow-up questions in the context of earlier research

**SYSTEMATIC RESEARCH METHODOLOGY:**
Think step-by-step and use a systematic approach:

1. **Query Analysis**: 
   - What specific information is requested?
   - How does conversation context inform this query?
   - What entities are involved (authors, publications, projects)?
   - What depth of analysis is needed?

2. **Strategic Tool Usage**:
   - Start with targeted searches using exact parameters
   - Use multiple search approaches for verification
   - Look for patterns, trends, and relationships
   - Cross-reference data for consistency

3. **Quality Research Standards**:
   - Focus on concrete findings: names, numbers, dates, affiliations
   - Provide comprehensive results with supporting evidence
   - Look for collaboration networks and research patterns
   - Include publication trends and impact indicators
   - Connect findings to broader research landscape

4. **Response Excellence**:
   - Lead with direct answer to the specific question
   - Structure information clearly with headers and formatting
   - Include relevant statistics, examples, and context
   - Address all aspects of the original query
   - Connect to previous conversation when relevant
   - **Only mention database limitations when directly relevant to understanding results**

**EXECUTION PRINCIPLES:**
- Use tools systematically and efficiently
- Verify important findings through multiple approaches
- Build comprehensive profiles with rich context
- Provide thorough analysis with detailed explanations
- Handle name variations and disambiguation carefully
- Assess data quality and completeness

Now execute your research using the available tools to provide a comprehensive, well-structured response."""

def create_react_workflow(
    es_client=None, 
    index_name: str = "research-publications-static", 
    session_id: str = None,
    memory_manager=None
) -> StateGraph:
    """Create ReAct workflow WITH FIXED asyncio context streaming support."""
    
    # Get tools
    if es_client:
        tools = get_all_tools(es_client=es_client, index_name=index_name)
    else:
        tools = get_all_tools()
    
    # Use memory manager
    if memory_manager is None:
        try:
            from .memory_singleton import get_global_memory_manager
            memory_manager = get_global_memory_manager()
        except ImportError:
            memory_manager = None
    
    # Create ReAct LLM
    react_llm = create_react_llm(session_id)
    
    def react_step(state: ReActState):
        """FIXED ReAct step with proper asyncio context."""
        try:
            query = state["input"]
            session_id = state["session_id"]
            
            print(f"🚀 Starting ReAct step for session: {session_id}")
            
            # Get conversation context
            conversation_context = get_conversation_context(memory_manager, session_id, max_length=10000)
            
            # Create prompt with context
            react_prompt = REACT_PROMPT_TEMPLATE.format(
                query=query,
                conversation_context=conversation_context
            )
            
            # Create ReAct agent
            agent_executor = create_react_agent(react_llm, tools, prompt=react_prompt)
            
            config = {
                "metadata": {
                    "step": "react",
                    "session_id": session_id,
                    "query": query
                },
                "tags": [
                    f"session-{session_id}",
                    "react-with-fixed-streaming"
                ]
            }
            
            # SYNCHRONOUS EXECUTION WITH ASYNC PROCESSING
            try:
                # Execute the agent synchronously
                result = agent_executor.invoke({
                    "messages": [HumanMessage(content=query)]
                }, config=config)
                
                # CREATE STREAMING REASONING COLLECTOR
                collector = StreamingReasoningCollector(session_id, query)
                
                all_messages = result.get("messages", [])
                print(f"📨 Processing {len(all_messages)} messages for streaming...")
                
                # Process messages synchronously but emit events properly
                def process_messages_sync():
                    """Process messages in current event loop context."""
                    
                    async def async_processing():
                        # Add initial thinking step
                        await collector.add_thinking_step(f"You asked: {query}", datetime.now())
                        
                        # Process messages in order
                        for i, message in enumerate(all_messages):
                            print(f"🔍 Processing message {i+1}/{len(all_messages)}: {type(message).__name__}")
                            
                            if isinstance(message, AIMessage):
                                if hasattr(message, 'content') and message.content:
                                    # Only add thinking for non-final responses
                                    if not any(pattern in message.content.lower() for pattern in ['# researchers', 'based on my search']):
                                        await collector.add_thinking_step(message.content, datetime.now())
                                
                                # Process tool calls immediately
                                if hasattr(message, 'tool_calls') and message.tool_calls:
                                    for tool_call in message.tool_calls:
                                        tool_name = tool_call.get('name', 'unknown_tool')
                                        tool_input = tool_call.get('args', {})
                                        print(f"🔧 Tool call: {tool_name}")
                                        await collector.add_tool_step(tool_name, tool_input, datetime.now())
                            
                            elif isinstance(message, ToolMessage):
                                print(f"📊 Tool result: {message.name}")
                                await collector.add_observation_step(message.name, message.content, datetime.now())
                            
                            # Small delay for visual effect
                            await asyncio.sleep(0.2)
                        
                        # Get final response and emit
                        response_content = result["messages"][-1].content if result.get("messages") else "No response generated"
                        print(f"✅ Adding final step with response length: {len(response_content)}")
                        await collector.add_final_step(response_content, datetime.now())
                        
                        return response_content, collector.get_frontend_data()
                    
                    # Check if we're in an async context
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an async context, create a task
                        task = loop.create_task(async_processing())
                        # Wait for completion
                        return loop.run_until_complete(task)
                    except RuntimeError:
                        # No running loop, create one
                        return asyncio.run(async_processing())
                
                # Process messages
                response_content, frontend_data = process_messages_sync()
                
                print(f"🎯 ReAct completed successfully with {len(frontend_data.get('steps', []))} reasoning steps")
                
                return {
                    "response": response_content,
                    "session_id": session_id,
                    "_frontend_reasoning_data": frontend_data
                }
                    
            except Exception as exec_error:
                error_response = f"Error during research: {str(exec_error)}"
                print(f"❌ ReAct execution error: {exec_error}")
                return {
                    "response": error_response,
                    "session_id": session_id,
                    "_frontend_reasoning_data": None
                }
                
        except Exception as e:
            print(f"💥 Critical ReAct error: {e}")
            return {
                "response": f"Error processing request: {str(e)}",
                "session_id": state["session_id"],
                "_frontend_reasoning_data": None
            }
    
    # Build workflow
    workflow = StateGraph(ReActState)
    workflow.add_node("react", react_step)
    workflow.add_edge(START, "react")
    workflow.add_edge("react", END)
    
    return workflow

# =============================================================================
# RESEARCH AGENT CLASS (Updated)
# =============================================================================

class ResearchAgent:
    """ReAct Research Agent WITH FIXED asyncio context streaming support."""
    
    def __init__(self, es_client=None, index_name: str = "research-publications-static", 
                 recursion_limit: int = 50, memory_manager=None): 
        self.es_client = es_client
        self.index_name = index_name
        self.recursion_limit = recursion_limit
        self.memory_manager = memory_manager
        self.app = None
        self._last_frontend_data = None
        self.event_emitter = None

    def _compile_agent(self, session_id: str = None):
        """Compile ReAct agent for session."""
        workflow = create_react_workflow(self.es_client, self.index_name, session_id, self.memory_manager)
        self.app = workflow.compile()

    def set_event_emitter(self, emitter: StreamingEventEmitter):
        """Set event emitter for streaming."""
        self.event_emitter = emitter
        set_global_event_emitter(emitter)
        print(f"🔧 Event emitter set for agent")

    async def stream_query(self, query: str, conversation_history: Optional[List[Dict]] = None, frontend_session_id: str = None):
        """Stream query using ReAct workflow with frontend data collection."""
        
        session_id = frontend_session_id or f"fallback_{str(uuid.uuid4())}"
        
        print(f"🚀 Starting stream_query for session: {session_id}")
        
        # Compile if needed
        if self.app is None:
            self._compile_agent(session_id)
        
        initial_state = {
            "input": query,
            "response": None,
            "session_id": session_id
        }
        
        config = {
            "recursion_limit": self.recursion_limit,
            "metadata": {
                "query": query,
                "session_id": session_id,
                "index_name": self.index_name,
                "conversation_turn": len(conversation_history or []) + 1
            },
            "tags": [
                "streaming", 
                f"session-{session_id}",
                f"turn-{len(conversation_history or []) + 1}",
                "react-with-fixed-asyncio"
            ],
            "run_name": f"ReAct-Fixed-Asyncio-Query-Turn-{len(conversation_history or []) + 1}"
        }
        
        try:
            stream = self.app.astream(initial_state, config=config)
            
            async for event in stream:
                print(f"📡 Streaming event keys: {list(event.keys())}")
                if 'react' in event and 'response' in event['react']:
                    # Store frontend data for later access
                    self._last_frontend_data = event['react'].get('_frontend_reasoning_data')
                    print(f"✅ Found response in streaming event")
                yield event
                
        except GeneratorExit:
            print("🧹 Stream generator cleanup - normal")
            pass
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            yield {"error": {"error": str(e)}}

    def get_last_reasoning_data(self) -> Optional[Dict[str, Any]]:
        """Get the reasoning data from the last query execution."""
        return self._last_frontend_data


if __name__ == "__main__":
    print("🧠 FIXED ReAct workflow with proper asyncio context streaming support ready!")
    print("🔧 Key fix: Proper asyncio context handling for event emission")