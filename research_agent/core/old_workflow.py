"""
Complete Updated workflow.py - WITH Frontend Reasoning Data Output
Captures reasoning steps and prints JSON data ready for frontend consumption
"""

from typing import Dict, Any, List, Optional
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
from datetime import datetime
from dotenv import load_dotenv

# Import tools
from research_agent.tools import get_all_tools

# =============================================================================
# FRONTEND DATA STRUCTURES
# =============================================================================

class FrontendReasoningStep:
    """Reasoning step formatted for frontend consumption."""
    def __init__(self, step_number: int, step_type: str, title: str, description: str, 
                 timestamp: str = None, duration: str = None, tool_name: str = None, 
                 tool_input: str = None, details: str = None):
        self.step_number = step_number
        self.step_type = step_type  # 'thinking', 'searching', 'analyzing', 'concluding'
        self.title = title
        self.description = description
        self.timestamp = timestamp or datetime.now().strftime("%H:%M:%S")
        self.duration = duration
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.details = details
        
        # UI properties
        self.icon = {
            'thinking': '🤔',
            'searching': '🔍', 
            'analyzing': '📊',
            'concluding': '✅',
            'error': '❌'
        }.get(step_type, '📝')
        
        self.color = {
            'thinking': '#4A90E2',
            'searching': '#F5A623',
            'analyzing': '#7ED321', 
            'concluding': '#9013FE',
            'error': '#D0021B'
        }.get(step_type, '#666666')
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'step_number': self.step_number,
            'step_type': self.step_type,
            'title': self.title,
            'description': self.description,
            'timestamp': self.timestamp,
            'duration': self.duration,
            'tool_name': self.tool_name,
            'tool_input': self.tool_input,
            'details': self.details,
            'icon': self.icon,
            'color': self.color
        }

class FrontendReasoningCollector:
    """Collects and formats reasoning data for frontend."""
    
    def __init__(self, session_id: str, query: str):
        self.session_id = session_id
        self.query = query
        self.steps: List[FrontendReasoningStep] = []
        self.start_time = datetime.now()
        self.step_counter = 0
        self.tools_used = set()
    
    def add_thinking_step(self, content: str, timestamp: datetime = None):
        """Add a thinking/reasoning step."""
        self.step_counter += 1
        
        # Simplify thinking content for users
        simplified = self._simplify_thinking(content)
        
        step = FrontendReasoningStep(
            step_number=self.step_counter,
            step_type='thinking',
            title=f"Understanding your question",
            description=simplified,
            timestamp=(timestamp or datetime.now()).strftime("%H:%M:%S"),
            details=content
        )
        
        self.steps.append(step)
    
    def add_tool_step(self, tool_name: str, tool_input: Dict, timestamp: datetime = None):
        """Add a tool execution step."""
        self.step_counter += 1
        self.tools_used.add(tool_name)
        
        # Create user-friendly description
        description = self._explain_tool_usage(tool_name, tool_input)
        
        step = FrontendReasoningStep(
            step_number=self.step_counter,
            step_type='searching',
            title=f"Searching the database",
            description=description,
            timestamp=(timestamp or datetime.now()).strftime("%H:%M:%S"),
            tool_name=tool_name,
            tool_input=str(tool_input),
            details=f"Using {tool_name} with parameters: {json.dumps(tool_input, indent=2)}"
        )
        
        self.steps.append(step)
    
    def add_observation_step(self, tool_name: str, result: str, timestamp: datetime = None):
        """Add a tool result observation step."""
        self.step_counter += 1
        
        # Analyze and summarize results
        summary = self._summarize_results(result, tool_name)
        
        step = FrontendReasoningStep(
            step_number=self.step_counter,
            step_type='analyzing',
            title=f"Processing the results",
            description=summary,
            timestamp=(timestamp or datetime.now()).strftime("%H:%M:%S"),
            tool_name=tool_name,
            details=f"Raw results: {result[:500]}..." if len(result) > 500 else result
        )
        
        self.steps.append(step)
    
    def add_final_step(self, response: str, timestamp: datetime = None):
        """Add final answer formulation step."""
        self.step_counter += 1
        
        step = FrontendReasoningStep(
            step_number=self.step_counter,
            step_type='concluding',
            title=f"Formulating your answer",
            description="Organizing all findings into a comprehensive response",
            timestamp=(timestamp or datetime.now()).strftime("%H:%M:%S"),
            details=f"Generated response: {len(response)} characters"
        )
        
        self.steps.append(step)
    
    def _simplify_thinking(self, content: str) -> str:
        """Simplify AI thinking for user consumption."""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['search', 'find', 'look']):
            return "I need to search the research database to answer your question"
        elif 'analyze' in content_lower:
            return "Let me analyze the available information" 
        elif any(word in content_lower for word in ['understand', 'question', 'query']):
            return "I'm processing your question to determine the best approach"
        else:
            # Take first sentence and simplify
            first_sentence = content.split('.')[0].strip()
            if len(first_sentence) > 80:
                return first_sentence[:77] + "..."
            return first_sentence
    
    def _explain_tool_usage(self, tool_name: str, tool_input: Dict) -> str:
        """Explain tool usage in user-friendly terms."""
        
        explanations = {
            'search_persons': "Looking for researchers and their profiles",
            'search_publications_by_keywords': "Searching for research papers and publications", 
            'search_publications_by_author': "Finding publications by specific authors",
            'search_projects': "Looking up research projects and grants",
            'get_author_details': "Getting detailed information about researchers",
            'analyze_collaboration': "Analyzing research collaboration networks"
        }
        
        base_explanation = explanations.get(tool_name, f"Using the {tool_name} research tool")
        
        # Add specific search details
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
        """Summarize tool results for users."""
        # Extract numbers from results
        import re
        numbers = re.findall(r'\b\d+\b', result)
        
        if numbers and 'search' in tool_name:
            count = numbers[0]
            if int(count) > 0:
                return f"Found {count} relevant results in the database"
            else:
                return "No matching results found in the database"
        
        # Look for success indicators
        if any(word in result.lower() for word in ['found', 'identified', 'retrieved']):
            return "Successfully retrieved information from the database"
        elif 'error' in result.lower():
            return "Encountered an issue while searching"
        else:
            # Provide a preview
            preview = result[:100] + "..." if len(result) > 100 else result
            return f"Received results: {preview}"
    
    def get_frontend_data(self) -> Dict[str, Any]:
        """Generate complete data package for frontend."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        return {
            # Core data
            'reasoning_available': True,
            'session_id': self.session_id,
            'query': self.query,
            
            # Summary
            'summary': {
                'description': self._generate_summary(),
                'total_steps': len(self.steps),
                'total_duration': f"{duration:.1f}s",
                'tools_used': list(self.tools_used),
                'success': self._determine_success()
            },
            
            # Detailed steps
            'steps': [step.to_dict() for step in self.steps],
            
            # Timestamps
            'start_time': self.start_time.strftime("%H:%M:%S"),
            'end_time': end_time.strftime("%H:%M:%S"),
            
            # UI configuration
            'ui_config': {
                'default_collapsed': True,
                'show_timestamps': True,
                'show_durations': True,
                'show_tools': True,
                'color_coded': True
            }
        }
    
    def _generate_summary(self) -> str:
        """Generate a user-friendly summary."""
        if not self.steps:
            return "Processing completed"
        
        tool_count = len(self.tools_used)
        if tool_count == 1:
            return f"I searched the database using {list(self.tools_used)[0]} and processed {len(self.steps)} steps"
        elif tool_count > 1:
            return f"I used {tool_count} research tools across {len(self.steps)} steps to find comprehensive information"
        else:
            return f"I processed your question through {len(self.steps)} reasoning steps"
    
    def _determine_success(self) -> bool:
        """Determine if the reasoning was successful."""
        # Simple heuristic: if we have steps and no error steps
        return len(self.steps) > 0 and not any(step.step_type == 'error' for step in self.steps)

# =============================================================================
# TERMINAL REASONING LOGGING (Enhanced with Frontend Data Collection)
# =============================================================================

# Add colored output for better terminal display
try:
    from colorama import init, Fore, Back, Style
    init()
    COLORS_AVAILABLE = True
except ImportError:
    print("💡 Tip: Install colorama for colored terminal output: pip install colorama")
    COLORS_AVAILABLE = False
    # Fallback empty classes
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
    class Back:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ""

def log_reasoning_to_terminal(content: str, message_type: str = "ai", session_id: str = "unknown", collector: FrontendReasoningCollector = None):
    """Extract and log reasoning steps to terminal with colors and emojis."""
    if not content or len(content.strip()) < 10:
        return
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Color scheme
    if COLORS_AVAILABLE:
        thought_color = Fore.BLUE + Style.BRIGHT
        action_color = Fore.GREEN + Style.BRIGHT
        observation_color = Fore.YELLOW + Style.BRIGHT
        final_color = Fore.MAGENTA + Style.BRIGHT
        reset = Style.RESET_ALL
    else:
        thought_color = action_color = observation_color = final_color = reset = ""
    
    # Look for thought patterns
    thought_patterns = [
        r"Thought:\s*([^.\n]+(?:\.[^.\n]*)*?)(?=\s*(?:Action:|$))",
        r"I (?:should|need to|will|must)\s+([^.\n]+(?:\.[^.\n]*)*?)(?=\s*(?:Action:|Let me|$))",
        r"Let me\s+([^.\n]+(?:\.[^.\n]*)*?)(?=\s*(?:Action:|$))",
        r"(?:First|Next|Now),?\s*I\s+(?:need to|should|will)\s+([^.\n]+(?:\.[^.\n]*)*?)(?=\s*(?:Action:|$))"
    ]
    
    found_thought = False
    for pattern in thought_patterns:
        thought_match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if thought_match and not found_thought:
            thought = thought_match.group(1).strip()
            # Better filtering - avoid partial matches
            if len(thought) > 20 and not thought.endswith(('to', 'the', 'a', 'an', 'in', 'on', 'for')):
                print(f"\n{thought_color}{'='*60}")
                print(f"🤔 REASONING STEP [{timestamp}] - Session: {session_id}")
                print(f"{'='*60}{reset}")
                print(f"{thought_color}{thought}{reset}")
                found_thought = True
                
                # Add to frontend collector
                if collector:
                    collector.add_thinking_step(thought, datetime.now())
                break
    
    # Look for final answer patterns
    final_patterns = [
        r"^(?:Based on|According to|The (?:research|analysis|results|search) (?:shows?|indicates?|found)).*",
        r"^(?:I found|There are|Here are).*(?:\d+|several|many).*(?:researchers?|publications?|results?).*",
        r"^#\s+[^#\n]*(?:researchers?|publications?|results?)"  # Markdown headers about results
    ]
    
    found_final = False
    for pattern in final_patterns:
        final_match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if final_match and not found_final:
            # This looks like a final answer
            preview = content[:300] + "..." if len(content) > 300 else content
            print(f"\n{final_color}{'='*60}")
            print(f"✅ GENERATING FINAL ANSWER [{timestamp}] - Session: {session_id}")
            print(f"{'='*60}{reset}")
            print(f"{final_color}{preview}{reset}")
            found_final = True
            
            # Add to frontend collector
            if collector:
                collector.add_final_step(content, datetime.now())
            break

def log_tool_call_to_terminal(tool_name: str, tool_input: Dict[str, Any], session_id: str = "unknown", collector: FrontendReasoningCollector = None):
    """Log tool calls to terminal."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if COLORS_AVAILABLE:
        color = Fore.CYAN + Style.BRIGHT
        reset = Style.RESET_ALL
    else:
        color = reset = ""
    
    print(f"\n{color}{'='*60}")
    print(f"⚡ TOOL EXECUTION [{timestamp}] - Session: {session_id}")
    print(f"{'='*60}{reset}")
    print(f"{color}🔧 Tool: {tool_name}{reset}")
    
    # Enhanced input formatting
    if tool_input:
        if isinstance(tool_input, dict):
            # Format dictionary nicely
            formatted_input = []
            for key, value in tool_input.items():
                if isinstance(value, str) and len(value) > 50:
                    value = value[:47] + "..."
                formatted_input.append(f"{key}: {value}")
            input_str = ", ".join(formatted_input)
        else:
            input_str = str(tool_input)
        
        if len(input_str) > 200:
            input_str = input_str[:200] + "..."
        
        print(f"{color}📥 Parameters: {input_str}{reset}")
    
    # Add to frontend collector
    if collector:
        collector.add_tool_step(tool_name, tool_input, datetime.now())

def log_tool_result_to_terminal(tool_name: str, result: str, session_id: str = "unknown", collector: FrontendReasoningCollector = None):
    """Log tool results to terminal."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if COLORS_AVAILABLE:
        color = Fore.YELLOW + Style.BRIGHT
        reset = Style.RESET_ALL
    else:
        color = reset = ""
    
    # Truncate long results for terminal
    display_result = result[:200] + "..." if len(result) > 200 else result
    
    print(f"\n{color}{'='*60}")
    print(f"👀 TOOL RESULT [{timestamp}] - Session: {session_id}")
    print(f"{'='*60}{reset}")
    print(f"{color}{display_result}{reset}")
    
    # Add to frontend collector
    if collector:
        collector.add_observation_step(tool_name, result, datetime.now())

def log_session_start(query: str, session_id: str, collector: FrontendReasoningCollector = None):
    """Log the start of a reasoning session."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if COLORS_AVAILABLE:
        color = Fore.WHITE + Back.BLUE + Style.BRIGHT
        reset = Style.RESET_ALL
    else:
        color = reset = ""
    
    print(f"\n{color}{'='*80}")
    print(f"🚀 NEW REASONING SESSION STARTED [{timestamp}]")
    print(f"Session ID: {session_id}")
    print(f"{'='*80}{reset}")
    print(f"❓ QUERY: {query}")
    print(f"{'='*80}")

def log_session_end(response: str, session_id: str, collector: FrontendReasoningCollector = None):
    """Log the end of a reasoning session with summary."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if COLORS_AVAILABLE:
        color = Fore.WHITE + Back.MAGENTA + Style.BRIGHT
        reset = Style.RESET_ALL
    else:
        color = reset = ""
    
    response_preview = response[:200] + "..." if len(response) > 200 else response
    
    print(f"\n{color}{'='*80}")
    print(f"🎯 REASONING SESSION COMPLETED [{timestamp}]")
    print(f"Session ID: {session_id}")
    print(f"{'='*80}{reset}")
    print(f"✅ FINAL RESPONSE: {response_preview}")
    print(f"{color}{'='*80}{reset}")
    
    # Print frontend data
    if collector:
        frontend_data = collector.get_frontend_data()
        print(f"\n{Fore.CYAN if COLORS_AVAILABLE else ''}{'='*80}")
        print(f"📊 FRONTEND REASONING DATA")
        print(f"{'='*80}{Style.RESET_ALL if COLORS_AVAILABLE else ''}")
        print(json.dumps(frontend_data, indent=2, default=str))
        print(f"{Fore.CYAN if COLORS_AVAILABLE else ''}{'='*80}{Style.RESET_ALL if COLORS_AVAILABLE else ''}\n")

# =============================================================================
# ORIGINAL STATE SCHEMA (UNCHANGED)
# =============================================================================

class ReActState(TypedDict):
    """Clean ReAct state schema."""
    input: str
    response: Optional[str]
    session_id: str

# =============================================================================
# ORIGINAL CONTEXT HELPER (UNCHANGED)
# =============================================================================

def get_conversation_context(memory_manager, session_id: str, max_length: int = 2000) -> str:
    """Get conversation context with proper length management."""
    if not session_id or not memory_manager:
        return "No previous conversation context available."
    
    try:
        history = memory_manager.get_conversation_history_for_state(session_id)
        if not history:
            return "No previous conversation context available."
        
        # Build context from most recent messages
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
        print(f"Context built: {len(context)} chars from {len(context_parts)} messages")
        return context
        
    except Exception as e:
        print(f"Error getting conversation context: {e}")
        return "Error retrieving conversation context."

# =============================================================================
# ORIGINAL MODEL CONFIGURATION (UNCHANGED)
# =============================================================================

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

# =============================================================================
# ORIGINAL REACT PROMPT TEMPLATE (UNCHANGED)
# =============================================================================

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

# =============================================================================
# ENHANCED WORKFLOW CREATION WITH FRONTEND DATA COLLECTION
# =============================================================================

def create_react_workflow(
    es_client=None, 
    index_name: str = "research-publications-static", 
    session_id: str = None,
    memory_manager=None
) -> StateGraph:
    """Create ReAct workflow WITH frontend reasoning data collection."""
    
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
            print("Using global memory manager singleton")
        except ImportError:
            print("Warning: Could not import global memory manager")
            memory_manager = None
    
    # Create ReAct LLM
    react_llm = create_react_llm(session_id)
    
    def react_step(state: ReActState):
        """ReAct step - reasoning and acting WITH frontend data collection."""
        try:
            query = state["input"]
            session_id = state["session_id"]
            
            # CREATE FRONTEND DATA COLLECTOR
            collector = FrontendReasoningCollector(session_id, query)
            
            # LOG SESSION START
            log_session_start(query, session_id, collector)
            
            # Get conversation context
            conversation_context = get_conversation_context(memory_manager, session_id, max_length=10000)
            
            print(f"ReAct executing with context length: {len(conversation_context)} chars")
            
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
                    "react-with-frontend-logging"
                ]
            }
            
            try:
                result = agent_executor.invoke({
                    "messages": [HumanMessage(content=query)]
                }, config=config)
                
                # EXTRACT AND LOG REASONING FROM ALL MESSAGES WITH FRONTEND COLLECTION
                all_messages = result.get("messages", [])
                print(f"\n📨 Processing {len(all_messages)} messages for reasoning extraction...")
                
                for i, message in enumerate(all_messages):
                    if isinstance(message, AIMessage):
                        if hasattr(message, 'content') and message.content:
                            print(f"\n🔍 Analyzing AI message {i+1}...")
                            log_reasoning_to_terminal(message.content, "ai", session_id, collector)
                        
                        # Log tool calls if present
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            for tool_call in message.tool_calls:
                                tool_name = tool_call.get('name', 'unknown_tool')
                                tool_input = tool_call.get('args', {})
                                log_tool_call_to_terminal(tool_name, tool_input, session_id, collector)
                    
                    elif isinstance(message, ToolMessage):
                        print(f"\n🔍 Analyzing tool result message {i+1}...")
                        log_tool_result_to_terminal(message.name, message.content, session_id, collector)
                
                response_content = result["messages"][-1].content
                
                # LOG SESSION END WITH FRONTEND DATA
                log_session_end(response_content, session_id, collector)
                
                print(f"ReAct completed with response length: {len(response_content)} chars")
                
                # STORE FRONTEND DATA IN RESULT FOR LATER ACCESS
                frontend_data = collector.get_frontend_data()
                
                return {
                    "response": response_content,
                    "session_id": session_id,
                    "_frontend_reasoning_data": frontend_data  # Store for agent manager to access
                }
                
            except Exception as exec_error:
                error_response = f"Error during research: {str(exec_error)}"
                print(f"ReAct execution error: {exec_error}")
                
                # Log error
                if COLORS_AVAILABLE:
                    print(f"\n{Fore.RED + Style.BRIGHT}❌ EXECUTION ERROR: {error_response}{Style.RESET_ALL}")
                else:
                    print(f"\n❌ EXECUTION ERROR: {error_response}")
                
                return {
                    "response": error_response,
                    "session_id": session_id,
                    "_frontend_reasoning_data": collector.get_frontend_data()
                }
        
        except Exception as e:
            print(f"ReAct error: {e}")
            
            # Log critical error
            if COLORS_AVAILABLE:
                print(f"\n{Fore.RED + Back.WHITE + Style.BRIGHT}💥 CRITICAL ERROR: {str(e)}{Style.RESET_ALL}")
            else:
                print(f"\n💥 CRITICAL ERROR: {str(e)}")
            
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
# RESEARCH AGENT CLASS WITH FRONTEND DATA SUPPORT
# =============================================================================

class ResearchAgent:
    """ReAct Research Agent WITH frontend reasoning data support."""
    
    def __init__(self, es_client=None, index_name: str = "research-publications-static", 
                 recursion_limit: int = 50, memory_manager=None): 
        self.es_client = es_client
        self.index_name = index_name
        self.recursion_limit = recursion_limit
        self.memory_manager = memory_manager
        self.app = None
        self._last_frontend_data = None  # Store last reasoning data

    def _compile_agent(self, session_id: str = None):
        """Compile ReAct agent for session."""
        workflow = create_react_workflow(self.es_client, self.index_name, session_id, self.memory_manager)
        self.app = workflow.compile()

    async def stream_query(self, query: str, conversation_history: Optional[List[Dict]] = None, frontend_session_id: str = None):
        """Stream query using ReAct workflow with frontend data collection."""
        
        session_id = frontend_session_id or f"fallback_{str(uuid.uuid4())}"
        
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
                "react-with-frontend-data"
            ],
            "run_name": f"ReAct-Query-Turn-{len(conversation_history or []) + 1}"
        }
        
        try:
            # Use astream with proper async context handling
            stream = self.app.astream(initial_state, config=config)
            
            async for event in stream:
                print(f"Streaming event: {list(event.keys())}")
                if 'react' in event and 'response' in event['react']:
                    # Store frontend data for later access
                    self._last_frontend_data = event['react'].get('_frontend_reasoning_data')
                    print(f"Found response in event: {event['react']['response'][:100]}...")
                yield event
                
        except GeneratorExit:
            # Handle generator cleanup gracefully
            print("Stream generator cleanup - this is normal")
            pass
        except Exception as e:
            print(f"Streaming error: {e}")
            yield {"error": {"error": str(e)}}

    def get_last_reasoning_data(self) -> Optional[Dict[str, Any]]:
        """Get the reasoning data from the last query execution."""
        return self._last_frontend_data

    async def stream_query_without_recompile(self, query: str, conversation_history: Optional[List[Dict]] = None, frontend_session_id: str = None):
        """Backward compatibility method."""
        async for event in self.stream_query(query, conversation_history, frontend_session_id):
            yield event


if __name__ == "__main__":
    print("🧠 Enhanced ReAct workflow with frontend data collection ready!")
    print("💡 Install colorama for colored output: pip install colorama")
    print("📊 Frontend reasoning data will be printed to terminal and stored for API access")