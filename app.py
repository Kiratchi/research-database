"""
Flask application for the Research Publications Chat Agent - WITH LANGSMITH TRACING

This Flask app provides a web interface for the research agent system,
calling the research workflow directly with LangSmith observability.
"""

import os
import json
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

from flask import Flask, render_template, request, jsonify, Response, session, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import BadRequest

# Import the core research components directly
from research_agent.core.workflow import ResearchAgent, run_research_query
from elasticsearch import Elasticsearch

# LangSmith imports
from langsmith import Client
import traceback

def handle_simple_query(query: str) -> Optional[str]:
    """Handle basic greetings and help requests with enhanced pattern matching."""
    # Clean the query: strip whitespace and convert to lowercase
    query_clean = query.lower().strip()
    
    # Remove punctuation for better matching
    query_no_punct = query_clean.rstrip('!?.,;:')
    
    # Greeting patterns
    greeting_patterns = [
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 
        'good evening', 'howdy', 'greetings', 'hiya', 'sup'
    ]
    
    # Thank you patterns
    thanks_patterns = [
        'thanks', 'thank you', 'thank you very much', 'thank you so much',
        'thankyou', 'thx', 'ty', 'much appreciated', 'appreciate it'
    ]
    
    # Goodbye patterns
    goodbye_patterns = [
        'bye', 'goodbye', 'see you', 'see ya', 'farewell', 'take care',
        'that\'s all', 'that is all', 'i\'m done', 'im done', 'all done'
    ]
    
    # Help patterns
    help_patterns = [
        'what can you do', 'help me', 'how do you work', 'what do you do',
        'how can you help', 'what are your capabilities', 'help',
        'what can i ask', 'how does this work', 'instructions'
    ]
    
    # Check for greetings
    if query_no_punct in greeting_patterns:
        return "Hello! I'm here to help you search and analyze research publications. What would you like to know?"
    
    # Check for thanks
    if query_no_punct in thanks_patterns:
        return "You're welcome! Feel free to ask if you need help with any research publications."
    
    # Check for goodbyes
    if query_no_punct in goodbye_patterns:
        return "You're welcome! Feel free to ask if you need help with any research publications."
    
    # Check for help requests (using 'in' for partial matching)
    if any(pattern in query_clean for pattern in help_patterns):
        return ("I can help you search and analyze research publications. I can:\n"
               "• Find papers by author, title, or topic\n"
               "• Count publications and analyze trends\n"
               "• Answer questions about specific papers\n"
               "• Handle follow-up questions in context\n\n"
               "Just ask me anything about research publications!")
    
    return None


class FlaskResearchAgent:
    """Flask wrapper for research agent access with LangSmith tracing."""
    
    def __init__(self):
        """Initialize the Flask research agent."""
        self.es_client = None
        self.index_name = "research-publications-static"
        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0
        }
        self.langsmith_client = None
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize Elasticsearch client and LangSmith."""
        try:
            # Initialize Elasticsearch client
            es_host = os.getenv("ES_HOST")
            es_user = os.getenv("ES_USER")
            es_pass = os.getenv("ES_PASS")
            
            if es_host and es_user and es_pass:
                self.es_client = Elasticsearch(
                    [es_host],
                    http_auth=(es_user, es_pass),
                    timeout=30,
                    max_retries=3,
                    retry_on_timeout=True
                )
                
                # Test connection
                if self.es_client.ping():
                    print("✅ Elasticsearch connected successfully")
                else:
                    print("❌ Elasticsearch connection failed")
                    self.es_client = None
            else:
                print("❌ Elasticsearch credentials not found in environment")
            
            # Initialize LangSmith client
            if os.getenv("LANGCHAIN_API_KEY"):
                try:
                    self.langsmith_client = Client(
                        api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
                        api_key=os.getenv("LANGCHAIN_API_KEY")
                    )
                    print("✅ LangSmith client initialized successfully")
                except Exception as e:
                    print(f"⚠️ LangSmith initialization warning: {e}")
            else:
                print("📝 LangSmith not configured (LANGCHAIN_API_KEY missing)")
                
        except Exception as e:
            print(f"❌ Error initializing components: {str(e)}")
            self.es_client = None
    
    def is_ready(self) -> bool:
        """Check if the agent is ready to process queries."""
        return self.es_client is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        es_connected = self.es_client is not None and self.es_client.ping() if self.es_client else False
        langsmith_configured = self.langsmith_client is not None
            
        return {
            "elasticsearch_connected": es_connected,
            "langsmith_configured": langsmith_configured,
            "system_ready": self.is_ready(),
            "query_stats": self.query_stats,
            "index_name": self.index_name,
            "performance_stats": {
                "total_queries": self.query_stats["total_queries"],
                "fast_path_percentage": 85.0
            }
        }
    
    def process_query_direct(self, query: str, conversation_history: Optional[List[Dict]] = None, session_id: str = None) -> Dict[str, Any]:
        """Process query directly without router with LangSmith tracing."""
        self.query_stats["total_queries"] += 1
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        try:
            # Check for simple queries first
            simple_response = handle_simple_query(query)
            if simple_response:
                self.query_stats["successful_queries"] += 1
                return {
                    'success': True,
                    'response': simple_response,
                    'metadata': {
                        'workflow_type': 'simple_response',
                        'session_id': session_id
                    }
                }
            
            if not self.is_ready():
                self.query_stats["failed_queries"] += 1
                return {
                    'success': False,
                    'error': 'Elasticsearch not connected',
                    'response': None
                }
            
            # Process with research workflow directly with session ID
            result = run_research_query(
                query=query,
                es_client=self.es_client,
                index_name=self.index_name,
                recursion_limit=50,
                stream=False,
                conversation_history=conversation_history,
                session_id=session_id  # Pass session ID for tracing
            )
            
            self.query_stats["successful_queries"] += 1
            return {
                'success': True,
                'response': result.get('response', 'No response generated'),
                'metadata': {
                    'workflow_type': 'research_workflow',
                    'plan': result.get('plan', []),
                    'total_results': result.get('total_results'),
                    'session_id': session_id
                }
            }
                
        except Exception as e:
            self.query_stats["failed_queries"] += 1
            return {
                'success': False,
                'error': str(e),
                'response': None
            }
    
    async def stream_query_direct(self, query: str, conversation_history: Optional[List[Dict]] = None, session_id: str = None):
        """Stream a query directly using ResearchAgent with LangSmith tracing."""
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
            
        print(f"🔍 Direct Agent: Starting stream query: {query}")
        print(f"🔍 Direct Agent: Session ID: {session_id}")
        print(f"🔍 Direct Agent: Conversation history length: {len(conversation_history) if conversation_history else 0}")
        
        self.query_stats["total_queries"] += 1
        
        try:
            # Check for simple queries first
            simple_response = handle_simple_query(query)
            if simple_response:
                self.query_stats["successful_queries"] += 1
                yield json.dumps({
                    'type': 'final',
                    'content': {
                        'response': simple_response,
                        'workflow_type': 'simple_response',
                        'session_id': session_id
                    },
                    'timestamp': datetime.now().isoformat()
                }) + '\n'
                return

            if not self.is_ready():
                self.query_stats["failed_queries"] += 1
                yield json.dumps({
                    'type': 'error',
                    'content': 'Elasticsearch not connected',
                    'timestamp': datetime.now().isoformat()
                }) + '\n'
                return

            # Initialize research agent and stream directly
            research_agent = ResearchAgent(
                es_client=self.es_client,
                index_name=self.index_name,
                recursion_limit=50
            )
            
            event_count = 0
            final_response = None
            
            async for event in research_agent.stream_query(query, conversation_history):
                event_count += 1
                
                # Process events from LangGraph workflow
                for node_name, node_data in event.items():
                   
                    # Handle None node_data
                    if node_data is None:
                        print(f"⚠️ Direct Agent: Node '{node_name}' returned None data, skipping")
                        continue
                    
                    # Ensure node_data is a dictionary
                    if not isinstance(node_data, dict):
                        print(f"⚠️ Direct Agent: Node '{node_name}' data is not a dict: {type(node_data)}")
                        continue
                    
                    if node_name == "__end__":
                        print(f"🎯 Direct Agent: Found __end__ node!")
                        final_response = node_data.get('response', 'No response in end node')
                        
                        self.query_stats["successful_queries"] += 1
                        yield json.dumps({
                            'type': 'final',
                            'content': {
                                'response': final_response,
                                'workflow_type': 'research_workflow',
                                'source': '__end__ node',
                                'session_id': session_id
                            },
                            'timestamp': datetime.now().isoformat()
                        }) + '\n'
                        
                    elif node_name == "complete":
                        final_response = node_data.get('response', 'No response in complete node')
                        
                        self.query_stats["successful_queries"] += 1
                        yield json.dumps({
                            'type': 'final',
                            'content': {
                                'response': final_response,
                                'workflow_type': 'research_workflow',
                                'source': 'complete node',
                                'session_id': session_id
                            },
                            'timestamp': datetime.now().isoformat()
                        }) + '\n'
                        
                    elif node_name == "replan":
                        replan_response = node_data.get('response') or node_data.get('final_response')
                        if replan_response:
                            final_response = replan_response
                            self.query_stats["successful_queries"] += 1
                            yield json.dumps({
                                'type': 'final',
                                'content': {
                                    'response': final_response,
                                    'workflow_type': 'research_workflow',
                                    'source': 'replan node',
                                    'session_id': session_id
                                },
                                'timestamp': datetime.now().isoformat()
                            }) + '\n'
                        else:
                            yield json.dumps({
                                'type': 'replan',
                                'content': {
                                    **node_data,
                                    'message': 'Updating research plan...',
                                    'session_id': session_id
                                },
                                'timestamp': datetime.now().isoformat()
                            }) + '\n'
                            
                    elif node_name == "planner":
                        yield json.dumps({
                            'type': 'plan',
                            'content': {
                                'plan': node_data.get('plan', []),
                                'message': 'Creating research plan...',
                                'session_id': session_id
                            },
                            'timestamp': datetime.now().isoformat()
                        }) + '\n'
                        
                    elif node_name == "agent":
                        agent_response = node_data.get('response')
                        if agent_response:
                            final_response = agent_response
                        
                        yield json.dumps({
                            'type': 'execution',
                            'content': {
                                **node_data,
                                'message': 'Executing research step...',
                                'session_id': session_id
                            },
                            'timestamp': datetime.now().isoformat()
                        }) + '\n'
                        
                    else:
                        yield json.dumps({
                            'type': 'step',
                            'content': {
                                **node_data,
                                'session_id': session_id
                            },
                            'node': node_name,
                            'timestamp': datetime.now().isoformat()
                        }) + '\n'
            
            # Send final response if captured
            if final_response and final_response not in ["No response in end node", "No response in complete node"]:
                yield json.dumps({
                    'type': 'final',
                    'content': {
                        'response': final_response,
                        'workflow_type': 'research_workflow',
                        'source': 'captured response',
                        'session_id': session_id
                    },
                    'timestamp': datetime.now().isoformat()
                }) + '\n'
            else:
                self.query_stats["failed_queries"] += 1
                yield json.dumps({
                    'type': 'final',
                    'content': {
                        'response': f"Research completed but no final response was generated. Processed {event_count} events.",
                        'workflow_type': 'research_workflow_incomplete',
                        'session_id': session_id
                    },
                    'timestamp': datetime.now().isoformat()
                }) + '\n'
                
        except Exception as e:
            self.query_stats["failed_queries"] += 1
            print(f"❌ Direct Agent: Stream error: {str(e)}")
            print(f"❌ Direct Agent: Full traceback: {traceback.format_exc()}")
            yield json.dumps({
                'type': 'error',
                'content': f'Query processing error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }) + '\n'


# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')

# Make sessions temporary
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Enable CORS for frontend JavaScript
CORS(app)

# Store conversation histories per session ID with timestamps
session_conversations = {}
session_timestamps = {}

# Initialize the research agent
research_agent = FlaskResearchAgent()


def cleanup_old_sessions():
    """Remove sessions older than 1 hour to prevent memory leaks."""
    try:
        current_time = time.time()
        old_sessions = []
        
        for session_id, timestamp in session_timestamps.items():
            if current_time - timestamp > 3600:
                old_sessions.append(session_id)
        
        for session_id in old_sessions:
            if session_id in session_conversations:
                del session_conversations[session_id]
            if session_id in session_timestamps:
                del session_timestamps[session_id]
        
        if old_sessions:
            print(f"🧹 Cleaned up {len(old_sessions)} old sessions")
    
    except Exception as e:
        print(f"⚠️ Error cleaning up sessions: {e}")


@app.route('/')
def index():
    """Main chat interface."""
    try:
        # Try to serve index.html from current directory
        import os
        if os.path.exists('index.html'):
            with open('index.html', 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return '''
            <h1>Missing index.html</h1>
            <p>Please make sure index.html is in the same directory as app.py</p>
            <p>Current directory: ''' + os.getcwd() + '''</p>
            <p>Files in directory: ''' + str(os.listdir('.')) + '''</p>
            '''
    except Exception as e:
        return f'Error serving index.html: {str(e)}'


@app.route('/status')
def status():
    """Get system status including LangSmith configuration."""
    return jsonify(research_agent.get_status())


@app.route('/chat/respond', methods=['POST'])
def chat_respond():
    """Non-streaming endpoint for chat responses with LangSmith tracing."""
    try:
        data = request.get_json()

        if not data or 'message' not in data:
            raise BadRequest('Missing message in request')

        query = data['message'].strip()
        session_id = data.get('session_id', f'flask_session_{int(time.time())}_{str(uuid.uuid4())[:8]}')

        if not query:
            raise BadRequest('Empty message')

        # Get or create session memory
        if session_id not in session_conversations:
            session_conversations[session_id] = []
            session_timestamps[session_id] = time.time()
            cleanup_old_sessions()

        conversation_history = session_conversations[session_id]

        # Run the async generator and accumulate content
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def run_query():
            response_content = ""
            async for event_json in research_agent.stream_query_direct(query, conversation_history, session_id):
                try:
                    event = json.loads(event_json.strip())
                    if event.get("type") == "final":
                        response_content = event["content"]["response"]
                except Exception as e:
                    print(f"⚠️ Failed to decode event: {e}")
            return response_content

        response_content = loop.run_until_complete(run_query())
        loop.close()

        return jsonify({
            "success": True,
            "response_content": response_content,
            "session_id": session_id
        })

    except BadRequest as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500


@app.route('/chat/update-history', methods=['POST'])
def update_conversation_history():
    """Update conversation history for specific session."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        response = data.get('response', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not query or not response:
            return jsonify({'success': False, 'error': 'Missing query or response'})
        
        # Get conversation history for this session
        conversation_history = session_conversations.get(session_id, [])
        
        # Add the new exchange
        conversation_history.append({'role': 'user', 'content': query})
        conversation_history.append({'role': 'assistant', 'content': response})
        
        # Keep last 10 messages (5 exchanges)
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        
        # Save back to session-specific storage
        session_conversations[session_id] = conversation_history
        
        return jsonify({
            'success': True, 
            'conversation_length': len(conversation_history),
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/chat/clear-memory', methods=['POST'])
def clear_memory():
    """Clear conversation memory for specific session only."""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', 'default')
        
        # Remove only this session from memory
        if session_id in session_conversations:
            del session_conversations[session_id]
        if session_id in session_timestamps:
            del session_timestamps[session_id]
        
        return jsonify({
            'success': True,
            'message': f'Memory cleared for session {session_id}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# New endpoint for LangSmith run information
@app.route('/chat/langsmith-info/<run_id>')
def get_langsmith_info(run_id):
    """Get LangSmith run information for debugging."""
    try:
        if not research_agent.langsmith_client:
            return jsonify({
                'success': False,
                'error': 'LangSmith not configured'
            }), 400
        
        # Get run information from LangSmith
        run_info = research_agent.langsmith_client.read_run(run_id)
        
        return jsonify({
            'success': True,
            'run_info': {
                'id': run_info.id,
                'name': run_info.name,
                'start_time': run_info.start_time.isoformat() if run_info.start_time else None,
                'end_time': run_info.end_time.isoformat() if run_info.end_time else None,
                'status': run_info.status,
                'inputs': run_info.inputs,
                'outputs': run_info.outputs,
                'error': run_info.error,
                'tags': run_info.tags,
                'extra': run_info.extra
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # Load environment variables if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded")
    except ImportError:
        print("📝 Note: python-dotenv not available, using system environment variables")
    
    # Check if system is ready
    status = research_agent.get_status()
    print(f"🚀 System Status: {status}")
    
    if not research_agent.is_ready():
        print("⚠️  WARNING: System not fully initialized!")
        print("Please check your .env file and Elasticsearch connection.")
    
    # Run Flask app
    debug_mode = True  # Enable debug for local development
    port = 5000
    
    print(f"🚀 Starting Local Research Agent Server on port {port}")
    print(f"🌐 Access the application at: http://localhost:{port}")
    print(f"🔧 Debug mode: {debug_mode}")
    print(f"📊 LangSmith configured: {research_agent.langsmith_client is not None}")
    
    app.run(
        host='localhost',  # localhost for local development
        port=port,
        debug=debug_mode,
        use_reloader=False  # Disable auto-reload to prevent memory loss
    )