"""
Flask application for the Research Publications Chat Agent - FIXED VERSION

This Flask app provides a web interface for the research agent system,
calling the research workflow directly without any router layer.
"""

import os
import json
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

from flask import Flask, render_template, request, jsonify, Response, session
from flask_cors import CORS
from werkzeug.exceptions import BadRequest

# Import the core research components directly
from src.research_agent.core.workflow import ResearchAgent, run_research_query
from elasticsearch import Elasticsearch
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
    """Simplified Flask wrapper for direct research agent access."""
    
    def __init__(self):
        """Initialize the Flask research agent."""
        self.es_client = None
        self.index_name = "research-publications-static"
        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0
        }
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize Elasticsearch client."""
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
                
        except Exception as e:
            print(f"❌ Error initializing Elasticsearch: {str(e)}")
            self.es_client = None
    
    def is_ready(self) -> bool:
        """Check if the agent is ready to process queries."""
        return self.es_client is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "elasticsearch_connected": self.es_client is not None and self.es_client.ping() if self.es_client else False,
            "system_ready": self.is_ready(),
            "query_stats": self.query_stats,
            "index_name": self.index_name
        }
    
    def process_query_direct(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Process query directly without router."""
        self.query_stats["total_queries"] += 1
        
        try:
            # Check for simple queries first
            simple_response = handle_simple_query(query)
            if simple_response:
                self.query_stats["successful_queries"] += 1
                return {
                    'success': True,
                    'response': simple_response,
                    'metadata': {'workflow_type': 'simple_response'}
                }
            
            if not self.is_ready():
                self.query_stats["failed_queries"] += 1
                return {
                    'success': False,
                    'error': 'Elasticsearch not connected',
                    'response': None
                }
            
            # Process with research workflow directly
            result = run_research_query(
                query=query,
                es_client=self.es_client,
                index_name=self.index_name,
                recursion_limit=50,
                stream=False,
                conversation_history=conversation_history
            )
            
            self.query_stats["successful_queries"] += 1
            return {
                'success': True,
                'response': result.get('response', 'No response generated'),
                'metadata': {
                    'workflow_type': 'research_workflow',
                    'plan': result.get('plan', []),
                    'total_results': result.get('total_results'),
                    'session_id': result.get('session_id')
                }
            }
            
        except Exception as e:
            self.query_stats["failed_queries"] += 1
            return {
                'success': False,
                'error': str(e),
                'response': None
            }
    
    async def stream_query_direct(self, query: str, conversation_history: Optional[List[Dict]] = None):
        """Stream a query directly using ResearchAgent with FIXED response capture."""
        print(f"🔍 Direct Agent: Starting stream query: {query}")
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
                        'workflow_type': 'simple_response'
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
            
            print("✅ Direct Agent: ResearchAgent initialized, starting stream")
            event_count = 0
            final_response = None
            
            async for event in research_agent.stream_query(query, conversation_history):
                event_count += 1
                print(f"📨 Direct Agent: Received event #{event_count}")
                print(f"🔍 Direct Agent: Event structure: {list(event.keys())}")
                
                # Process events from LangGraph workflow
                for node_name, node_data in event.items():
                    print(f"🔍 Direct Agent: Processing node '{node_name}' with data keys: {list(node_data.keys()) if isinstance(node_data, dict) else 'not a dict'}")
                    
                    # FIXED: Handle None node_data
                    if node_data is None:
                        print(f"⚠️ Direct Agent: Node '{node_name}' returned None data, skipping")
                        continue
                    
                    # Ensure node_data is a dictionary
                    if not isinstance(node_data, dict):
                        print(f"⚠️ Direct Agent: Node '{node_name}' data is not a dict: {type(node_data)}")
                        continue
                    
                    if node_name == "__end__":
                        print(f"🎯 Direct Agent: Found __end__ node!")
                        print(f"🎯 Direct Agent: End node data: {node_data}")
                        
                        final_response = node_data.get('response', 'No response in end node')
                        print(f"🎯 Direct Agent: Final response from __end__: {final_response[:200]}...")
                        
                        self.query_stats["successful_queries"] += 1
                        yield json.dumps({
                            'type': 'final',
                            'content': {
                                'response': final_response,
                                'workflow_type': 'research_workflow',
                                'source': '__end__ node'
                            },
                            'timestamp': datetime.now().isoformat()
                        }) + '\n'
                        
                    elif node_name == "complete":
                        print(f"🎯 Direct Agent: Found complete node!")
                        print(f"🎯 Direct Agent: Complete node data: {node_data}")
                        
                        final_response = node_data.get('response', 'No response in complete node')
                        print(f"🎯 Direct Agent: Final response from complete: {final_response[:200]}...")
                        
                        self.query_stats["successful_queries"] += 1
                        yield json.dumps({
                            'type': 'final',
                            'content': {
                                'response': final_response,
                                'workflow_type': 'research_workflow',
                                'source': 'complete node'
                            },
                            'timestamp': datetime.now().isoformat()
                        }) + '\n'
                        
                    elif node_name == "replan":
                        print(f"🔄 Direct Agent: Replan data: {node_data}")
                        
                        # FIXED: Check if replan contains a final response OR a final_response
                        replan_response = node_data.get('response') or node_data.get('final_response')
                        if replan_response:
                            print(f"🎯 Direct Agent: Found final response in replan node!")
                            print(f"🎯 Direct Agent: Replan response: {replan_response[:200]}...")
                            
                            final_response = replan_response
                            self.query_stats["successful_queries"] += 1
                            yield json.dumps({
                                'type': 'final',
                                'content': {
                                    'response': final_response,
                                    'workflow_type': 'research_workflow',
                                    'source': 'replan node'
                                },
                                'timestamp': datetime.now().isoformat()
                            }) + '\n'
                        else:
                            # Regular replan event
                            yield json.dumps({
                                'type': 'replan',
                                'content': {
                                    **node_data,
                                    'message': 'Updating research plan...'
                                },
                                'timestamp': datetime.now().isoformat()
                            }) + '\n'
                            
                    elif node_name == "planner":
                        print(f"📋 Direct Agent: Planner data: {node_data}")
                        yield json.dumps({
                            'type': 'plan',
                            'content': {
                                'plan': node_data.get('plan', []),
                                'message': 'Creating research plan...'
                            },
                            'timestamp': datetime.now().isoformat()
                        }) + '\n'
                        
                    elif node_name == "agent":
                        print(f"🔧 Direct Agent: Agent execution data keys: {list(node_data.keys()) if isinstance(node_data, dict) else 'not a dict'}")
                        
                        # Check for response in agent data
                        agent_response = node_data.get('response')
                        if agent_response:
                            print(f"🎯 Direct Agent: Found response in agent node: {agent_response[:200]}...")
                            final_response = agent_response
                        
                        yield json.dumps({
                            'type': 'execution',
                            'content': {
                                **node_data,
                                'message': 'Executing research step...'
                            },
                            'timestamp': datetime.now().isoformat()
                        }) + '\n'
                        
                    else:
                        print(f"📦 Direct Agent: Other node '{node_name}': {node_data}")
                        yield json.dumps({
                            'type': 'step',
                            'content': node_data,
                            'node': node_name,
                            'timestamp': datetime.now().isoformat()
                        }) + '\n'
            
            print(f"✅ Direct Agent: Stream completed, sent {event_count} events")
            print(f"📋 Direct Agent: Final response captured: {bool(final_response)}")
            
            # FIXED: If we captured a final response but didn't send it yet, send it now
            if final_response and final_response not in ["No response in end node", "No response in complete node"]:
                print(f"🎯 Direct Agent: Sending captured final response")
                yield json.dumps({
                    'type': 'final',
                    'content': {
                        'response': final_response,
                        'workflow_type': 'research_workflow',
                        'source': 'captured response'
                    },
                    'timestamp': datetime.now().isoformat()
                }) + '\n'
            else:
                # If we still don't have a proper final response
                print("⚠️ Direct Agent: No final response captured, sending fallback")
                self.query_stats["failed_queries"] += 1
                yield json.dumps({
                    'type': 'final',
                    'content': {
                        'response': f"Research completed but no final response was generated. Processed {event_count} events.",
                        'workflow_type': 'research_workflow_incomplete',
                        'debug_info': f"Processed {event_count} events but no response found"
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
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here-change-in-production')

# Make sessions temporary (not permanent) so they reset on browser close
app.config['PERMANENT_SESSION_LIFETIME'] = 60  # 1 minute - very short
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Enable CORS for frontend JavaScript
CORS(app)

# Store conversation histories per session ID with timestamps
session_conversations = {}
session_timestamps = {}  # Track when sessions were created

# Initialize the research agent (simplified)
research_agent = FlaskResearchAgent()


def cleanup_old_sessions():
    """Remove sessions older than 1 hour to prevent memory leaks."""
    try:
        current_time = time.time()
        old_sessions = []
        
        for session_id, timestamp in session_timestamps.items():
            # Remove sessions older than 1 hour (3600 seconds)
            if current_time - timestamp > 3600:
                old_sessions.append(session_id)
        
        for session_id in old_sessions:
            if session_id in session_conversations:
                del session_conversations[session_id]
            if session_id in session_timestamps:
                del session_timestamps[session_id]
        
        if old_sessions:
            print(f"🧹 Cleaned up {len(old_sessions)} old sessions")
            print(f"📊 Active sessions: {len(session_conversations)}")
    
    except Exception as e:
        print(f"⚠️ Error cleaning up sessions: {e}")


@app.route('/')
def index():
    """Main chat interface."""
    return render_template('index.html')


@app.route('/status')
def status():
    """Get system status."""
    return jsonify(research_agent.get_status())


@app.route('/chat', methods=['POST'])
def chat():
    """Process a chat query - non-streaming version."""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            raise BadRequest('Missing message in request')
        
        query = data['message'].strip()
        session_id = data.get('session_id', f'default_{int(time.time())}')
        
        if not query:
            raise BadRequest('Empty message')
        
        # Get or create conversation history for this session
        if session_id not in session_conversations:
            session_conversations[session_id] = []
            session_timestamps[session_id] = time.time()
            cleanup_old_sessions()
        
        conversation_history = session_conversations[session_id]
        
        # Process query directly
        result = research_agent.process_query_direct(query, conversation_history)
        
        if result['success']:
            # Update conversation history
            session_conversations[session_id].append({'role': 'user', 'content': query})
            session_conversations[session_id].append({'role': 'assistant', 'content': result['response']})
            
            # Keep last 10 messages
            if len(session_conversations[session_id]) > 10:
                session_conversations[session_id] = session_conversations[session_id][-10:]
        
        return jsonify(result)
        
    except BadRequest as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500


@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    """Process a chat query with streaming response - DIRECT TO RESEARCH AGENT."""
    try:
        data = request.get_json()
        print(f"🔍 Stream endpoint: Received data: {data}")
        
        if not data or 'message' not in data:
            raise BadRequest('Missing message in request')
        
        query = data['message'].strip()
        session_id = data.get('session_id', f'default_{int(time.time())}')
        
        if not query:
            raise BadRequest('Empty message')
        
        # Get or create conversation history for this session
        if session_id not in session_conversations:
            session_conversations[session_id] = []
            session_timestamps[session_id] = time.time()
            print(f"🆕 Created new conversation for session: {session_id}")
            cleanup_old_sessions()
        
        conversation_history = session_conversations[session_id]
        print(f"🔍 Stream endpoint: Session {session_id} has {len(conversation_history)} messages")
        
        def generate():
            """Generator for Server-Sent Events."""
            response_content = ""
            
            try:
                # Run the async streaming in a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def stream_handler():
                    nonlocal response_content
                    async for event_json in research_agent.stream_query_direct(query, conversation_history):
                        yield f"data: {event_json}\n\n"
                        
                        # Track response content for conversation history
                        try:
                            event = json.loads(event_json.strip())
                            if event.get('type') == 'final' and event.get('content', {}).get('response'):
                                response_content = event['content']['response']
                                print(f"✅ Direct Agent: Captured final response")
                        except Exception as e:
                            print(f"⚠️ Direct Agent: Error processing event: {e}")
                            pass
                    
                    # Send completion event
                    yield f"data: {json.dumps({'type': 'done', 'response_content': response_content})}\n\n"
                
                # Run the async generator
                async_gen = stream_handler()
                
                # Convert async generator to sync generator
                while True:
                    try:
                        event = loop.run_until_complete(async_gen.__anext__())
                        yield event
                    except StopAsyncIteration:
                        break
                        
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            finally:
                loop.close()
        
        return Response(
            generate(),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except BadRequest as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


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
        
        print(f"💾 Updated conversation history for session {session_id}: {len(conversation_history)} messages")
        
        return jsonify({
            'success': True, 
            'conversation_length': len(conversation_history),
            'session_id': session_id,
            'message': 'Conversation history updated successfully'
        })
        
    except Exception as e:
        print(f"❌ Error updating conversation history: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/chat/clear-memory', methods=['POST'])
def clear_memory():
    """Clear conversation memory for specific session only."""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', 'default')
        
        print(f"🧹 Clearing memory for session: {session_id}")
        
        # Remove only this session from memory
        old_length = len(session_conversations.get(session_id, []))
        if session_id in session_conversations:
            del session_conversations[session_id]
        if session_id in session_timestamps:
            del session_timestamps[session_id]
        
        print(f"✅ Session {session_id} cleared: {old_length} messages removed")
        print(f"📊 Total sessions remaining: {len(session_conversations)}")
        
        return jsonify({
            'success': True,
            'message': f'Memory cleared for session {session_id}. Removed {old_length} messages.',
            'sessions_remaining': len(session_conversations)
        })
        
    except Exception as e:
        print(f"❌ Error clearing memory: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/debug/conversation', methods=['GET'])
def debug_conversation():
    """Debug endpoint to check conversation history."""
    session_id = request.args.get('session_id', 'default')
    conversation_history = session_conversations.get(session_id, [])
    
    return jsonify({
        'session_id': session_id,
        'conversation_length': len(conversation_history),
        'conversation_history': conversation_history,
        'total_sessions': len(session_conversations),
        'all_session_ids': list(session_conversations.keys())
    })


@app.route('/performance')
def performance():
    """Get performance statistics."""
    return jsonify(research_agent.query_stats)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Page not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check if system is ready
    if not research_agent.is_ready():
        print("⚠️  WARNING: System not fully initialized!")
        print("Please check your .env file and Elasticsearch connection.")
        print("System status:", research_agent.get_status())
    
    # Run Flask app
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('FLASK_PORT', 5000))
    
    print(f"🚀 Starting Simplified Flask Research Agent on port {port}")
    print(f"🌐 Access the application at: http://localhost:{port}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode,
        use_reloader=False  # Disable auto-reload to prevent memory loss
    )