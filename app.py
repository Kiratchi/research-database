"""
Enhanced Async Flask Application with Streaming Support
Adds Server-Sent Events for real-time reasoning display
"""

import os
import asyncio
import json
from quart import Quart, request, jsonify, Response
from quart_cors import cors

from research_agent.core.agent_manager import AgentManager


def create_app():
    """Create and configure async Flask app using Quart with streaming support."""
    app = Quart(__name__)
    
    # Enable CORS with streaming support
    app = cors(app, allow_origin="*", allow_headers=["Content-Type"], allow_methods=["GET", "POST"])
    
    # Initialize the enhanced agent manager
    agent_manager = AgentManager()
    
    @app.route('/')
    async def index():
        """Serve the web interface."""
        try:
            if os.path.exists('index.html'):
                with open('index.html', 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                return '<h1>Research Publications Chat Agent</h1><p>Missing index.html</p>'
        except Exception as e:
            return f'<h1>Error</h1><p>{str(e)}</p>'

    @app.route('/status')
    async def status():
        """Get basic system status."""
        return jsonify(agent_manager.get_status())

    @app.route('/health')
    async def health():
        """Simple health check."""
        health_info = agent_manager.health_check()
        
        if health_info["status"] == "healthy":
            return jsonify(health_info), 200
        else:
            return jsonify(health_info), 503

    # BACKWARD COMPATIBLE: Original chat endpoint
    @app.route('/chat/respond', methods=['POST'])
    async def chat_respond():
        """Handle chat requests - backward compatible with reasoning data."""
        try:
            data = await request.get_json()
            
            if not data or 'message' not in data:
                return jsonify({
                    "success": False, 
                    "error": "Missing 'message' field"
                }), 400

            query = data['message'].strip()
            session_id = data.get('session_id')
            
            if not query:
                return jsonify({
                    "success": False, 
                    "error": "Empty message"
                }), 400

            # Use enhanced processing with reasoning data
            result = await agent_manager.process_query_async(query, session_id)
            
            if result['success']:
                return jsonify({
                    "success": True,
                    "response_content": result['response'],
                    "reasoning_data": result.get('reasoning_data'),  # NEW: Include reasoning
                    "session_id": result['session_id'],
                    "execution_time": result.get('execution_time', 0),
                    "response_type": result.get('response_type', 'research')
                })
            else:
                return jsonify({
                    "success": False,
                    "error": result['error'],
                    "session_id": result.get('session_id')
                }), 500
                
        except Exception as e:
            return jsonify({
                "success": False, 
                "error": f"Server error: {str(e)}"
            }), 500

    # NEW: Streaming endpoint with Server-Sent Events
    @app.route('/chat/respond-stream', methods=['POST'])
    async def chat_respond_stream():
        """Handle streaming chat requests with real-time reasoning."""
        try:
            data = await request.get_json()
            
            if not data or 'message' not in data:
                return jsonify({
                    "success": False, 
                    "error": "Missing 'message' field"
                }), 400

            query = data['message'].strip()
            session_id = data.get('session_id')
            
            if not query:
                return jsonify({
                    "success": False, 
                    "error": "Empty message"
                }), 400

            # Stream events generator
            async def generate_events():
                try:
                    async for event in agent_manager.stream_query_with_reasoning(query, session_id):
                        # Format as Server-Sent Event
                        event_json = json.dumps(event, default=str)
                        yield f"data: {event_json}\n\n"
                    
                    # Send final completion marker
                    yield f"data: {json.dumps({'event': 'stream_complete', 'data': {}})}\n\n"
                    
                except Exception as e:
                    # Send error event
                    error_event = {
                        "event": "error",
                        "data": {"error": f"Streaming error: {str(e)}"}
                    }
                    yield f"data: {json.dumps(error_event)}\n\n"

            # Return SSE response
            return Response(
                generate_events(), 
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Cache-Control'
                }
            )
                
        except Exception as e:
            return jsonify({
                "success": False, 
                "error": f"Server error: {str(e)}"
            }), 500

    # ALTERNATIVE: GET-based streaming endpoint (for EventSource compatibility)
    @app.route('/chat/stream/<path:encoded_params>')
    async def chat_stream_get(encoded_params):
        """GET-based streaming endpoint for EventSource compatibility."""
        try:
            import urllib.parse
            import base64
            
            # Decode parameters
            decoded = base64.b64decode(encoded_params).decode('utf-8')
            params = json.loads(decoded)
            
            query = params.get('message', '').strip()
            session_id = params.get('session_id')
            
            if not query:
                return "data: " + json.dumps({
                    "event": "error", 
                    "data": {"error": "Empty message"}
                }) + "\n\n", 400

            # Stream events generator
            async def generate_events():
                try:
                    async for event in agent_manager.stream_query_with_reasoning(query, session_id):
                        event_json = json.dumps(event, default=str)
                        yield f"data: {event_json}\n\n"
                    
                    yield f"data: {json.dumps({'event': 'stream_complete', 'data': {}})}\n\n"
                    
                except Exception as e:
                    error_event = {
                        "event": "error",
                        "data": {"error": f"Streaming error: {str(e)}"}
                    }
                    yield f"data: {json.dumps(error_event)}\n\n"

            return Response(
                generate_events(), 
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Cache-Control'
                }
            )
                
        except Exception as e:
            return "data: " + json.dumps({
                "event": "error", 
                "data": {"error": f"Server error: {str(e)}"}
            }) + "\n\n", 500

    @app.route('/chat/clear-memory', methods=['POST'])
    async def clear_memory():
        """Clear conversation memory for a session."""
        try:
            data = await request.get_json() or {}
            session_id = data.get('session_id')
            
            if not session_id:
                return jsonify({
                    "success": False,
                    "error": "session_id required"
                }), 400
            
            result = agent_manager.clear_memory(session_id)
            
            if result['success']:
                return jsonify(result)
            else:
                return jsonify(result), 500
                
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Error: {str(e)}"
            }), 500

    @app.route('/chat/session-info/<session_id>')
    async def get_session_info(session_id):
        """Get basic session information."""
        try:
            result = agent_manager.get_session_info(session_id)
            
            if result['success']:
                return jsonify(result)
            else:
                return jsonify(result), 404
                
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Error: {str(e)}"
            }), 500

    @app.route('/admin/tools-info')
    async def tools_info():
        """Get information about available tools."""
        try:
            info = agent_manager.get_tools_info()
            return jsonify(info)
        except Exception as e:
            return jsonify({
                "error": f"Error: {str(e)}"
            }), 500

    # Simple error handlers
    @app.errorhandler(404)
    async def not_found(error):
        return jsonify({
            "success": False,
            "error": "Endpoint not found"
        }), 404

    @app.errorhandler(500)
    async def internal_error(error):
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

    return app


if __name__ == '__main__':
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Environment variables loaded")
    except ImportError:
        print("Using system environment variables")
    except Exception as e:
        print(f"⚠️ Warning loading .env: {e}")

    # Create and run async app
    try:
        app = create_app()
        
        # Configuration
        host = os.getenv('FLASK_HOST', 'localhost')
        port = int(os.getenv('FLASK_PORT', 5000))
        debug = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'
        
        print("Research Agent Server starting (Async + Streaming)...")
        print(f"URL: http://{host}:{port}")
        print("🔄 Streaming enabled at /chat/respond-stream")
        
        # Use Quart's async run method
        app.run(host=host, port=port, debug=debug, use_reloader=False)
        
    except KeyboardInterrupt:
        print("Server stopped")
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        exit(1)