"""
Async Flask Application - Modern async/await pattern
Removes: Complex threading, event loop management, double timeouts
Adds: Direct async execution, cleaner error handling, better performance
"""

import os
import asyncio
from quart import Quart, request, jsonify
from quart_cors import cors

from research_agent.core.agent_manager import AgentManager


def create_app():
    """Create and configure async Flask app using Quart."""
    app = Quart(__name__)
    
    # Enable CORS
    app = cors(app)
    
    # Initialize the agent manager
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

    @app.route('/chat/respond', methods=['POST'])
    async def chat_respond():
        """Handle chat requests - now fully async."""
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

            # Direct async processing - properly calling the async method!
            result = await agent_manager.process_query_async(query, session_id)
            
            if result['success']:
                return jsonify({
                    "success": True,
                    "response_content": result['response'],
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
        
        print("Research Agent Server starting (Async)...")
        print(f"URL: http://{host}:{port}")
        
        # Use Quart's async run method
        app.run(host=host, port=port, debug=debug, use_reloader=False)
        
    except KeyboardInterrupt:
        print("Server stopped")
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        exit(1)