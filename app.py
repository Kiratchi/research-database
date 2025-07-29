"""
Updated Flask Application - properly disables sessions and removes deprecated functionality
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from research_agent.core.agent_manager import AgentManager


def create_app():
    """Create and configure Flask app."""
    app = Flask(__name__)
    
    # Properly disable Flask sessions entirely
    app.config['SECRET_KEY'] = None  # No secret key = no session encryption
    app.config['SESSION_COOKIE_NAME'] = None  # Disable session cookie entirely
    app.config['PERMANENT_SESSION_LIFETIME'] = 0
    
    # Enable CORS
    CORS(app)
    
    # Initialize the agent manager (handles all business logic)
    agent_manager = AgentManager()
    
    @app.route('/')
    def index():
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
    def status():
        """Get system status."""
        return jsonify(agent_manager.get_status())

    @app.route('/health')
    def health():
        """Health check."""
        health_info = agent_manager.health_check()
        
        if health_info["status"] == "healthy":
            return jsonify(health_info), 200
        elif health_info["status"] == "degraded":
            return jsonify(health_info), 206
        else:
            return jsonify(health_info), 503

    @app.route('/chat/respond', methods=['POST'])
    def chat_respond():
        """Handle chat requests with automatic memory management."""
        try:
            data = request.get_json()
            
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

            # AgentManager handles everything: processing + memory management
            result = agent_manager.process_query(query, session_id)
            
            if result['success']:
                return jsonify({
                    "success": True,
                    "response_content": result['response'],
                    "session_id": result['session_id'],
                    "execution_time": result.get('execution_time', 0),
                    "response_type": result.get('response_type', 'unknown')
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
    def clear_memory():
        """Clear conversation memory for a session."""
        try:
            data = request.get_json() or {}
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
    def get_session_info(session_id):
        """Get session information and memory stats."""
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

    @app.route('/admin/memory-stats')
    def memory_stats():
        """Get memory statistics for all sessions."""
        try:
            stats = agent_manager.get_memory_stats()
            return jsonify(stats)
        except Exception as e:
            return jsonify({
                "error": f"Error: {str(e)}"
            }), 500

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "success": False,
            "error": "Endpoint not found"
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
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
        print("✅ Environment variables loaded")
    except ImportError:
        print("📝 Using system environment variables")
    except Exception as e:
        print(f"⚠️ Warning loading .env: {e}")

    # Create and run app
    try:
        app = create_app()
        
        # Configuration
        host = os.getenv('FLASK_HOST', 'localhost')
        port = int(os.getenv('FLASK_PORT', 5000))
        debug = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'
        
        print(f"🚀 Starting Research Agent Server")
        print(f"🌐 URL: http://{host}:{port}")
        print(f"🔧 Debug: {debug}")
        print(f"📝 Sessions: Fully disabled")
        
        app.run(host=host, port=port, debug=debug, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        exit(1)