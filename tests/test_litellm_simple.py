#!/usr/bin/env python3
"""
Test LiteLLM exactly as shown in the working notebook
"""

import os
import sys
import litellm
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_notebook_pattern():
    """Test exactly as shown in the notebook."""
    print("🧪 Testing LiteLLM with exact notebook pattern")
    print("=" * 50)
    
    # Load environment variables - exact same as notebook
    load_dotenv()
    
    # Get values from environment - exact same as notebook
    LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
    LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
    
    # Validate environment variables - exact same as notebook
    if not LITELLM_API_KEY or not LITELLM_BASE_URL:
        print("Error: Missing API key or base URL in .env file")
        return False
    
    print(f"✅ API Key: {LITELLM_API_KEY[:20]}...")
    print(f"✅ Base URL: {LITELLM_BASE_URL}")
    
    # Set LiteLLM config - exact same as notebook
    litellm.api_base = LITELLM_BASE_URL
    litellm.api_key = LITELLM_API_KEY
    
    print(f"✅ LiteLLM configured")
    
    # Test function exactly from notebook
    def chat_with_model(model_name, message, stream=False):
        """Chat with any available model through LiteLLM proxy"""
        try:
            response = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": message}],
                api_key=LITELLM_API_KEY,
                api_base=LITELLM_BASE_URL,
                stream=stream
            )
            
            if stream:
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        full_response += content
                print()
                return full_response
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Test exactly as in notebook
    model_name = "anthropic/claude-haiku-3.5"
    message = "What goes well with pancakes besides jam?"
    
    print(f"\n💬 Testing with model: {model_name}")
    print(f"📝 Query: {message}")
    
    result = chat_with_model(model_name, message)
    print(f"📄 Response: {result}")
    
    # Check if successful
    if "Error:" not in result:
        print("✅ LiteLLM test successful!")
        return True
    else:
        print("❌ LiteLLM test failed!")
        return False

if __name__ == "__main__":
    success = test_notebook_pattern()
    if success:
        print("\n🎉 LiteLLM is working! Ready to integrate with LangChain.")
    else:
        print("\n❌ LiteLLM connection failed. Check your setup.")