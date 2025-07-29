"""
Simple script to list ALL available models on your LiteLLM server
"""

import requests
import os
import json
from dotenv import load_dotenv

def list_all_models():
    """List all available models on your LiteLLM server."""
    
    # Load environment variables
    load_dotenv()
    
    # Get LiteLLM configuration
    base_url = os.getenv("LITELLM_BASE_URL")
    api_key = os.getenv("LITELLM_API_KEY")
    
    if not base_url:
        print("❌ LITELLM_BASE_URL not found in environment variables")
        print("Please set LITELLM_BASE_URL in your .env file")
        return
    
    # Clean up base URL (remove /v1 if present)
    if base_url.endswith('/v1'):
        base_url = base_url[:-3]
    
    models_url = f"{base_url}/models"
    
    print(f"🔍 Fetching all models from: {models_url}")
    print("=" * 60)
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json"
    }
    
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["x-api-key"] = api_key
    
    try:
        response = requests.get(models_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Simple list of all model IDs
            all_models = []
            if 'data' in data:
                for model in data['data']:
                    model_id = model.get('id', 'Unknown')
                    all_models.append(model_id)
            
            print(f"✅ ALL AVAILABLE MODELS ({len(all_models)} total):")
            print("=" * 60)
            
            for i, model in enumerate(all_models, 1):
                print(f"{i:2d}. {model}")
            
            print("=" * 60)
            
            # Copy-paste ready suggestions
            print("\n📋 COPY-PASTE READY MODEL NAMES:")
            print("-" * 40)
            
            for model in all_models:
                print(f'"{model}"')
            
            # Show raw JSON response for debugging
            print(f"\n🔧 RAW RESPONSE (first 1000 chars):")
            print("-" * 40)
            print(json.dumps(data, indent=2)[:1000] + "...")
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Full response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection failed to {base_url}")
        print("Is your LiteLLM proxy server running?")
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Show environment info
    print(f"\n📋 ENVIRONMENT INFO:")
    print(f"  LITELLM_BASE_URL: {base_url}")
    print(f"  LITELLM_API_KEY: {'***' + api_key[-4:] if api_key and len(api_key) > 4 else 'Not set'}")


if __name__ == "__main__":
    list_all_models()