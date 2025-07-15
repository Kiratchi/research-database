#!/usr/bin/env python3
"""
Test different URL formats for LiteLLM
"""

import os
import sys
import litellm
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_url_formats():
    """Test different URL formats to find the correct one."""
    print("🔍 Testing different LiteLLM URL formats")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
    BASE_HOST = "https://anast.ita.chalmers.se:8083"
    
    # Different URL formats to try
    url_formats = [
        f"{BASE_HOST}",                    # Current format
        f"{BASE_HOST}/",                   # With trailing slash
        f"{BASE_HOST}/v1",                 # With /v1
        f"{BASE_HOST}/v1/",                # With /v1/
        f"{BASE_HOST}/api",                # With /api
        f"{BASE_HOST}/api/v1",             # With /api/v1
        f"{BASE_HOST}/litellm",            # With /litellm
        f"{BASE_HOST}/litellm/v1",         # With /litellm/v1
        f"{BASE_HOST}/chat",               # With /chat
        f"{BASE_HOST}/chat/completions",   # With /chat/completions
    ]
    
    print(f"🔑 Using API Key: {LITELLM_API_KEY[:20]}...")
    print(f"🏠 Base Host: {BASE_HOST}")
    
    for i, url in enumerate(url_formats, 1):
        print(f"\n🧪 Test {i}: {url}")
        
        try:
            # Set LiteLLM config
            litellm.api_base = url
            litellm.api_key = LITELLM_API_KEY
            
            # Try a simple completion
            response = litellm.completion(
                model="anthropic/claude-haiku-3.5",
                messages=[{"role": "user", "content": "Hello! Just say 'Hi' back."}],
                api_key=LITELLM_API_KEY,
                api_base=url,
                max_tokens=10
            )
            
            result = response.choices[0].message.content
            print(f"✅ SUCCESS: {result}")
            print(f"🎯 WORKING URL: {url}")
            return url
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ FAILED: {error_msg[:100]}...")
            
            # Check for specific error types
            if "Method Not Allowed" in error_msg:
                print("   → 405 Method Not Allowed")
            elif "Not Found" in error_msg:
                print("   → 404 Not Found")
            elif "Unauthorized" in error_msg:
                print("   → 401 Unauthorized")
            elif "Forbidden" in error_msg:
                print("   → 403 Forbidden")
            elif "Connection" in error_msg:
                print("   → Connection Error")
    
    print("\n❌ No working URL format found!")
    return None


def test_health_endpoints():
    """Test common health/info endpoints."""
    print("\n🏥 Testing health/info endpoints")
    print("=" * 40)
    
    import requests
    
    LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
    BASE_HOST = "https://anast.ita.chalmers.se:8083"
    
    # Common endpoints to check
    endpoints = [
        "/health",
        "/status", 
        "/info",
        "/models",
        "/v1/models",
        "/api/v1/models",
        "/litellm/models",
        "/",
    ]
    
    headers = {
        "Authorization": f"Bearer {LITELLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    for endpoint in endpoints:
        url = f"{BASE_HOST}{endpoint}"
        print(f"\n🔍 Testing: {url}")
        
        try:
            response = requests.get(url, headers=headers, timeout=5, verify=False)
            print(f"   Status: {response.status_code}")
            print(f"   Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
            
            if response.status_code == 200:
                try:
                    json_data = response.json()
                    print(f"   ✅ JSON Response: {str(json_data)[:100]}...")
                except:
                    print(f"   📄 Text Response: {response.text[:100]}...")
            
        except requests.exceptions.SSLError:
            print(f"   ❌ SSL Error - trying without verification...")
            try:
                response = requests.get(url, headers=headers, timeout=5, verify=False)
                print(f"   Status (no SSL): {response.status_code}")
            except Exception as e:
                print(f"   ❌ Still failed: {str(e)[:50]}...")
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}...")


def suggest_solutions():
    """Suggest potential solutions."""
    print("\n💡 Potential Solutions:")
    print("=" * 30)
    
    suggestions = [
        "1. Check if the server is running LiteLLM proxy",
        "2. Verify the port (8083) is correct",
        "3. Check if you need /v1 or /api/v1 in the URL",
        "4. Confirm the API key format matches the server",
        "5. Try without SSL verification (requests.get(..., verify=False))",
        "6. Check if there's a different endpoint for completions",
        "7. Ask your admin for the correct API endpoint format"
    ]
    
    for suggestion in suggestions:
        print(f"   {suggestion}")


def main():
    """Run all URL format tests."""
    print("🔧 LiteLLM URL Format Debugger")
    print("=" * 60)
    
    # Test different URL formats
    working_url = test_url_formats()
    
    if working_url:
        print(f"\n🎉 SUCCESS! Working URL: {working_url}")
        print("\n📝 Update your .env file:")
        print(f"LITELLM_BASE_URL = {working_url}")
    else:
        # Test health endpoints to get more info
        test_health_endpoints()
        
        # Suggest solutions
        suggest_solutions()
        
        print("\n🔍 Next Steps:")
        print("1. Try the working URL format in your .env file")
        print("2. Test with the simple test script")
        print("3. Proceed with LangChain integration")


if __name__ == "__main__":
    main()