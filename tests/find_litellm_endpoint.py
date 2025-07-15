#!/usr/bin/env python3
"""
Find the correct LiteLLM endpoint by testing different ports and paths
"""

import os
import sys
import requests
from dotenv import load_dotenv
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_ports():
    """Test different ports to find LiteLLM."""
    print("🔍 Scanning ports for LiteLLM...")
    
    load_dotenv()
    LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
    
    # Test different ports
    base_host = "https://anast.ita.chalmers.se"
    ports = [8083, 8084, 8085, 8080, 8081, 8082, 8000, 8001, 8008, 9000, 9001]
    
    headers = {
        "Authorization": f"Bearer {LITELLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    for port in ports:
        print(f"\n🔍 Testing port {port}...")
        
        # Test health endpoint
        try:
            health_url = f"{base_host}:{port}/health"
            response = requests.get(health_url, headers=headers, timeout=3, verify=False)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"   ✅ Health endpoint: {data}")
                    
                    # Test if it's LiteLLM by checking models
                    models_url = f"{base_host}:{port}/v1/models"
                    models_response = requests.get(models_url, headers=headers, timeout=3, verify=False)
                    
                    if models_response.status_code == 200:
                        try:
                            models_data = models_response.json()
                            if "data" in models_data and isinstance(models_data["data"], list):
                                print(f"   🎯 FOUND LITELLM at port {port}!")
                                print(f"   📋 Models available: {len(models_data['data'])}")
                                return f"{base_host}:{port}"
                        except:
                            pass
                            
                except json.JSONDecodeError:
                    print(f"   📄 HTML response (likely Open WebUI)")
                    
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Connection failed: {str(e)[:50]}...")
    
    return None


def test_subdirectories():
    """Test different subdirectories on the known working port."""
    print("\n🔍 Testing subdirectories on port 8083...")
    
    load_dotenv()
    LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
    
    base_url = "https://anast.ita.chalmers.se:8083"
    
    # Test subdirectories that might host LiteLLM
    subdirs = [
        "/litellm",
        "/api/litellm", 
        "/proxy",
        "/llm",
        "/models",
        "/ai",
        "/chat",
        "/completion",
        "/openai",
        "/anthropic"
    ]
    
    headers = {
        "Authorization": f"Bearer {LITELLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    for subdir in subdirs:
        print(f"\n🧪 Testing: {base_url}{subdir}")
        
        # Test health
        try:
            health_url = f"{base_url}{subdir}/health"
            response = requests.get(health_url, headers=headers, timeout=3, verify=False)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"   ✅ Health: {data}")
                    
                    # Test models
                    models_url = f"{base_url}{subdir}/v1/models"
                    models_response = requests.get(models_url, headers=headers, timeout=3, verify=False)
                    
                    if models_response.status_code == 200:
                        try:
                            models_data = models_response.json()
                            if "data" in models_data:
                                print(f"   🎯 FOUND LITELLM at {base_url}{subdir}")
                                return f"{base_url}{subdir}"
                        except:
                            pass
                            
                except json.JSONDecodeError:
                    print(f"   📄 HTML response")
                    
        except requests.exceptions.RequestException:
            pass
    
    return None


def test_alternative_endpoints():
    """Test alternative API endpoints that might work."""
    print("\n🔍 Testing alternative API endpoints...")
    
    load_dotenv()
    LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
    
    # Different base URLs to try
    alternative_hosts = [
        "https://anast.ita.chalmers.se:8083",
        "http://anast.ita.chalmers.se:8083",  # Try HTTP
        "https://anast.ita.chalmers.se",      # Try default HTTPS port
        "http://anast.ita.chalmers.se",       # Try default HTTP port
    ]
    
    endpoints = [
        "/api/v1/chat/completions",
        "/v1/chat/completions", 
        "/chat/completions",
        "/completions",
        "/api/completions",
        "/openai/v1/chat/completions"
    ]
    
    headers = {
        "Authorization": f"Bearer {LITELLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Simple test payload
    test_payload = {
        "model": "anthropic/claude-haiku-3.5",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5
    }
    
    for host in alternative_hosts:
        for endpoint in endpoints:
            url = f"{host}{endpoint}"
            print(f"\n🧪 Testing: {url}")
            
            try:
                response = requests.post(
                    url, 
                    json=test_payload, 
                    headers=headers, 
                    timeout=5, 
                    verify=False
                )
                
                print(f"   Status: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if "choices" in data:
                            print(f"   🎯 WORKING ENDPOINT: {url}")
                            return url
                    except:
                        pass
                elif response.status_code == 401:
                    print(f"   🔑 Authentication issue - but endpoint exists!")
                elif response.status_code == 404:
                    print(f"   ❌ Not found")
                elif response.status_code == 405:
                    print(f"   ❌ Method not allowed")
                else:
                    print(f"   ❓ Other response")
                    
            except requests.exceptions.RequestException as e:
                print(f"   ❌ Connection failed")
    
    return None


def main():
    """Run all tests to find LiteLLM endpoint."""
    print("🔍 LiteLLM Endpoint Discovery")
    print("=" * 50)
    
    print("Your notebook shows LiteLLM working, but our tests fail.")
    print("This suggests the endpoint might be different than expected.\n")
    
    # Test 1: Different ports
    litellm_port = test_ports()
    if litellm_port:
        print(f"\n🎉 FOUND! LiteLLM is running at: {litellm_port}")
        print(f"Update your .env file:")
        print(f"LITELLM_BASE_URL = {litellm_port}")
        return
    
    # Test 2: Subdirectories
    litellm_subdir = test_subdirectories() 
    if litellm_subdir:
        print(f"\n🎉 FOUND! LiteLLM is at: {litellm_subdir}")
        print(f"Update your .env file:")
        print(f"LITELLM_BASE_URL = {litellm_subdir}")
        return
    
    # Test 3: Alternative endpoints
    litellm_endpoint = test_alternative_endpoints()
    if litellm_endpoint:
        print(f"\n🎉 FOUND! Working endpoint: {litellm_endpoint}")
        return
    
    print("\n❌ Could not find LiteLLM endpoint automatically.")
    print("\n💡 Next steps:")
    print("1. Check your notebook - what URL does it actually use?")
    print("2. Ask your admin for the correct LiteLLM endpoint")
    print("3. Try running LiteLLM proxy locally for testing")
    print("4. Check if authentication is the issue")


if __name__ == "__main__":
    main()