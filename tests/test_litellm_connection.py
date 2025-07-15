#!/usr/bin/env python3
"""
Test script to validate LiteLLM connection and LangChain integration
"""

import sys
import os
import asyncio
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
import requests
from pydantic import BaseModel
from typing import List

def test_environment_setup():
    """Test that environment variables are loaded correctly."""
    print("🔧 Testing environment setup...")
    
    # Load environment variables
    load_dotenv()
    
    litellm_api_key = os.getenv("LITELLM_API_KEY")
    litellm_base_url = os.getenv("LITELLM_BASE_URL")
    
    print(f"✅ LITELLM_API_KEY: {'Set' if litellm_api_key else 'Missing'}")
    print(f"✅ LITELLM_BASE_URL: {litellm_base_url if litellm_base_url else 'Missing'}")
    
    if not litellm_api_key or not litellm_base_url:
        print("❌ Missing required environment variables")
        return False
    
    return True, litellm_api_key, litellm_base_url


def test_direct_litellm_api(api_key, base_url):
    """Test direct LiteLLM API access."""
    print("\n🌐 Testing direct LiteLLM API access...")
    
    try:
        # Test models endpoint
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(f"{base_url}/models", headers=headers, timeout=10)
        
        print(f"🔍 Response Status: {response.status_code}")
        print(f"🔍 Response Headers: {dict(response.headers)}")
        print(f"🔍 Response Text: {response.text[:200]}...")
        
        if response.status_code == 200:
            try:
                models = response.json()
                if isinstance(models, dict) and "data" in models:
                    model_ids = [model["id"] for model in models["data"]]
                    print(f"✅ Found {len(model_ids)} available models")
                    print("📋 Available models:")
                    for model in model_ids[:10]:  # Show first 10
                        print(f"  - {model}")
                    return True, model_ids
                else:
                    print(f"❌ Unexpected response format: {models}")
                    return False, []
            except Exception as json_error:
                print(f"❌ JSON parsing error: {str(json_error)}")
                print(f"🔍 Raw response: {response.text}")
                return False, []
        else:
            print(f"❌ API request failed: {response.status_code} - {response.text}")
            return False, []
            
    except Exception as e:
        print(f"❌ Error testing direct API: {str(e)}")
        traceback.print_exc()
        return False, []


def test_langchain_litellm_import():
    """Test LangChain LiteLLM import."""
    print("\n📦 Testing LangChain LiteLLM import...")
    
    try:
        from langchain_litellm import ChatLiteLLM
        print("✅ ChatLiteLLM imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        traceback.print_exc()
        return False


def test_chatllm_initialization(model_name="claude-haiku-3.5"):
    """Test ChatLiteLLM initialization."""
    print(f"\n🤖 Testing ChatLiteLLM initialization with model: {model_name}")
    
    try:
        from langchain_litellm import ChatLiteLLM
        
        # Initialize ChatLiteLLM
        llm = ChatLiteLLM(
            model=model_name,
            temperature=0.1,
            max_tokens=100
        )
        
        print(f"✅ ChatLiteLLM initialized successfully")
        return True, llm
        
    except Exception as e:
        print(f"❌ Initialization error: {str(e)}")
        traceback.print_exc()
        return False, None


def test_basic_chat(llm):
    """Test basic chat functionality."""
    print("\n💬 Testing basic chat functionality...")
    
    try:
        from langchain_core.messages import HumanMessage
        
        # Test simple message
        message = HumanMessage(content="Hello! Please respond with just 'Hello back!'")
        response = llm.invoke([message])
        
        print(f"✅ Basic chat successful")
        print(f"📝 Response: {response.content}")
        return True, response.content
        
    except Exception as e:
        print(f"❌ Basic chat error: {str(e)}")
        traceback.print_exc()
        return False, None


async def test_async_chat(llm):
    """Test async chat functionality."""
    print("\n⚡ Testing async chat functionality...")
    
    try:
        from langchain_core.messages import HumanMessage
        
        # Test async message
        message = HumanMessage(content="Say 'Async works!' in exactly those words.")
        response = await llm.ainvoke([message])
        
        print(f"✅ Async chat successful")
        print(f"📝 Response: {response.content}")
        return True, response.content
        
    except Exception as e:
        print(f"❌ Async chat error: {str(e)}")
        traceback.print_exc()
        return False, None


def test_structured_output(llm):
    """Test structured output functionality."""
    print("\n🏗️ Testing structured output...")
    
    try:
        from langchain_core.messages import HumanMessage
        
        # Define a simple structure
        class PlanStep(BaseModel):
            step_number: int
            description: str
        
        class SimplePlan(BaseModel):
            steps: List[PlanStep]
        
        # Test if structured output works
        try:
            structured_llm = llm.with_structured_output(SimplePlan)
            
            message = HumanMessage(
                content="Create a simple 2-step plan for making coffee. Return as structured data."
            )
            
            response = structured_llm.invoke([message])
            
            print(f"✅ Structured output successful")
            print(f"📝 Response: {response}")
            return True, response
            
        except Exception as struct_error:
            print(f"⚠️ Structured output not supported: {str(struct_error)}")
            print("💡 Will need to handle structured output manually")
            return False, None
        
    except Exception as e:
        print(f"❌ Structured output test error: {str(e)}")
        traceback.print_exc()
        return False, None


def test_streaming(llm):
    """Test streaming functionality."""
    print("\n🌊 Testing streaming functionality...")
    
    try:
        from langchain_core.messages import HumanMessage
        
        # Test streaming
        message = HumanMessage(content="Count from 1 to 5, one number per line.")
        
        print("📡 Streaming response:")
        full_response = ""
        
        for chunk in llm.stream([message]):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content
        
        print(f"\n✅ Streaming successful")
        print(f"📝 Full response: {full_response}")
        return True, full_response
        
    except Exception as e:
        print(f"❌ Streaming error: {str(e)}")
        traceback.print_exc()
        return False, None


def test_multiple_models(api_key, base_url, available_models):
    """Test multiple models for performance comparison."""
    print("\n🔄 Testing multiple models...")
    
    test_models = []
    
    # Prioritize good models for our use case
    priority_models = [
        "claude-sonet-4",
        "claude-haiku-3.5", 
        "gpt-4.5-preview",
        "gpt-4.1-2025-04-14"
    ]
    
    # Find available priority models
    for model in priority_models:
        if model in available_models:
            test_models.append(model)
    
    # Add a few more if we have less than 3
    if len(test_models) < 3:
        for model in available_models[:5]:
            if model not in test_models:
                test_models.append(model)
                if len(test_models) >= 3:
                    break
    
    results = {}
    
    for model in test_models:
        print(f"\n🧪 Testing model: {model}")
        
        try:
            success, llm = test_chatllm_initialization(model)
            if success:
                start_time = datetime.now()
                success, response = test_basic_chat(llm)
                end_time = datetime.now()
                
                if success:
                    duration = (end_time - start_time).total_seconds()
                    results[model] = {
                        'success': True,
                        'response_time': duration,
                        'response_length': len(response) if response else 0
                    }
                    print(f"✅ {model}: {duration:.2f}s")
                else:
                    results[model] = {'success': False, 'error': 'Chat failed'}
            else:
                results[model] = {'success': False, 'error': 'Initialization failed'}
                
        except Exception as e:
            results[model] = {'success': False, 'error': str(e)}
            print(f"❌ {model}: {str(e)}")
    
    return results


def main():
    """Run all tests."""
    print("🚀 LiteLLM Connection and LangChain Integration Test")
    print("=" * 60)
    
    # Test environment setup
    env_result = test_environment_setup()
    if not env_result:
        print("❌ Environment setup failed. Exiting.")
        return 1
    
    success, api_key, base_url = env_result
    
    # Test direct API access
    api_success, available_models = test_direct_litellm_api(api_key, base_url)
    if not api_success:
        print("❌ Direct API access failed. Check your credentials.")
        return 1
    
    # Test LangChain import
    if not test_langchain_litellm_import():
        print("❌ LangChain LiteLLM import failed.")
        return 1
    
    # Test ChatLiteLLM initialization
    llm_success, llm = test_chatllm_initialization("claude-haiku-3.5")
    if not llm_success:
        print("❌ ChatLiteLLM initialization failed.")
        return 1
    
    # Test basic functionality
    tests = [
        ("Basic Chat", lambda: test_basic_chat(llm)),
        ("Async Chat", lambda: asyncio.run(test_async_chat(llm))),
        ("Structured Output", lambda: test_structured_output(llm)),
        ("Streaming", lambda: test_streaming(llm)),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result and result[0]:  # Check if test passed
                passed += 1
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
    
    # Test multiple models
    print(f"\n📊 Testing multiple models...")
    model_results = test_multiple_models(api_key, base_url, available_models)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} core tests passed")
    
    print("\n🏆 Model Performance:")
    for model, result in model_results.items():
        if result['success']:
            print(f"✅ {model}: {result['response_time']:.2f}s")
        else:
            print(f"❌ {model}: {result['error']}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    working_models = [m for m, r in model_results.items() if r['success']]
    if working_models:
        fastest = min(working_models, key=lambda m: model_results[m]['response_time'])
        print(f"🚀 Fastest model: {fastest} ({model_results[fastest]['response_time']:.2f}s)")
        
        if "claude-sonet-4" in working_models:
            print("🧠 Recommended for complex planning: claude-sonet-4")
        if "claude-haiku-3.5" in working_models:
            print("⚡ Recommended for fast execution: claude-haiku-3.5")
    
    if passed >= total * 0.75:  # 75% pass rate
        print("\n🎉 LiteLLM setup is working well!")
        return 0
    else:
        print("\n⚠️ Some issues detected. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())