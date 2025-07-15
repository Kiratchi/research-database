#!/usr/bin/env python3
"""
Test LiteLLM direct connection following the notebook pattern
"""

import sys
import os
import asyncio
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
import litellm

def test_litellm_direct():
    """Test direct LiteLLM connection following notebook pattern."""
    print("🚀 Testing LiteLLM Direct Connection")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get values from environment
    LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
    LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
    
    print(f"✅ LITELLM_API_KEY: {'Set' if LITELLM_API_KEY else 'Missing'}")
    print(f"✅ LITELLM_BASE_URL: {LITELLM_BASE_URL}")
    
    # Validate environment variables
    if not LITELLM_API_KEY or not LITELLM_BASE_URL:
        print("❌ Missing API key or base URL in .env file")
        return False
    
    # Set LiteLLM configuration
    litellm.api_base = LITELLM_BASE_URL
    litellm.api_key = LITELLM_API_KEY
    
    print(f"\n🔧 LiteLLM Configuration:")
    print(f"   API Base: {litellm.api_base}")
    print(f"   API Key: {LITELLM_API_KEY[:20]}...")
    
    # Test basic completion
    print(f"\n💬 Testing basic completion...")
    
    try:
        response = litellm.completion(
            model="anthropic/claude-haiku-3.5",  # Use provider/model format
            messages=[{"role": "user", "content": "Hello! Please respond with exactly 'Hello back!'"}],
            api_key=LITELLM_API_KEY,
            api_base=LITELLM_BASE_URL
        )
        
        print(f"✅ Basic completion successful!")
        print(f"📝 Response: {response.choices[0].message.content}")
        
        return True, response
        
    except Exception as e:
        print(f"❌ Basic completion failed: {str(e)}")
        traceback.print_exc()
        return False, None


def test_streaming():
    """Test streaming completion."""
    print(f"\n🌊 Testing streaming completion...")
    
    try:
        LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
        LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
        
        response = litellm.completion(
            model="anthropic/claude-haiku-3.5",
            messages=[{"role": "user", "content": "Count from 1 to 3, one number per line."}],
            api_key=LITELLM_API_KEY,
            api_base=LITELLM_BASE_URL,
            stream=True
        )
        
        print(f"📡 Streaming response:")
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        
        print(f"\n✅ Streaming successful!")
        print(f"📝 Full response: {full_response}")
        
        return True, full_response
        
    except Exception as e:
        print(f"❌ Streaming failed: {str(e)}")
        traceback.print_exc()
        return False, None


def test_langchain_integration():
    """Test LangChain integration."""
    print(f"\n🔗 Testing LangChain integration...")
    
    try:
        from langchain_litellm import ChatLiteLLM
        from langchain_core.messages import HumanMessage
        
        # Initialize ChatLiteLLM
        llm = ChatLiteLLM(
            model="anthropic/claude-haiku-3.5",
            temperature=0.1,
            max_tokens=100
        )
        
        print(f"✅ ChatLiteLLM initialized")
        
        # Test basic message
        message = HumanMessage(content="Say 'LangChain works!' exactly.")
        response = llm.invoke([message])
        
        print(f"✅ LangChain integration successful!")
        print(f"📝 Response: {response.content}")
        
        return True, response.content
        
    except Exception as e:
        print(f"❌ LangChain integration failed: {str(e)}")
        traceback.print_exc()
        return False, None


async def test_langchain_async():
    """Test LangChain async integration."""
    print(f"\n⚡ Testing LangChain async integration...")
    
    try:
        from langchain_litellm import ChatLiteLLM
        from langchain_core.messages import HumanMessage
        
        # Initialize ChatLiteLLM
        llm = ChatLiteLLM(
            model="anthropic/claude-haiku-3.5",
            temperature=0.1,
            max_tokens=100
        )
        
        # Test async message
        message = HumanMessage(content="Say 'Async LangChain works!' exactly.")
        response = await llm.ainvoke([message])
        
        print(f"✅ Async LangChain integration successful!")
        print(f"📝 Response: {response.content}")
        
        return True, response.content
        
    except Exception as e:
        print(f"❌ Async LangChain integration failed: {str(e)}")
        traceback.print_exc()
        return False, None


def test_multiple_models():
    """Test multiple models."""
    print(f"\n🔄 Testing multiple models...")
    
    # Models to test based on your notebook
    test_models = [
        "anthropic/claude-haiku-3.5",
        "anthropic/claude-sonet-4", 
        "openai/gpt-4.5-preview"
    ]
    
    results = {}
    
    for model in test_models:
        print(f"\n🧪 Testing model: {model}")
        
        try:
            LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
            LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
            
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "Hello! Respond with just your model name."}],
                api_key=LITELLM_API_KEY,
                api_base=LITELLM_BASE_URL,
                max_tokens=50
            )
            
            response_text = response.choices[0].message.content
            results[model] = {'success': True, 'response': response_text}
            print(f"✅ {model}: {response_text}")
            
        except Exception as e:
            results[model] = {'success': False, 'error': str(e)}
            print(f"❌ {model}: {str(e)}")
    
    return results


def main():
    """Run all tests."""
    print("🧪 LiteLLM Direct Connection Test Suite")
    print("=" * 60)
    
    # Test direct connection
    basic_success, basic_response = test_litellm_direct()
    if not basic_success:
        print("❌ Basic connection failed. Stopping.")
        return 1
    
    # Test streaming
    streaming_success, streaming_response = test_streaming()
    
    # Test LangChain integration
    langchain_success, langchain_response = test_langchain_integration()
    
    # Test async LangChain
    async_success, async_response = asyncio.run(test_langchain_async())
    
    # Test multiple models
    model_results = test_multiple_models()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    
    tests = [
        ("Basic Connection", basic_success),
        ("Streaming", streaming_success),
        ("LangChain Integration", langchain_success),
        ("Async LangChain", async_success),
    ]
    
    passed = sum(1 for _, success in tests if success)
    total = len(tests)
    
    for test_name, success in tests:
        print(f"{'✅' if success else '❌'} {test_name}")
    
    print(f"\n🏆 Core Tests: {passed}/{total} passed")
    
    # Model results
    print(f"\n🤖 Model Results:")
    working_models = []
    for model, result in model_results.items():
        if result['success']:
            print(f"✅ {model}: Working")
            working_models.append(model)
        else:
            print(f"❌ {model}: {result['error']}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if working_models:
        print(f"🎯 Working models: {', '.join(working_models)}")
        if "claude-sonet-4" in working_models:
            print("🧠 Use claude-sonet-4 for complex reasoning (planner)")
        if "claude-haiku-3.5" in working_models:
            print("⚡ Use claude-haiku-3.5 for fast responses (executor)")
    
    if passed >= 3 and working_models:
        print("\n🎉 LiteLLM is working perfectly! Ready to integrate.")
        return 0
    else:
        print("\n⚠️ Some issues found. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())