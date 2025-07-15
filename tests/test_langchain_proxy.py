#!/usr/bin/env python3
"""
Test LangChain with LiteLLM proxy configuration
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_langchain_proxy_config():
    """Test LangChain with proper proxy configuration."""
    print("🔗 Testing LangChain with LiteLLM Proxy Configuration")
    print("=" * 60)
    
    load_dotenv()
    
    LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
    LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
    
    print(f"✅ API Key: {LITELLM_API_KEY[:20]}...")
    print(f"✅ Base URL: {LITELLM_BASE_URL}")
    
    try:
        from langchain_litellm import ChatLiteLLM
        from langchain_core.messages import HumanMessage
        
        # Configure ChatLiteLLM with proxy settings
        llm = ChatLiteLLM(
            model="anthropic/claude-haiku-3.5",
            api_key=LITELLM_API_KEY,
            api_base=LITELLM_BASE_URL,
            temperature=0.1,
            max_tokens=100
        )
        
        print(f"✅ ChatLiteLLM initialized with proxy config")
        
        # Test basic message
        message = HumanMessage(content="Say 'LangChain proxy works!' exactly.")
        response = llm.invoke([message])
        
        print(f"✅ LangChain proxy integration successful!")
        print(f"📝 Response: {response.content}")
        
        return True, response.content
        
    except Exception as e:
        print(f"❌ LangChain proxy integration failed: {str(e)}")
        return False, None


async def test_langchain_async_proxy():
    """Test async LangChain with proxy configuration."""
    print(f"\n⚡ Testing async LangChain with proxy...")
    
    load_dotenv()
    
    LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
    LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
    
    try:
        from langchain_litellm import ChatLiteLLM
        from langchain_core.messages import HumanMessage
        
        # Configure with explicit proxy settings
        llm = ChatLiteLLM(
            model="anthropic/claude-haiku-3.5",
            api_key=LITELLM_API_KEY,
            api_base=LITELLM_BASE_URL,
            temperature=0.1,
            max_tokens=100
        )
        
        # Test async message
        message = HumanMessage(content="Say 'Async proxy works!' exactly.")
        response = await llm.ainvoke([message])
        
        print(f"✅ Async LangChain proxy integration successful!")
        print(f"📝 Response: {response.content}")
        
        return True, response.content
        
    except Exception as e:
        print(f"❌ Async LangChain proxy integration failed: {str(e)}")
        return False, None


def test_structured_output_proxy():
    """Test structured output with proxy configuration."""
    print(f"\n🏗️ Testing structured output with proxy...")
    
    load_dotenv()
    
    LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
    LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
    
    try:
        from langchain_litellm import ChatLiteLLM
        from langchain_core.messages import HumanMessage
        from pydantic import BaseModel
        from typing import List
        
        # Configure with proxy settings
        llm = ChatLiteLLM(
            model="anthropic/claude-haiku-3.5",
            api_key=LITELLM_API_KEY,
            api_base=LITELLM_BASE_URL,
            temperature=0.1,
            max_tokens=200
        )
        
        # Define a simple structure
        class PlanStep(BaseModel):
            step_number: int
            description: str
        
        class SimplePlan(BaseModel):
            steps: List[PlanStep]
        
        # Test structured output
        try:
            structured_llm = llm.with_structured_output(SimplePlan)
            
            message = HumanMessage(
                content="Create a simple 2-step plan for making coffee. Return as structured data."
            )
            
            response = structured_llm.invoke([message])
            
            print(f"✅ Structured output with proxy successful!")
            print(f"📝 Response: {response}")
            return True, response
            
        except Exception as struct_error:
            print(f"⚠️ Structured output not supported: {str(struct_error)}")
            print("💡 Will need to handle structured output manually")
            return False, None
        
    except Exception as e:
        print(f"❌ Structured output test failed: {str(e)}")
        return False, None


def test_streaming_proxy():
    """Test streaming with proxy configuration."""
    print(f"\n🌊 Testing streaming with proxy...")
    
    load_dotenv()
    
    LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
    LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
    
    try:
        from langchain_litellm import ChatLiteLLM
        from langchain_core.messages import HumanMessage
        
        # Configure with proxy settings
        llm = ChatLiteLLM(
            model="anthropic/claude-haiku-3.5",
            api_key=LITELLM_API_KEY,
            api_base=LITELLM_BASE_URL,
            temperature=0.1,
            max_tokens=100
        )
        
        # Test streaming
        message = HumanMessage(content="Count from 1 to 3, one number per line.")
        
        print("📡 Streaming response:")
        full_response = ""
        
        for chunk in llm.stream([message]):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content
        
        print(f"\n✅ Streaming with proxy successful!")
        print(f"📝 Full response: {full_response}")
        
        return True, full_response
        
    except Exception as e:
        print(f"❌ Streaming with proxy failed: {str(e)}")
        return False, None


def test_claude_sonet_4():
    """Test with Claude Sonet 4 model."""
    print(f"\n🧠 Testing Claude Sonet 4 (complex model)...")
    
    load_dotenv()
    
    LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
    LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
    
    try:
        from langchain_litellm import ChatLiteLLM
        from langchain_core.messages import HumanMessage
        
        # Configure with Claude Sonet 4
        llm = ChatLiteLLM(
            model="anthropic/claude-sonet-4",
            api_key=LITELLM_API_KEY,
            api_base=LITELLM_BASE_URL,
            temperature=0.1,
            max_tokens=150
        )
        
        # Test with a planning query
        message = HumanMessage(
            content="Create a 3-step plan for researching a scientific topic. Be concise."
        )
        
        response = llm.invoke([message])
        
        print(f"✅ Claude Sonet 4 working!")
        print(f"📝 Response: {response.content}")
        
        return True, response.content
        
    except Exception as e:
        print(f"❌ Claude Sonet 4 failed: {str(e)}")
        return False, None


def main():
    """Run all proxy configuration tests."""
    print("🧪 LangChain LiteLLM Proxy Integration Test Suite")
    print("=" * 60)
    
    print("Testing LangChain with explicit proxy configuration...")
    print("This should bypass the provider API key requirements.\n")
    
    # Test sync
    sync_success, sync_response = test_langchain_proxy_config()
    
    # Test async
    async_success, async_response = asyncio.run(test_langchain_async_proxy())
    
    # Test structured output
    struct_success, struct_response = test_structured_output_proxy()
    
    # Test streaming
    stream_success, stream_response = test_streaming_proxy()
    
    # Test Claude Sonet 4
    sonet_success, sonet_response = test_claude_sonet_4()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Proxy Configuration Test Results:")
    
    tests = [
        ("Basic Sync", sync_success),
        ("Async", async_success),
        ("Structured Output", struct_success),
        ("Streaming", stream_success),
        ("Claude Sonet 4", sonet_success),
    ]
    
    passed = sum(1 for _, success in tests if success)
    total = len(tests)
    
    for test_name, success in tests:
        print(f"{'✅' if success else '❌'} {test_name}")
    
    print(f"\n🏆 Results: {passed}/{total} tests passed")
    
    if passed >= 4:  # Allow structured output to fail
        print("\n🎉 LangChain proxy integration is working!")
        print("✅ Ready to integrate with our research agent workflow.")
        return 0
    else:
        print("\n⚠️ Some proxy configuration issues remain.")
        return 1


if __name__ == "__main__":
    sys.exit(main())