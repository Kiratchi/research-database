from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_litellm import ChatLiteLLM
import os

# 1. Define a simple tool
def dummy_tool(input: str) -> str:
    return f"Received input: {input[:50]}..."

tools = [
    Tool(
        name="DummyTool",
        func=dummy_tool,
        description="Returns a string showing input preview"
    )
]

# 2. Load Claude via LiteLLM
llm = ChatLiteLLM(
    model="anthropic/claude-sonnet-4",
    api_key=os.getenv("LITELLM_API_KEY"),
    api_base=os.getenv("LITELLM_BASE_URL"),
    temperature=0,
    modify_params=True  # Helps prevent empty-message errors
)

# 3. Prompt with required ReAct variables
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful research assistant that uses tools to answer questions.\n\n"
     "Available tools:\n{tools}\n\n"
     "Tool names: {tool_names}\n\n"
     "When you use a tool, format your response as:\n"
     "Thought: I need to use a tool\n"
     "Action: <tool name>\n"
     "Action Input: <input>\n\n"
     "Then wait for the result.\n"
     "When you have enough information, respond with:\n"
     "Final Answer: <answer>"
    ),
    ("user", "{input}"),
    ("assistant", "{agent_scratchpad}")
])
# 4. Create the agent
runnable_agent: Runnable = create_react_agent(prompt=prompt, tools=tools, llm=llm)

# 5. Wrap with AgentExecutor
agent_executor = AgentExecutor(
    agent=runnable_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

# 6. Run it
if __name__ == "__main__":
    user_input = "Use DummyTool to test prompt flow with Claude."

    # Required inputs for the ReAct prompt
    tool_descriptions = "\n".join([f"{t.name}: {t.description}" for t in tools])
    tool_names = [t.name for t in tools]

    result = agent_executor.invoke({
        "input": user_input,
        "tools": tool_descriptions,
        "tool_names": tool_names
    })

    print("\n✅ Final Output:\n", result.get("output"))
