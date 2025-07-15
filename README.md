# Research Publications Chat Agent

A production-ready research agent system for querying publication databases using Elasticsearch, with LangGraph-based plan-and-execute workflows and interactive Streamlit interface.

## 🎯 **Project Overview**

This project provides an AI-powered research agent that enables natural language querying of publication databases. Built on LangGraph's plan-and-execute pattern, it combines powerful Elasticsearch tools with modern AI agents to deliver an intuitive research experience.

### **Key Features**
- 🔍 **Natural Language Queries**: Ask questions in plain English
- 🤖 **AI-Powered Planning**: LangGraph-based agents create and execute research plans
- 📊 **Interactive UI**: Streamlit-based chat interface with streaming responses
- 🔧 **Production-Ready**: Comprehensive error handling and debugging tools
- 🚀 **Performance Optimized**: Sub-2-second first response times
- 🌐 **Unicode Support**: Full international character support
- 📈 **Real-time Updates**: Streaming execution with step-by-step progress

## Architecture

### Current Implementation
```
User Query → LangGraph Workflow → Plan Generation → Tool Execution → Response
                                        ↓
Streamlit UI ← Response Formatting ← Result Processing ← Elasticsearch Tools
```

### Core Components

- **`src/research_agent/core/workflow.py`**: LangGraph plan-and-execute workflow
- **`src/research_agent/tools/elasticsearch_tools.py`**: Elasticsearch search tools
- **`src/research_agent/core/models.py`**: Pydantic models for structured output
- **`streamlit_agent.py`**: Bridge between Streamlit and research agent
- **`streamlit_app.py`**: Chat interface with streaming updates

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Elasticsearch Configuration
ES_HOST=your-elasticsearch-host
ES_USER=your-username
ES_PASS=your-password

# LiteLLM Configuration (for LLM access)
LITELLM_API_KEY=your-api-key
LITELLM_BASE_URL=https://your-litellm-endpoint
```

### 3. Run the Application

```bash
# Activate virtual environment
source venv/bin/activate

# Run Streamlit app
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## Available Tools

The research agent has access to these Elasticsearch tools:

1. **`search_publications`**: Full-text search across publications
2. **`search_by_author`**: Search publications by author name (exact, partial, fuzzy)
3. **`get_field_statistics`**: Get statistics for specific fields
4. **`get_publication_details`**: Get detailed information about a publication
5. **`get_database_summary`**: Get overview of database contents

## Test Questions

Try these example queries to test the agent:

### Author Searches
- "How many publications has `<author_name>` published?"
- "List all publications by `<author_name>`"
- "What is the ORCID of `<author_name>`?"

### Topic Searches
- "Find publications about `<topic>` from `<year>`"
- "Find publications containing the keyword `<keyword>`"

### Statistical Queries
- "What is the total number of publications in the database?"
- "What are the most common publication types?"
- "Which years have the most publications?"

### Comparative Analysis
- "Compare publication counts between `<author1>` and `<author2>`"

## Current Status

### ✅ Working Features
- Plan-and-execute workflow with LangGraph
- Streaming chat interface
- Elasticsearch integration with proper field mapping
- Error handling and debugging tools
- Clean response formatting

### ⚠️ Known Limitations
- **Single-message only**: No conversation memory between messages
- **No pagination**: Limited to 10 results per tool call
- **Tool knowledge gaps**: Agent has minimal information about tool capabilities

### 🔧 Planned Improvements
1. **Enhanced Tool Documentation**: Improve agent knowledge of tool parameters and output formats
2. **Conversation Memory**: Add support for multi-turn conversations
3. **Pagination Support**: Handle large result sets
4. **Better Error Handling**: More graceful handling of edge cases

## Development

### Project Structure
```
es_workspace/
├── src/research_agent/           # Core research agent
│   ├── core/                     # Workflow and models
│   ├── tools/                    # Elasticsearch tools
│   └── agents/                   # Legacy agent components
├── streamlit_app.py              # Main Streamlit application
├── streamlit_agent.py            # Streamlit-agent bridge
├── tests/                        # Test files
├── docs/                         # Documentation
├── examples/                     # Example notebooks
├── bin/                          # Deprecated files
└── requirements.txt              # Dependencies
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Test specific components
python -m pytest tests/test_workflow_routing.py -v
python -m pytest tests/test_fixed_search_tools.py -v

# Test Streamlit integration
python tests/test_streamlit_integration.py
```

### Debug Mode

Enable debug mode in the Streamlit app to see:
- Raw event streams
- Response processing
- Error details
- System information

## Troubleshooting

### Common Issues

1. **Connection Error**: Check your `.env` file and Elasticsearch credentials
2. **LiteLLM Error**: Verify LITELLM_API_KEY and LITELLM_BASE_URL
3. **Import Error**: Ensure all dependencies are installed in the virtual environment
4. **Tool Errors**: Check the debug panel for detailed error information

### Debug Information

The Streamlit app provides comprehensive debug information:
- Event stream processing
- Tool execution results
- Error tracebacks
- System status

## Next Steps

### Short-term (Current Focus)
1. **Tool Documentation Enhancement**: Improve agent knowledge of available tools
2. **Output Format Improvements**: Better response formatting for different query types
3. **Completion Logic Refinement**: More accurate task completion detection

### Medium-term
1. **Pagination Support**: Handle large result sets effectively
2. **Conversation Memory**: Multi-turn conversation support
3. **Advanced Search Features**: More sophisticated query capabilities

### Long-term
1. **Multi-database Support**: Extend beyond Elasticsearch
2. **Advanced Analytics**: Visualization and trend analysis
3. **API Interface**: REST API for programmatic access

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is for research and educational purposes.