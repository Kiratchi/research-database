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

1. **`search_publications`**: Full-text search across Title, Abstract, Persons.PersonData.DisplayName, and Keywords fields
2. **`search_by_author`**: Search publications by author name in Persons.PersonData.DisplayName field (exact, partial, fuzzy strategies)
3. **`get_field_statistics`**: Get statistics for specific fields (Year, Persons.PersonData.DisplayName, Source, PublicationType)
4. **`get_publication_details`**: Get detailed information about a publication using its ID
5. **`get_database_summary`**: Get overview of database contents with totals and statistics

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

### ✅ **Working Features**
- **Plan-and-Execute Workflow**: LangGraph-based multi-step planning and execution
- **Streaming Chat Interface**: Real-time updates with step-by-step progress
- **Elasticsearch Integration**: Proper field mapping and search functionality
- **Error Handling**: Comprehensive error recovery and debugging tools
- **Clean Response Formatting**: User-friendly output instead of raw JSON
- **Tool Documentation**: Enhanced agent knowledge of available tools

### ⚠️ **Known Limitations**
- **Single-Message Only**: No conversation memory between messages
- **No Pagination**: Limited to 10 results per tool call
- **Premature Completion**: Agent sometimes completes tasks before fully addressing user requests
- **No ORCID Extraction**: Publication details don't contain ORCID information

### 🔧 **Recent Improvements**
- **Fixed Workflow Routing**: Removed forced replan cycles, agent now completes tasks efficiently
- **Enhanced Output Formatting**: Users see clean answers instead of raw state objects
- **Corrected Field Names**: Updated search tools to use actual Elasticsearch schema
- **Better Tool Descriptions**: Added parameter specifications and output format details

## Technical Implementation

### LangGraph Workflow

The agent uses a sophisticated plan-and-execute pattern:

1. **Planning**: Generate multi-step research plan
2. **Execution**: Execute tools and gather results
3. **Completion Detection**: Determine when task is complete
4. **Response Formatting**: Format results for user presentation

### Elasticsearch Schema

The system works with the following field structure:
- **Authors**: `Persons.PersonData.DisplayName` (not `authors`)
- **Journal**: `Source` (not `journal`)
- **Type**: `PublicationType` (not `publication_type`)
- **Year**: `Year`
- **Title**: `Title`
- **Abstract**: `Abstract`

### Tool Integration

Each tool provides:
- **Parameter specifications**: Clear input requirements
- **Output format**: JSON structure documentation
- **Error handling**: Graceful failure modes

## Development

### Project Structure
```
es_workspace/
├── src/research_agent/           # Core research agent
│   ├── core/                     # Workflow and models
│   │   ├── workflow.py           # LangGraph plan-and-execute
│   │   ├── models.py             # Pydantic models
│   │   └── state.py              # State management
│   ├── tools/                    # Elasticsearch tools
│   │   └── elasticsearch_tools.py
│   └── agents/                   # Legacy agent components
├── streamlit_app.py              # Main Streamlit application
├── streamlit_agent.py            # Streamlit-agent bridge
├── tests/                        # Test files
│   ├── test_workflow_routing.py  # Workflow tests
│   ├── test_fixed_search_tools.py # Tool tests
│   └── test_pydantic_fix.py      # Model tests
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

# Test search tools functionality
python tests/test_anna_dubois_search.py
```

### Debug Mode

Enable debug mode in the Streamlit app to see:
- Raw event streams from LangGraph
- Tool execution details
- Error tracebacks
- System status information

## Troubleshooting

### Common Issues

1. **Connection Error**: Check your `.env` file and Elasticsearch credentials
2. **LiteLLM Error**: Verify `LITELLM_API_KEY` and `LITELLM_BASE_URL`
3. **Import Error**: Ensure all dependencies are installed in the virtual environment
4. **Tool Errors**: Check the debug panel for detailed error information
5. **Empty Results**: Verify field names match Elasticsearch schema

### Debug Information

The Streamlit app provides comprehensive debug information:
- Event stream processing details
- Tool execution results and timing
- Error tracebacks with context
- Agent state and workflow progress

## Implementation History

### Major Milestones
- **Initial Implementation**: LangGraph plan-and-execute workflow
- **Elasticsearch Integration**: Connected search tools with proper schema
- **Streamlit Interface**: Interactive chat with streaming updates
- **Workflow Fixes**: Corrected routing and completion logic
- **Tool Enhancement**: Updated descriptions and field mappings

### Key Fixes Applied
1. **Workflow Routing**: Removed forced replan cycles
2. **Output Formatting**: Clean user responses instead of raw JSON
3. **Field Mapping**: Corrected Elasticsearch field names
4. **Tool Documentation**: Enhanced agent knowledge of capabilities
5. **Async/Sync Issues**: Fixed LiteLLM compatibility

## Future Enhancements

### Short-term (Next Version)
1. **Conversation Memory**: Support for multi-turn conversations
2. **Pagination Support**: Handle large result sets effectively
3. **ORCID Integration**: Extract author identifiers from publications
4. **Better Completion Logic**: More accurate task completion detection

### Medium-term
1. **Advanced Search Features**: More sophisticated query capabilities
2. **Result Export**: Save conversations and research results
3. **Performance Optimization**: Caching and query optimization
4. **Enhanced Error Recovery**: Better handling of edge cases

### Long-term
1. **Multi-database Support**: Extend beyond Elasticsearch
2. **Advanced Analytics**: Visualization and trend analysis
3. **API Interface**: REST API for programmatic access
4. **User Management**: Authentication and session persistence

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is for research and educational purposes.

---

**Current Status**: ✅ **Production-Ready MVP with Enhanced Tool Integration**  
**Repository**: https://github.com/wikefjol/db_chat.git  
**Last Updated**: Recent tool documentation and workflow routing improvements