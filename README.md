# Research Publications Chat Agent

A conversational AI system for searching and analyzing research publications from Chalmers University of Technology's research database ([chalmers.research.se](https://chalmers.research.se)).

## Overview

This application provides a natural language interface to the Chalmers research database, enabling users to query information about authors, publications, research projects, and institutional data through conversational interactions.

**Primary capabilities:**
- Author and researcher information lookup
- Publication search and analysis
- Research project and funding information
- Statistical analysis of research output
- Multi-turn conversational queries with context retention

## Technical Architecture

**Backend Framework:** Quart (async Flask) for high-performance asynchronous request handling  
**AI Workflow:** LangGraph implementing ReAct (Reasoning and Acting) pattern  
**Language Model:** Claude 3.7 Sonnet via LiteLLM proxy  
**Database:** Elasticsearch containing Chalmers research data  
**Memory Management:** Global singleton pattern with automatic session cleanup  
**Frontend:** Responsive web interface with markdown rendering support  

### System Components

```
Web Interface → Quart Server → Agent Manager → Research Agent (ReAct) → Research Tools → Elasticsearch
                     ↓
              Memory Singleton (Global State Management)
```

### ReAct Workflow Process

1. **Query Analysis:** Parse user input and determine research intent
2. **Tool Selection:** Choose appropriate research tools based on query type
3. **Information Retrieval:** Execute searches using selected tools
4. **Result Processing:** Analyze and synthesize retrieved data
5. **Response Generation:** Formulate comprehensive answer with supporting evidence

## Installation and Setup

### Prerequisites

- Python 3.11+
- Access to Elasticsearch instance with Chalmers research data
- LiteLLM proxy configured with Claude model access
- Environment variables for authentication

### Installation Steps

1. Clone repository and create virtual environment:
   ```bash
   git clone <repository-url>
   cd db_chat
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables in `.env` file:
   ```env
   # Elasticsearch Configuration
   ES_HOST=your-elasticsearch-host
   ES_USER=your-username
   ES_PASS=your-password
   
   # LiteLLM Configuration
   LITELLM_API_KEY=your-api-key
   LITELLM_BASE_URL=your-proxy-url
   
   # Application Settings
   FLASK_HOST=localhost
   FLASK_PORT=5000
   FLASK_DEBUG=true
   
   # Optional: LangSmith Tracing
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your-langsmith-key
   ```

4. Run application:
   ```bash
   python app.py
   ```

5. Access web interface at `http://localhost:5000`

## Research Tools and Capabilities

The system includes specialized tools for different types of research queries:

**Publication Tools:**
- Search publications by title, abstract, or keywords
- Retrieve detailed publication metadata
- Analyze publication trends and statistics

**Person/Author Tools:**
- Search researchers by name with fuzzy matching
- Retrieve author profiles and publication lists
- Analyze collaboration networks

**Project Tools:**
- Search research projects and grants
- Analyze funding sources and amounts
- Track project timelines and outcomes

**Organization Tools:**
- Map departments and research groups
- Analyze institutional research output
- Compare organizational metrics

## API Endpoints

### Chat Interface
- `POST /chat/respond` - Process natural language queries
- `POST /chat/clear-memory` - Clear conversation history
- `GET /chat/session-info/<session_id>` - Retrieve session information

### System Status
- `GET /health` - System health check
- `GET /status` - Detailed system status including database connectivity
- `GET /admin/tools-info` - Available research tools information

### Example API Request
```bash
curl -X POST http://localhost:5000/chat/respond \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find publications by Maria Andersson",
    "session_id": "unique-session-id"
  }'
```

## Configuration Details

### Memory Management
- **Session Capacity:** 10 messages (5 question-answer pairs) per session
- **Context Window:** 2000 characters with intelligent truncation
- **Cleanup Policy:** Automatic removal of inactive sessions after 1 hour
- **Storage Pattern:** Global singleton ensures memory persistence across requests

### Model Configuration
- **Primary Model:** claude-sonnet-3.7 for reasoning and response generation
- **Recursion Limit:** 50 workflow steps maximum per query
- **Timeout:** 10 minutes per query processing

### Database Limitations
- **Scope:** Research from Chalmers University affiliations only
- **Coverage:** Comprehensive from 2012+ for projects, 2009+ for theses
- **Content:** Published research outputs, excludes pre-prints and unpublished work
- **Access:** Metadata available, full-text may have restrictions

## Project Structure

```
db_chat/
├── app.py                          # Quart application entry point
├── index.html                      # Web frontend
├── requirements.txt                # Python dependencies
├── research_agent/
│   ├── core/
│   │   ├── agent_manager.py       # Query coordination and async processing
│   │   ├── memory_singleton.py    # Global memory management
│   │   └── workflow.py            # ReAct workflow implementation
│   └── tools/
│       ├── search/
│       │   ├── publications/       # Publication search tools
│       │   ├── persons/           # Author/researcher tools
│       │   ├── projects/          # Research project tools
│       │   ├── organizations/     # Department/institution tools
│       │   └── base/              # Core Elasticsearch integration
│       └── utils/                 # Text processing and utilities
```

## Usage Examples

**Basic Queries:**
- "Who is [Researcher]?"
- "Find papers about machine learning"
- "Show me recent publications from the computer science department"

**Advanced Analysis:**
- "Compare publication output between departments"
- "Find researchers working on sustainable technology"
- "Analyze collaboration patterns in engineering research"

**Contextual Follow-ups:**
```
User: "Find information about renewable energy research"
System: [Provides overview and key researchers]
User: "Who has published the most papers in this area?"
System: [Analyzes publication counts with context from previous query]
```
## Troubleshooting

**Common Issues:**

*Elasticsearch Connection Failures:*
- Verify host URL, username, and password in environment variables
- Check network connectivity to Elasticsearch instance
- Confirm database contains expected research data

*LiteLLM/Model Access Issues:*
- Validate API key and proxy URL configuration
- Check model availability and rate limits
- Review LangSmith traces for detailed error information

*Memory or Performance Issues:*
- Monitor session count via `/status` endpoint
- Check memory cleanup frequency in logs
- Verify query timeout settings

**Debug Configuration:**
```env
FLASK_DEBUG=true
LANGCHAIN_VERBOSE=true
LANGCHAIN_TRACING_V2=true
```

## Development Notes

The system uses modern async patterns throughout. When adding new features:
- Use `async/await` for all I/O operations
- Follow the existing tool structure in `research_agent/tools/`
- Implement proper error handling for database operations
- Add appropriate logging for debugging and monitoring

New research tools should extend the base Elasticsearch tool and register themselves in the appropriate category (publications, persons, projects, or organizations).

## Dependencies

See `requirements.txt` for complete dependency list. Key dependencies include:
- Quart and quart-cors for async web framework
- LangChain and LangGraph for AI workflow orchestration
- Elasticsearch client for database connectivity
- LangChain-LiteLLM for model integration