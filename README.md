# Research Publications Chat Agent

An intelligent conversational AI system for exploring and analyzing research publications from Chalmers University of Technology's research database ([chalmers.research.se](https://chalmers.research.se)).

## 🎯 Overview

This project provides a natural language interface to search, analyze, and explore academic publications, enabling researchers and students to easily discover information about:

- **Authors and researchers** - Find publications, research areas, and collaboration patterns
- **Research topics** - Discover papers, trends, and developments in specific fields  
- **Publication statistics** - Get counts, trends, and analytical insights
- **Institutional research** - Explore Chalmers' research output and impact

## ✨ Key Features

### 🧠 **Intelligent Query Processing**
- **Natural Language Understanding** - Ask questions in plain English
- **Context-Aware Conversations** - Follow-up questions remember previous context
- **Multi-Turn Interactions** - Build complex queries through conversation

### 🔍 **Advanced Research Capabilities**
- **Author Analysis** - Comprehensive profiles including publications, trends, and collaborations
- **Topic Exploration** - Deep dives into research fields and themes
- **Statistical Insights** - Publication counts, temporal trends, and distributions
- **Smart Search** - Handles ambiguous queries and provides relevant suggestions

### 🛠️ **Research Tools Integration**
- **Publication Search** - Find papers by title, topic, or keywords
- **Author Lookup** - Search by researcher names with fuzzy matching
- **Statistical Analysis** - Field distributions, publication trends, and metrics
- **Detail Retrieval** - Full publication metadata and abstracts

### 🎨 **User Experience**
- **Web Interface** - Clean, responsive chat-style interface
- **Real-time Processing** - Live feedback during research operations
- **Session Memory** - Maintains conversation context across queries
- **Error Handling** - Graceful handling of edge cases and limitations

## 🏗️ Architecture

### **Core Components**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Frontend  │────│  Flask Server    │────│  Agent Manager  │
│   (HTML/JS)     │    │  (HTTP API)      │    │  (Coordinator)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                         │
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Memory Manager   │    │ Research Agent  │
                       │ (Conversation)   │    │ (LangGraph)     │
                       └──────────────────┘    └─────────────────┘
                                                         │
                                              ┌─────────────────┐
                                              │ Elasticsearch   │
                                              │ (Publications)  │
                                              └─────────────────┘
```

### **Technology Stack**

- **Backend Framework**: Flask with async support
- **AI Orchestration**: LangGraph for complex workflow management
- **Language Models**: Claude (Anthropic) via LiteLLM proxy
- **Database**: Elasticsearch with Chalmers research publications
- **Memory Management**: LangChain conversation buffers
- **Monitoring**: LangSmith for tracing and debugging
- **Frontend**: Vanilla HTML/CSS/JavaScript with modern UI

## 🚀 Getting Started

### **Prerequisites**

- Python 3.11+
- Access to Elasticsearch database
- LiteLLM proxy setup with Claude models
- Environment variables configured

### **Installation**

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd db_chat
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configurations
   ```

### **Environment Configuration**

Create a `.env` file with the following variables:

```env
# Elasticsearch Configuration
ES_HOST=your-elasticsearch-host
ES_USER=your-username  
ES_PASS=your-password

# LiteLLM Configuration
LITELLM_API_KEY=your-litellm-api-key
LITELLM_BASE_URL=https://your-litellm-proxy.com/v1

# LangSmith Tracing (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=research-publications-agent

# Flask Configuration
FLASK_HOST=localhost
FLASK_PORT=5000
FLASK_DEBUG=true
```

### **Running the Application**

```bash
python app.py
```

Visit `http://localhost:5000` to access the web interface.

## 💡 Usage Examples

### **Basic Queries**
```
"Who is Per-Olof Arnäs?"
"How many papers has Maria Andersson published?"
"Find papers about machine learning"
```

### **Advanced Analysis**
```
"Show me the publication trends for artificial intelligence research at Chalmers"
"Who are the most cited authors in sustainable technology?"
"Compare the research output between different departments"
```

### **Follow-up Conversations**
```
User: "Who is John Smith?"
Agent: [Provides author information]
User: "Show me his most recent papers"
Agent: [Shows recent publications with context]
User: "What about his collaborations?"
Agent: [Analyzes collaboration patterns]
```

## 🔧 API Endpoints

### **Chat Interface**
- `POST /chat/respond` - Process user queries
- `POST /chat/clear-memory` - Clear conversation history
- `GET /chat/session-info/<session_id>` - Get session information

### **System Management**
- `GET /status` - System status and health
- `GET /health` - Detailed health check
- `GET /admin/memory-stats` - Memory usage statistics

## ⚙️ Configuration

### **Model Configuration**
The system uses Claude models through LiteLLM:
- **Main LLM**: `claude-sonnet-4` (for complex reasoning)
- **Replanner**: `claude-haiku-3.5` (for quick decisions)

### **Memory Settings**
- **Type**: Buffer window memory
- **Capacity**: Last 10 messages (5 Q&A pairs)
- **Cleanup**: Automatic cleanup after 1 hour of inactivity

### **Tool Limitations**
- **search_publications**: Max 15 results per call
- **search_by_author**: Max 20 results per call  
- **get_publication_details**: High context usage, use sparingly
- **Statistical tools**: No limits (aggregation-based)

## 🧪 Testing

### **Memory Inspection**
```bash
python check_models.py  # Check available LiteLLM models
curl http://localhost:5000/admin/memory-stats  # Check memory usage
```

### **Streaming Test**
```bash
python streaming_test.py server  # Test streaming capabilities
```

## 📁 Project Structure

```
db_chat/
├── app.py                      # Flask application entry point
├── check_models.py            # LiteLLM model inspection utility
├── streaming_test.py          # Streaming capabilities test
├── index.html                 # Web frontend interface
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── research_agent/
│   ├── core/
│   │   ├── agent_manager.py   # Main coordinator
│   │   ├── memory_manager.py  # Conversation memory
│   │   ├── workflow.py        # LangGraph research workflow
│   │   ├── state.py          # State management
│   │   ├── models.py         # Pydantic models
│   │   └── prompt_loader.py  # Prompt management
│   ├── tools/
│   │   └── elasticsearch_tools.py  # Database tools
│   ├── prompts/
│   │   ├── batched_planning_prompt.txt
│   │   ├── memory_aware_execution_prompt.txt
│   │   ├── executor_prompt.txt
│   │   └── replanner_prompt.txt
│   └── utils/
│       ├── config.py         # Configuration utilities
│       └── logging.py        # Logging setup
```

## 🛡️ Production Considerations

### **Proxy Setup**
Ensure your proxy server supports streaming:
```nginx
# Nginx configuration for streaming
proxy_buffering off;
proxy_cache off;
proxy_read_timeout 300s;
```

### **Security**
- Environment variables for sensitive data
- No hardcoded credentials
- Session isolation
- Input validation

### **Performance**
- Connection pooling for Elasticsearch
- Memory cleanup for long-running sessions
- Efficient tool selection based on query type
- Caching for repeated statistical queries

## 🐛 Troubleshooting

### **Common Issues**

**LiteLLM Model Errors**
```bash
python check_models.py  # Verify available models
```

**Memory Issues**
```bash
curl http://localhost:5000/admin/memory-stats  # Check memory usage
```

**Elasticsearch Connection**
```bash
curl http://localhost:5000/health  # Check system health
```

**Streaming Problems**
```bash
python streaming_test.py  # Test proxy streaming support
```

### **Debug Mode**
Enable detailed logging by setting `FLASK_DEBUG=true` in your environment.

