# TDD Plan: Elasticsearch Chat Agent with Streamlit

## Phase 1: Core Infrastructure (2-3 days)
**Goal**: Set up testing framework and core chat components

### Tests to Write First:
1. **Chat Message Parsing Tests**
   - Test natural language query parsing
   - Test intent classification (search, count, list, stats)
   - Test parameter extraction (names, years, keywords)

2. **Query Builder Tests**
   - Test conversion from parsed intent to ES queries
   - Test filter combination logic
   - Test error handling for invalid queries

3. **Response Formatter Tests**
   - Test result formatting for different query types
   - Test pagination and result limiting
   - Test error message formatting

### Components to Build:
- `chat_parser.py` - Parse natural language to structured queries
- `query_builder.py` - Convert parsed queries to ES function calls
- `response_formatter.py` - Format ES results into chat responses
- `test_chat_components.py` - Comprehensive test suite

## Phase 2: Streamlit Integration (1-2 days)
**Goal**: Build working chat interface

### Tests to Write:
1. **Session State Tests**
   - Test conversation history persistence
   - Test ES session management integration
   - Test state reset functionality

2. **UI Component Tests**
   - Test chat input/output display
   - Test search result rendering
   - Test error state handling

### Components to Build:
- `streamlit_app.py` - Main Streamlit application
- `chat_interface.py` - Chat UI components
- `test_streamlit_integration.py` - UI integration tests

## Phase 3: Enhancement & Polish (1-2 days)
**Goal**: Add advanced features and improve UX

### Tests to Write:
1. **Advanced Query Tests**
   - Test complex multi-filter queries
   - Test follow-up question handling
   - Test query suggestion system

2. **Performance Tests**
   - Test response time for large result sets
   - Test memory usage with long conversations
   - Test concurrent user simulation

### Components to Build:
- Advanced query patterns
- Query suggestions and autocomplete
- Performance optimizations
- Export functionality

## Development Approach:
1. **Red-Green-Refactor** cycle for each component
2. **Integration tests** after each phase
3. **Manual testing** with real queries after each feature
4. **Continuous deployment** to local Streamlit server

## Success Metrics:
- All target queries working correctly
- Sub-2 second response times
- Intuitive chat interface
- Comprehensive error handling
- 90%+ test coverage

## Target Query Examples:
- "How many articles has Christian Fager published?"
- "List all articles that Anna Dubois has published"
- "How many publications have been published in Nature in 2023?"
- "What are the top 10 keywords on publications per year from 2020 to 2024?"

## Technical Stack:
- **Backend**: Existing ES tools + agent_tools.py
- **Frontend**: Streamlit for chat interface
- **Testing**: pytest for TDD approach
- **LLM**: OpenAI API or local model for query parsing

Ready to start with Phase 1 when you return!