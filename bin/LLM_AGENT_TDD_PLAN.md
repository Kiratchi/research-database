# LLM Agent TDD Plan: Plan-and-Execute Research Publications Agent

## Overview

Create a plan-and-execute LLM agent using LangChain that can handle natural language queries about research publications. The agent will generate execution plans and use our existing Elasticsearch tools to fulfill complex queries.

## Success Criteria

### Primary Goals
1. **Natural Language Flexibility**: Handle any reasonable query format without regex pattern limitations
2. **Plan Generation**: Create step-by-step execution plans for complex queries
3. **Tool Composition**: Chain multiple tools together to answer complex questions
4. **Error Recovery**: Retry and adapt when tools fail or return unexpected results
5. **Parallel Capability**: Run alongside existing regex-based system for comparison

### Success Metrics
- Handle 95% of natural language variations (uppercase, lowercase, different phrasings)
- Successfully plan and execute multi-step queries (comparisons, trends, complex filters)
- Provide clear execution traces for debugging
- Response time < 10 seconds for typical queries
- Graceful error handling with meaningful messages

## Phase 1: Foundation and Simple Planning

### 1.1 Setup LangChain Infrastructure
**Test**: `test_langchain_setup.py`
```python
def test_openai_connection():
    """Test OpenAI API connection works"""
    
def test_langchain_imports():
    """Test all required LangChain components import correctly"""
```

**Implementation**:
- Add LangChain and OpenAI to requirements.txt
- Create basic LangChain configuration
- Set up OpenAI API key handling

**Success Criteria**: 
- LangChain imports without errors
- OpenAI API connection established
- Basic LLM calls work

### 1.2 Convert Existing Tools to LangChain Format
**Test**: `test_langchain_tools.py`
```python
def test_search_by_author_tool():
    """Test search_by_author works as LangChain tool"""
    
def test_search_publications_tool():
    """Test search_publications works as LangChain tool"""
    
def test_get_statistics_tool():
    """Test statistics tool works as LangChain tool"""
```

**Implementation**:
- Create `langchain_tools.py` with tool wrappers
- Add proper schemas using Pydantic
- Include examples and descriptions for each tool

**Success Criteria**:
- All existing tools work as LangChain tools
- Tools have proper schemas and descriptions
- Tools return structured data with metadata

### 1.3 Simple Query Planning
**Test**: `test_simple_planner.py`
```python
def test_single_step_author_query():
    """Test: 'How many papers has Christian Fager published?' -> single search_by_author call"""
    
def test_single_step_topic_query():
    """Test: 'Find papers about machine learning' -> single search_publications call"""
    
def test_plan_structure():
    """Test generated plans have correct structure (steps, tools, params)"""
```

**Implementation**:
- Create `llm_planner.py` with basic planning logic
- Use OpenAI function calling for tool selection
- Generate simple 1-2 step plans

**Success Criteria**:
- Plans generated for simple queries
- Plans include correct tool selection
- Plans have valid parameters

## Phase 2: Plan Execution and Multi-Step Queries

### 2.1 Plan Execution Engine
**Test**: `test_plan_executor.py`
```python
def test_execute_single_step_plan():
    """Test executing a single-step plan returns correct results"""
    
def test_execute_multi_step_plan():
    """Test executing multi-step plan with parameter passing"""
    
def test_execution_error_handling():
    """Test execution handles tool failures gracefully"""
```

**Implementation**:
- Create `plan_executor.py` with step-by-step execution
- Handle parameter passing between steps
- Add error handling and retry logic

**Success Criteria**:
- Single-step plans execute correctly
- Multi-step plans chain results properly
- Errors are caught and handled gracefully

### 2.2 Multi-Step Query Planning
**Test**: `test_multi_step_planning.py`
```python
def test_author_comparison_query():
    """Test: 'Compare Christian Fager and Anna Dubois publication counts' -> multi-step plan"""
    
def test_temporal_analysis_query():
    """Test: 'Show publication trends for machine learning from 2020-2024' -> multi-step plan"""
    
def test_complex_filter_query():
    """Test: 'Find recent papers by Swedish authors in Nature' -> multi-step plan"""
```

**Implementation**:
- Enhance planner for multi-step queries
- Add parameter passing logic ($step1.result format)
- Create query analysis and decomposition

**Success Criteria**:
- Multi-step plans generated correctly
- Parameter references work between steps
- Complex queries decomposed logically

### 2.3 Response Formatting
**Test**: `test_llm_response_formatting.py`
```python
def test_format_count_response():
    """Test count responses are formatted naturally"""
    
def test_format_comparison_response():
    """Test comparison responses are clear and informative"""
    
def test_format_list_response():
    """Test publication lists are well-formatted"""
```

**Implementation**:
- Create `llm_response_formatter.py` for natural language responses
- Use LLM to generate human-readable summaries
- Handle different response types (counts, lists, comparisons)

**Success Criteria**:
- Responses are natural and informative
- Different query types get appropriate formatting
- Responses include relevant details and context

## Phase 3: Advanced Features and Integration

### 3.1 Enhanced Tool Set
**Test**: `test_enhanced_tools.py`
```python
def test_author_comparison_tool():
    """Test direct author comparison tool"""
    
def test_trend_analysis_tool():
    """Test publication trend analysis tool"""
    
def test_keyword_extraction_tool():
    """Test keyword extraction from query"""
```

**Implementation**:
- Add `compare_authors()` tool
- Add `analyze_trends()` tool
- Add `extract_keywords()` tool for better query understanding

**Success Criteria**:
- New tools integrate seamlessly
- Tools provide rich, structured outputs
- Tools have clear documentation and examples

### 3.2 Error Recovery and Adaptation
**Test**: `test_error_recovery.py`
```python
def test_tool_failure_recovery():
    """Test system recovers when tools fail"""
    
def test_parameter_correction():
    """Test system corrects invalid parameters"""
    
def test_alternative_strategy():
    """Test system tries alternative approaches when first plan fails"""
```

**Implementation**:
- Add retry logic with exponential backoff
- Implement plan adaptation when tools fail
- Add parameter validation and correction

**Success Criteria**:
- System recovers from transient failures
- Invalid parameters are corrected automatically
- Alternative strategies are tried when needed

### 3.3 Streamlit Integration
**Test**: `test_llm_streamlit_integration.py`
```python
def test_llm_agent_in_streamlit():
    """Test LLM agent works in Streamlit app"""
    
def test_plan_visualization():
    """Test execution plans are visualized in UI"""
    
def test_parallel_comparison():
    """Test both regex and LLM agents can run side-by-side"""
```

**Implementation**:
- Create `llm_streamlit_app.py` or enhance existing app
- Add plan visualization and debugging
- Allow switching between regex and LLM agents

**Success Criteria**:
- LLM agent works in Streamlit
- Users can see plan generation and execution
- Both systems can be compared side-by-side

## Phase 4: Testing and Optimization

### 4.1 Comprehensive Testing
**Test**: `test_comprehensive_scenarios.py`
```python
def test_case_insensitive_queries():
    """Test all case variations work correctly"""
    
def test_complex_query_scenarios():
    """Test complex real-world query scenarios"""
    
def test_performance_benchmarks():
    """Test response times meet requirements"""
```

**Implementation**:
- Create comprehensive test suite
- Add performance benchmarks
- Test edge cases and error conditions

**Success Criteria**:
- All test cases pass
- Performance requirements met
- Edge cases handled gracefully

### 4.2 Documentation and Examples
**Test**: `test_documentation.py`
```python
def test_example_queries():
    """Test all documented example queries work"""
    
def test_tool_documentation():
    """Test tool documentation is accurate and complete"""
```

**Implementation**:
- Create comprehensive documentation
- Add example queries and expected outputs
- Document troubleshooting guide

**Success Criteria**:
- Documentation is complete and accurate
- Examples work as shown
- Troubleshooting guide is helpful

## Implementation Timeline

### Week 1: Foundation (Phase 1)
- Day 1-2: Setup LangChain infrastructure
- Day 3-4: Convert existing tools to LangChain format
- Day 5-7: Implement simple query planning

### Week 2: Execution (Phase 2)
- Day 1-3: Build plan execution engine
- Day 4-5: Implement multi-step planning
- Day 6-7: Add response formatting

### Week 3: Advanced Features (Phase 3)
- Day 1-3: Enhanced tool set
- Day 4-5: Error recovery and adaptation
- Day 6-7: Streamlit integration

### Week 4: Polish (Phase 4)
- Day 1-3: Comprehensive testing
- Day 4-5: Documentation and examples
- Day 6-7: Performance optimization

## Technology Stack

### Core Components
- **LangChain**: Agent framework and tool orchestration
- **OpenAI GPT-4**: Primary LLM for planning and formatting
- **Pydantic**: Tool schemas and data validation
- **Elasticsearch**: Existing search infrastructure (unchanged)

### Development Tools
- **pytest**: Testing framework
- **Streamlit**: UI framework
- **python-dotenv**: Environment management
- **langchain-community**: Additional LangChain tools

## Risk Mitigation

### Technical Risks
- **LLM API costs**: Start with GPT-3.5, upgrade to GPT-4 only if needed
- **Response latency**: Implement caching and optimize tool calls
- **Error handling**: Comprehensive testing of failure scenarios

### Implementation Risks
- **Scope creep**: Stick to TDD plan, avoid feature additions
- **Tool complexity**: Keep tools simple and atomic
- **Integration issues**: Test integration early and often

## Success Validation

### Acceptance Tests
1. **Natural Language Flexibility**: 20 query variations for each common pattern
2. **Complex Query Handling**: Multi-step scenarios like comparisons and trends
3. **Error Recovery**: Deliberate failures and malformed inputs
4. **Performance**: Response time benchmarks under load
5. **Parallel Operation**: Both systems working simultaneously

### Demo Scenarios
1. "How many papers has christian fager published?" (case insensitive)
2. "Compare publication counts between Christian Fager and Anna Dubois"
3. "Show me machine learning publication trends from 2020 to 2024"
4. "Find recent papers by Swedish authors in high-impact journals"
5. "What are the top research keywords in quantum computing papers?"

This plan provides a clear path from basic LangChain setup to a fully functional plan-and-execute agent, with testable milestones and clear success criteria at each phase.