"""
Query classifier to distinguish conversational vs research queries.

This module provides fast classification of user queries to determine
if they need database tools or can be answered conversationally.
"""

import re
from typing import Dict, List, Literal, Optional
from langchain_litellm import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

QueryType = Literal["conversational", "research", "mixed"]


class QueryClassification(BaseModel):
    """Result of query classification."""
    query_type: QueryType = Field(description="Type of query: conversational, research, or mixed")
    confidence: float = Field(description="Confidence score (0.0-1.0)")
    reasoning: str = Field(description="Brief explanation of classification")
    needs_tools: bool = Field(description="Whether the query needs database tools")
    escalate_if_needed: bool = Field(description="Whether to escalate to full workflow if needed")


class QueryClassifier:
    """
    Classifies queries to optimize response speed and tool usage using LLM.
    """
    
    def __init__(self):
        """Initialize the query classifier."""
        try:
            self.llm = ChatLiteLLM(
                model="anthropic/claude-sonet-3.7",  # Fast, cheap model for classification
                api_key=os.getenv("LITELLM_API_KEY"),
                api_base=os.getenv("LITELLM_BASE_URL"),
                temperature=0
            )
            
            self.classifier_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a query classifier for a research publications database system.

Classify user queries into exactly one of these categories:

**conversational**: Pure conversational queries that don't need database access
- Simple greetings: "Hello", "Hi", "Good morning"
- Thanks: "Thank you", "Thanks"
- Goodbyes: "Bye", "Goodbye", "See you"
- Simple questions: "How are you?", "What's up?"
- Acknowledgments: "OK", "Yes", "No", "Sure"
- Apologies: "Sorry", "My mistake"

**research**: Queries that require searching the publications database
- Author queries: "How many papers has John Smith published?"
- Publication searches: "Find articles about machine learning"
- Data requests: "List publications from 2023"
- Research questions: "Who published about AI?"
- Comparisons: "Compare authors Smith and Jones"
- Statistics: "How many papers in Nature?"

**mixed**: Queries combining conversational + research elements
- "Thanks! Now find papers by Anna Dubois"
- "Hello, can you search for machine learning papers?"
- "I don't understand, show me more publications"

CRITICAL RULES:
1. If query mentions authors, papers, publications, research → research
2. If query has conversational parts BUT also research content → research (for safety)
3. If uncertain or ambiguous → research (for safety)
4. Only classify as conversational if query is PURELY conversational
5. Consider the full query, not just the beginning
6. IMPORTANT: Mixed queries should be classified as "research" for safety, not "mixed"

Respond with structured output including query_type, confidence (0.0-1.0), reasoning, needs_tools (boolean), and escalate_if_needed (boolean)."""),
                ("human", "Query: {query}\n\nConversation context: {context}\n\nClassify this query:")
            ])
            
            self.classifier = self.classifier_prompt | self.llm.with_structured_output(QueryClassification)
            
        except Exception as e:
            print(f"Warning: Failed to initialize LLM classifier: {e}")
            self.llm = None
            self.classifier = None
    
    def classify_query(self, query: str, conversation_context: Optional[List[Dict]] = None) -> QueryClassification:
        """
        Classify a query as conversational, research, or mixed.
        
        Args:
            query: The user query to classify
            conversation_context: Recent conversation history for context
            
        Returns:
            QueryClassification with type, confidence, and reasoning
        """
        # Use LLM for classification if available
        if self.classifier:
            context_str = self._format_context(conversation_context) if conversation_context else "No previous context"
            
            try:
                classification = self.classifier.invoke({
                    "query": query,
                    "context": context_str
                })
                
                # Ensure confidence is reasonable
                if classification.confidence < 0.1:
                    classification.confidence = 0.5
                
                return classification
                
            except Exception as e:
                print(f"Warning: LLM classification failed: {e}")
                # Fallback to simple rule-based classification
                return self._simple_fallback_classify(query)
        
        # Fallback to simple rule-based classification if no LLM
        return self._simple_fallback_classify(query)
    
    def _simple_fallback_classify(self, query: str) -> QueryClassification:
        """
        Simple fallback classification when LLM is unavailable.
        
        Args:
            query: The user query
            
        Returns:
            QueryClassification with conservative defaults
        """
        if not query:
            return QueryClassification(
                query_type="research",
                confidence=0.5,
                reasoning="Empty query defaults to research",
                needs_tools=True,
                escalate_if_needed=True
            )
        
        query_lower = query.lower().strip()
        
        # Very simple heuristics - conservative approach
        research_keywords = [
            "paper", "publication", "article", "author", "research", "study",
            "published", "journal", "conference", "how many", "find", "search",
            "list", "show", "who", "what", "when", "where", "compare", "statistics"
        ]
        
        conversational_only = [
            "hello", "hi", "thank", "thanks", "bye", "goodbye", "how are you", 
            "what's up", "yes", "no", "okay", "ok", "sorry"
        ]
        
        # Check for research keywords
        if any(keyword in query_lower for keyword in research_keywords):
            return QueryClassification(
                query_type="research",
                confidence=0.7,
                reasoning="Contains research keywords",
                needs_tools=True,
                escalate_if_needed=True
            )
        
        # Check for purely conversational patterns
        if any(conv in query_lower for conv in conversational_only) and len(query_lower.split()) <= 3:
            return QueryClassification(
                query_type="conversational",
                confidence=0.6,
                reasoning="Simple conversational query",
                needs_tools=False,
                escalate_if_needed=False
            )
        
        # Default to research for safety
        return QueryClassification(
            query_type="research",
            confidence=0.5,
            reasoning="Fallback classification - defaulting to research for safety",
            needs_tools=True,
            escalate_if_needed=True
        )
    
    def _format_context(self, context: List[Dict]) -> str:
        """Format conversation context for LLM."""
        if not context:
            return "No previous context"
        
        # Take last 3 messages for context
        recent_context = context[-3:]
        formatted = []
        
        for msg in recent_context:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:100]  # Truncate for brevity
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    def should_use_fast_path(self, query: str, conversation_context: Optional[List[Dict]] = None) -> bool:
        """
        Determine if a query should use the fast conversational path.
        
        Args:
            query: The user query
            conversation_context: Recent conversation history
            
        Returns:
            True if query should use fast path, False for full workflow
        """
        classification = self.classify_query(query, conversation_context)
        
        # Use fast path for high-confidence conversational queries
        return (
            classification.query_type == "conversational" and 
            classification.confidence > 0.7 and 
            not classification.needs_tools
        )
    
    def get_processing_message(self, query: str, conversation_context: Optional[List[Dict]] = None) -> str:
        """
        Get appropriate processing message for user feedback.
        
        Args:
            query: The user query
            conversation_context: Recent conversation history
            
        Returns:
            Processing message string
        """
        classification = self.classify_query(query, conversation_context)
        
        if classification.query_type == "conversational":
            return "💬 Responding..."
        elif classification.query_type == "research":
            return "🔍 Researching publications..."
        else:  # mixed
            return "🔍 Processing and researching..."


# Convenience functions for easy usage
def classify_query(query: str, conversation_context: Optional[List[Dict]] = None) -> QueryClassification:
    """Classify a query using the global classifier."""
    classifier = QueryClassifier()
    return classifier.classify_query(query, conversation_context)


def should_use_fast_path(query: str, conversation_context: Optional[List[Dict]] = None) -> bool:
    """Determine if query should use fast path."""
    classifier = QueryClassifier()
    return classifier.should_use_fast_path(query, conversation_context)


def get_processing_message(query: str, conversation_context: Optional[List[Dict]] = None) -> str:
    """Get processing message for user feedback."""
    classifier = QueryClassifier()
    return classifier.get_processing_message(query, conversation_context)