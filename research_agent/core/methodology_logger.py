"""
Smart Methodology Learning System - LLM-powered analysis
AI observes and intelligently analyzes patterns using LLM reasoning
SMART: Uses LLM calls for dynamic categorization and insights
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import uuid
from langchain_litellm import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate

class SmartMethodologyLogger:
    """LLM-powered methodology logger with intelligent analysis in human-readable JSON format."""
    
    def __init__(self, log_file: str = "methodology_observations.json"):
        self.log_file = log_file
        self.observations_data = self._load_existing_data()
        
        # Initialize LLM for analysis
        self.analysis_llm = ChatLiteLLM(
            model="anthropic/claude-haiku-3.5",  # Fast and cheap for analysis
            api_key=os.getenv("LITELLM_API_KEY"),
            api_base=os.getenv("LITELLM_BASE_URL"),
            temperature=0.1  # Low temperature for consistent analysis
        )
        
        print(f"📋 Smart Methodology Logger initialized with LLM analysis: {log_file}")
    
    def _load_existing_data(self) -> Dict[str, List]:
        """Load existing observations data or create new structure."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"📖 Loaded existing observations: {sum(len(v) for v in data.values())} total")
                    return data
            except Exception as e:
                print(f"⚠️ Error loading existing data: {e}")
        
        # Create new structure
        new_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "description": "Smart Methodology Learning Log - LLM-Powered Analysis",
                "version": "1.0",
                "total_observations": 0
            },
            "query_starts": [],
            "tool_usages": [],
            "replanning_events": [],
            "session_completions": [],
            "followup_analyses": []
        }
        print(f"📁 Created new methodology observations structure")
        return new_data
    
    def log_query_start(self, session_id: str, query: str, is_followup: bool = False, 
                       previous_context: str = ""):
        """Log query start with LLM-powered classification and analysis."""
        
        # Use LLM to classify and analyze the query
        analysis = self._analyze_query_with_llm(query, is_followup, previous_context)
        
        observation = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "query": query,
            "is_followup": is_followup,
            "llm_analysis": analysis,
            "has_previous_context": bool(previous_context),
            "context_preview": previous_context[:200] if previous_context else ""
        }
        
        self.observations_data["query_starts"].append(observation)
        self._save_data()
        print(f"📝 Logged query start with LLM analysis: {query[:50]}... (type: {analysis.get('query_type', 'unknown')})")
    
    def log_tool_usage(self, session_id: str, tool_name: str, tool_input: str, 
                      success: bool, result_content: str = "", notes: str = ""):
        """Log tool usage with LLM-powered effectiveness analysis."""
        
        # Use LLM to analyze tool effectiveness
        effectiveness_analysis = self._analyze_tool_effectiveness_with_llm(
            tool_name, tool_input, success, result_content, notes
        )
        
        observation = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "tool_name": tool_name,
            "tool_input": tool_input[:1000],  # Truncate long inputs
            "success": success,
            "result_preview": result_content[:800] if result_content else "",
            "notes": notes,
            "llm_effectiveness_analysis": effectiveness_analysis
        }
        
        self.observations_data["tool_usages"].append(observation)
        self._save_data()
        quality = effectiveness_analysis.get('quality_assessment', 'unknown')
        print(f"🔧 Logged tool usage with LLM analysis: {tool_name} -> {quality}")
    
    def log_replanning_event(self, session_id: str, query: str, step_number: int,
                           reason: str, previous_approach: str, new_approach: str,
                           research_context: str = ""):
        """Log replanning with LLM-powered failure analysis."""
        
        # Use LLM to analyze why replanning was needed and predict improvement
        replanning_analysis = self._analyze_replanning_with_llm(
            query, reason, previous_approach, new_approach, research_context
        )
        
        observation = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "query": query,
            "step_number": step_number,
            "reason": reason,
            "previous_approach": previous_approach,
            "new_approach": new_approach,
            "research_context_preview": research_context[:1000] if research_context else "",
            "llm_replanning_analysis": replanning_analysis
        }
        
        self.observations_data["replanning_events"].append(observation)
        self._save_data()
        failure_type = replanning_analysis.get('failure_category', 'unknown')
        print(f"🔄 Logged replanning with LLM analysis: {failure_type}")
    
    def log_session_complete(self, session_id: str, query: str, total_steps: int,
                           replanning_count: int, final_success: str, 
                           execution_time: float, full_research_results: str = ""):
        """Log session completion with comprehensive LLM analysis."""
        
        # Use LLM to analyze the complete session and extract insights
        session_analysis = self._analyze_complete_session_with_llm(
            query, total_steps, replanning_count, final_success, 
            execution_time, full_research_results
        )
        
        observation = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "query": query,
            "execution_summary": {
                "total_steps": total_steps,
                "replanning_count": replanning_count,
                "execution_time_seconds": execution_time,
                "final_success": final_success,
            },
            "results_preview": full_research_results[:1000] if full_research_results else "",
            "llm_session_analysis": session_analysis
        }
        
        self.observations_data["session_completions"].append(observation)
        self._save_data()
        complexity = session_analysis.get('complexity_assessment', 'unknown')
        print(f"🎯 Logged session complete with LLM analysis: {final_success} ({complexity} complexity)")
    
    def log_followup_analysis(self, session_id: str, original_query: str, 
                            followup_query: str, context_usage_notes: str = "",
                            efficiency_observations: str = ""):
        """Log follow-up analysis with LLM-powered context effectiveness assessment."""
        
        # Use LLM to analyze follow-up effectiveness
        followup_analysis = self._analyze_followup_with_llm(
            original_query, followup_query, context_usage_notes, efficiency_observations
        )
        
        observation = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "original_query": original_query,
            "followup_query": followup_query,
            "context_usage_notes": context_usage_notes,
            "efficiency_observations": efficiency_observations,
            "llm_followup_analysis": followup_analysis
        }
        
        self.observations_data["followup_analyses"].append(observation)
        self._save_data()
        effectiveness = followup_analysis.get('context_effectiveness', 'unknown')
        print(f"🔗 Logged followup with LLM analysis: {effectiveness} context usage")
    
    def _analyze_query_with_llm(self, query: str, is_followup: bool, previous_context: str) -> Dict[str, Any]:
        """Use LLM to classify and analyze the query with robust JSON extraction."""
        
        prompt = ChatPromptTemplate.from_template("""
Analyze this research query and provide structured insights.

Query: "{query}"
Is Follow-up: {is_followup}
Previous Context: {previous_context}

IMPORTANT: Respond with ONLY valid JSON, no additional text before or after.

{{
    "query_type": "one of: expert_recommendation, author_lookup, geographic_search, topic_search, collaboration_search, temporal_search, publication_search, comparative_analysis, interdisciplinary_research, or other",
    "complexity_level": "simple, moderate, complex, or very_complex",
    "expected_challenges": ["list", "of", "potential", "challenges"],
    "suggested_approach": "brief strategy suggestion",
    "followup_insights": "analysis of how this relates to previous query (if applicable)",
    "success_predictors": ["factors", "that", "indicate", "likely", "success"],
    "key_entities": ["important", "entities", "mentioned"]
}}
""")
        
        try:
            response = self.analysis_llm.invoke(prompt.format(
                query=query,
                is_followup=is_followup,
                previous_context=previous_context[:800] if previous_context else "None"
            ))
            
            # Robust JSON extraction
            analysis = self._extract_json_simple(response.content)
            return analysis
            
        except Exception as e:
            print(f"⚠️ LLM query analysis failed: {e}")
            return self._create_fallback_query_analysis(query, str(e))
    
    def _analyze_tool_effectiveness_with_llm(self, tool_name: str, tool_input: str, 
                                           success: bool, result_content: str, notes: str) -> Dict[str, Any]:
        """Use LLM to analyze tool effectiveness with robust JSON extraction."""
        
        prompt = ChatPromptTemplate.from_template("""
Analyze this tool usage for effectiveness and insights.

Tool: {tool_name}
Input: {tool_input}
Success: {success}
Result Preview: {result_content}
Notes: {notes}

IMPORTANT: Respond with ONLY valid JSON, no additional text before or after.

{{
    "quality_assessment": "excellent, good, adequate, poor, or failed",
    "effectiveness_score": "0.0 to 1.0",
    "strengths": ["what", "worked", "well"],
    "weaknesses": ["what", "didn't", "work"],
    "improvement_suggestions": ["specific", "suggestions"],
    "input_optimization": "how to optimize inputs for this tool",
    "alternative_tools": ["better", "tools", "for", "this", "task"],
    "reusability": "high, medium, or low - how reusable is this approach"
}}
""")
        
        try:
            response = self.analysis_llm.invoke(prompt.format(
                tool_name=tool_name,
                tool_input=tool_input[:200],
                success=success,
                result_content=result_content[:800] if result_content else "No result content",
                notes=notes or "No notes"
            ))
            
            # Robust JSON extraction
            analysis = self._extract_json_simple(response.content)
            return analysis
            
        except Exception as e:
            print(f"⚠️ LLM tool analysis failed: {e}")
            return self._create_fallback_tool_analysis(tool_name, success, str(e))
    
    def _analyze_replanning_with_llm(self, query: str, reason: str, previous_approach: str, 
                                   new_approach: str, research_context: str) -> Dict[str, Any]:
        """Use LLM to analyze replanning decisions with robust JSON extraction."""
        
        prompt = ChatPromptTemplate.from_template("""
Analyze this replanning event to understand methodology evolution.

Original Query: {query}
Replanning Reason: {reason}
Previous Approach: {previous_approach}
New Approach: {new_approach}
Research Context: {research_context}

IMPORTANT: Respond with ONLY valid JSON, no additional text before or after.

{{
    "failure_category": "no_results, too_broad, irrelevant_results, tool_error, incomplete_info, or other",
    "root_cause": "deeper analysis of why the previous approach failed",
    "approach_improvement": "how the new approach addresses the failure",
    "success_probability": "0.0 to 1.0 - likelihood new approach will succeed",
    "lessons_learned": ["key", "insights", "from", "this", "failure"],
    "prevention_strategies": ["how", "to", "avoid", "this", "failure", "next", "time"],
    "pattern_recognition": "is this a common failure pattern?",
    "methodology_insights": ["broader", "insights", "about", "research", "methodology"]
}}
""")
        
        try:
            response = self.analysis_llm.invoke(prompt.format(
                query=query,
                reason=reason,
                previous_approach=previous_approach[:200],
                new_approach=new_approach[:200],
                research_context=research_context[:400] if research_context else "No context available"
            ))
            
            # Robust JSON extraction
            analysis = self._extract_json_simple(response.content)
            return analysis
            
        except Exception as e:
            print(f"⚠️ LLM replanning analysis failed: {e}")
            return self._create_fallback_replanning_analysis(reason, str(e))
    
    def _analyze_complete_session_with_llm(self, query: str, total_steps: int, replanning_count: int,
                                         final_success: str, execution_time: float, 
                                         full_results: str) -> Dict[str, Any]:
        """Use LLM to analyze complete research session with robust JSON extraction."""
        
        prompt = ChatPromptTemplate.from_template("""
Analyze this complete research session for methodology insights.

Query: {query}
Total Steps: {total_steps}
Replanning Events: {replanning_count}
Final Success: {final_success}
Execution Time: {execution_time} seconds
Results Preview: {full_results}

IMPORTANT: Respond with ONLY valid JSON, no additional text before or after.

{{
    "complexity_assessment": "simple, moderate, complex, or very_complex",
    "efficiency_score": "0.0 to 1.0 - how efficiently was this query handled",
    "methodology_effectiveness": "excellent, good, adequate, poor, or failed",
    "key_success_factors": ["what", "made", "this", "work"],
    "improvement_opportunities": ["specific", "ways", "to", "improve"],
    "optimal_step_count": "estimated optimal number of steps for this query",
    "reusable_patterns": ["patterns", "that", "could", "apply", "to", "similar", "queries"],
    "query_archetype": "what type of query pattern this represents",
    "scaling_insights": ["how", "this", "approach", "would", "scale"],
    "human_feedback_needed": ["areas", "where", "human", "review", "would", "help"]
}}
""")
        
        try:
            response = self.analysis_llm.invoke(prompt.format(
                query=query,
                total_steps=total_steps,
                replanning_count=replanning_count,
                final_success=final_success,
                execution_time=execution_time,
                full_results=full_results[:1000] if full_results else "No results available"
            ))
            
            # Robust JSON extraction
            analysis = self._extract_json_simple(response.content)
            return analysis
            
        except Exception as e:
            print(f"⚠️ LLM session analysis failed: {e}")
            return self._create_fallback_session_analysis(query, final_success, str(e))
    
    def _analyze_followup_with_llm(self, original_query: str, followup_query: str,
                                 context_usage_notes: str, efficiency_observations: str) -> Dict[str, Any]:
        """Use LLM to analyze follow-up question effectiveness with robust JSON extraction."""
        
        prompt = ChatPromptTemplate.from_template("""
Analyze this follow-up question interaction for context effectiveness.

Original Query: {original_query}
Follow-up Query: {followup_query}
Context Usage Notes: {context_usage_notes}
Efficiency Observations: {efficiency_observations}

IMPORTANT: Respond with ONLY valid JSON, no additional text before or after.

{{
    "followup_type": "entity_continuation, expansion_request, temporal_filter, relationship_exploration, or other",
    "context_relevance": "high, medium, or low",
    "context_effectiveness": "excellent, good, adequate, poor, or unused",
    "efficiency_gain": "high, medium, low, or none",
    "missed_opportunities": ["ways", "context", "could", "have", "been", "used", "better"],
    "context_optimization": ["suggestions", "for", "better", "context", "usage"],
    "conversation_flow": "natural, somewhat_natural, or awkward",
    "user_intent_clarity": "clear, somewhat_clear, or unclear",
    "memory_system_insights": ["insights", "about", "conversation", "memory", "effectiveness"]
}}
""")
        
        try:
            response = self.analysis_llm.invoke(prompt.format(
                original_query=original_query,
                followup_query=followup_query,
                context_usage_notes=context_usage_notes or "No notes provided",
                efficiency_observations=efficiency_observations or "No observations provided"
            ))
            
            # Robust JSON extraction
            analysis = self._extract_json_simple(response.content)
            return analysis
            
        except Exception as e:
            print(f"⚠️ LLM followup analysis failed: {e}")
            return self._create_fallback_followup_analysis(followup_query, str(e))
    
    def _save_data(self):
        """Save observations data to JSON file with proper formatting."""
        try:
            # Update metadata
            self.observations_data["metadata"]["last_updated"] = datetime.now().isoformat()
            self.observations_data["metadata"]["total_observations"] = sum(
                len(v) for k, v in self.observations_data.items() if k != "metadata"
            )
            
            # Write with nice formatting
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.observations_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"⚠️ Error saving methodology observations: {e}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of logged observations."""
        return {
            "total_observations": self.observations_data["metadata"]["total_observations"],
            "query_starts": len(self.observations_data["query_starts"]),
            "tool_usages": len(self.observations_data["tool_usages"]),
            "replanning_events": len(self.observations_data["replanning_events"]),
            "session_completions": len(self.observations_data["session_completions"]),
            "followup_analyses": len(self.observations_data["followup_analyses"]),
            "created_at": self.observations_data["metadata"]["created_at"],
            "last_updated": self.observations_data["metadata"].get("last_updated", "Never")
        }
    
    def get_recent_observations(self, observation_type: str = "all", limit: int = 10) -> List[Dict]:
        """Get recent observations of a specific type."""
        if observation_type == "all":
            # Combine all observations and sort by timestamp
            all_obs = []
            for obs_type, observations in self.observations_data.items():
                if obs_type != "metadata":
                    for obs in observations:
                        obs_with_type = obs.copy()
                        obs_with_type["observation_type"] = obs_type
                        all_obs.append(obs_with_type)
            
            all_obs.sort(key=lambda x: x["timestamp"], reverse=True)
            return all_obs[:limit]
        else:
            # Get specific type
            observations = self.observations_data.get(observation_type, [])
            return sorted(observations, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def _extract_json_simple(self, content: str) -> Dict[str, Any]:
        """Simple robust JSON extraction with fallback methods."""
        
        # Handle completely empty responses
        if not content or content.strip() == "":
            raise json.JSONDecodeError("Empty response from LLM", "", 0)

        # Method 1: Try direct JSON parsing first (handles clean responses)
        content_clean = content.strip()
        try:
            return json.loads(content_clean)
        except json.JSONDecodeError:
            pass
        
        # Method 2: Extract JSON between first { and last } (handles extra text)
        try:
            first_brace = content.find('{')
            last_brace = content.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_content = content[first_brace:last_brace + 1]
                return json.loads(json_content)
        except json.JSONDecodeError:
            pass
        
        # Method 3: If all fails, raise error for fallback handling
        raise json.JSONDecodeError("Could not extract valid JSON from LLM response", content, 0)
    
    def _create_fallback_query_analysis(self, query: str, error_msg: str) -> Dict[str, Any]:
        """Create fallback query analysis using simple pattern matching."""
        query_lower = query.lower()
        
        # Pattern-based classification
        if any(word in query_lower for word in ['examiner', 'thesis', 'supervisor', 'advisor']):
            query_type = 'expert_recommendation'
            complexity = 'complex'
        elif any(word in query_lower for word in ['who is', 'author', 'researcher']):
            query_type = 'author_lookup'
            complexity = 'moderate'
        elif any(word in query_lower for word in ['swedish', 'norway', 'country', 'geographic']):
            query_type = 'geographic_search'
            complexity = 'moderate'
        elif any(word in query_lower for word in ['publications', 'papers', 'articles']):
            query_type = 'publication_search'
            complexity = 'moderate'
        elif any(word in query_lower for word in ['collaboration', 'collaborate', 'work with']):
            query_type = 'collaboration_search'
            complexity = 'complex'
        else:
            query_type = 'other'
            complexity = 'moderate'
        
        return {
            "query_type": query_type,
            "complexity_level": complexity,
            "expected_challenges": ["llm_analysis_unavailable"],
            "suggested_approach": "comprehensive search approach",
            "success_predictors": ["relevant tools available"],
            "key_entities": [],
            "analysis_method": "fallback_pattern_matching",
            "llm_error": error_msg[:100]  # Truncate error message
        }
    
    def _create_fallback_tool_analysis(self, tool_name: str, success: bool, error_msg: str) -> Dict[str, Any]:
        """Create fallback tool analysis."""
        return {
            "quality_assessment": "good" if success else "failed",
            "effectiveness_score": "0.7" if success else "0.0",
            "strengths": ["execution_completed"] if success else [],
            "weaknesses": ["llm_analysis_unavailable"],
            "improvement_suggestions": ["enable_detailed_analysis"],
            "input_optimization": "optimize based on success patterns",
            "alternative_tools": [],
            "reusability": "medium",
            "analysis_method": "fallback_basic",
            "llm_error": error_msg[:100]
        }
    
    def _create_fallback_replanning_analysis(self, reason: str, error_msg: str) -> Dict[str, Any]:
        """Create fallback replanning analysis."""
        return {
            "failure_category": "other",
            "root_cause": reason,
            "approach_improvement": "attempting different strategy",
            "success_probability": "0.6",
            "lessons_learned": ["replanning_triggered"],
            "prevention_strategies": ["improve_initial_planning"],
            "pattern_recognition": "unknown",
            "methodology_insights": ["adaptation_capability_demonstrated"],
            "analysis_method": "fallback_basic",
            "llm_error": error_msg[:100]
        }
    
    def _create_fallback_session_analysis(self, query: str, final_success: str, error_msg: str) -> Dict[str, Any]:
        """Create fallback session analysis."""
        return {
            "complexity_assessment": "moderate",
            "efficiency_score": "0.7" if final_success == "success" else "0.3",
            "methodology_effectiveness": "adequate" if final_success == "success" else "poor",
            "key_success_factors": ["task_completion"] if final_success == "success" else [],
            "improvement_opportunities": ["enable_detailed_llm_analysis"],
            "optimal_step_count": "unknown",
            "reusable_patterns": ["basic_execution_pattern"],
            "query_archetype": "general_research",
            "scaling_insights": ["pattern_analysis_needed"],
            "human_feedback_needed": ["llm_analysis_setup"],
            "analysis_method": "fallback_basic",
            "llm_error": error_msg[:100]
        }
    
    def _create_fallback_followup_analysis(self, followup_query: str, error_msg: str) -> Dict[str, Any]:
        """Create fallback followup analysis."""
        followup_lower = followup_query.lower()
        
        # Simple followup type detection
        if any(word in followup_lower for word in ['his', 'her', 'their', 'this']):
            followup_type = 'entity_continuation'
        elif any(word in followup_lower for word in ['more', 'additional', 'other']):
            followup_type = 'expansion_request'
        elif any(word in followup_lower for word in ['recent', 'latest', '2023', '2024']):
            followup_type = 'temporal_filter'
        else:
            followup_type = 'other'
        
        return {
            "followup_type": followup_type,
            "context_relevance": "medium",
            "context_effectiveness": "adequate",
            "efficiency_gain": "medium",
            "missed_opportunities": ["detailed_context_analysis_unavailable"],
            "context_optimization": ["enable_llm_analysis"],
            "conversation_flow": "adequate",
            "user_intent_clarity": "somewhat_clear",
            "memory_system_insights": ["basic_followup_detection_working"],
            "analysis_method": "fallback_basic",
            "llm_error": error_msg[:100]
        }
        """Use LLM to analyze patterns across multiple observations."""
        
        # Load recent observations
        recent_observations = self._load_recent_observations_from_data(days)
        
        if not recent_observations:
            return {"error": "No recent observations found"}
        
        # Prepare summary of observations for LLM analysis
        obs_summary = self._prepare_observations_summary_from_data(recent_observations)
        
        prompt = ChatPromptTemplate.from_template("""
Analyze these research methodology observations to extract high-level insights:

{observations_summary}

Provide comprehensive analysis in this JSON format:
{{
    "overall_patterns": ["major", "patterns", "observed"],
    "success_patterns": ["what", "consistently", "works", "well"],
    "failure_patterns": ["what", "consistently", "fails"],
    "tool_insights": ["insights", "about", "tool", "effectiveness"],
    "query_type_insights": ["insights", "about", "different", "query", "types"],
    "efficiency_insights": ["insights", "about", "system", "efficiency"],
    "followup_insights": ["insights", "about", "followup", "handling"],
    "improvement_priorities": ["top", "priorities", "for", "improvement"],
    "methodology_recommendations": ["specific", "methodology", "changes"],
    "system_evolution": "how the system performance is trending",
    "human_action_items": ["specific", "actions", "humans", "should", "take"]
}}

Provide actionable, specific insights that will help improve the research system.
""")
        
        try:
            response = self.analysis_llm.invoke(prompt.format(
                observations_summary=obs_summary
            ))
            
            analysis = json.loads(response.content)
            analysis["analysis_period_days"] = days
            analysis["total_observations_analyzed"] = len(recent_observations)
            analysis["generated_at"] = datetime.now().isoformat()
            
            return analysis
            
        except Exception as e:
            print(f"⚠️ LLM insights analysis failed: {e}")
            return {
                "error": f"LLM analysis failed: {str(e)}",
                "total_observations_analyzed": len(recent_observations)
            }
    
    def _load_recent_observations_from_data(self, days: int) -> List[Dict[str, Any]]:
        """Load observations from recent days from the data structure."""
        cutoff = datetime.now() - timedelta(days=days)
        recent_obs = []
        
        try:
            for obs_type, observations in self.observations_data.items():
                if obs_type != "metadata":
                    for obs in observations:
                        try:
                            obs_time = datetime.fromisoformat(obs['timestamp'])
                            if obs_time > cutoff:
                                obs_with_type = obs.copy()
                                obs_with_type["observation_type"] = obs_type
                                recent_obs.append(obs_with_type)
                        except (KeyError, ValueError):
                            continue
        except Exception as e:
            print(f"⚠️ Error loading recent observations: {e}")
        
        return recent_obs
    
    def _prepare_observations_summary_from_data(self, observations: List[Dict[str, Any]]) -> str:
        """Prepare a summary of observations for LLM analysis from structured data."""
        summary_parts = []
        
        # Group by observation type
        by_type = {}
        for obs in observations:
            obs_type = obs.get('observation_type', 'unknown')
            if obs_type not in by_type:
                by_type[obs_type] = []
            by_type[obs_type].append(obs)
        
        # Summarize each type
        for obs_type, obs_list in by_type.items():
            summary_parts.append(f"\n{obs_type.upper()} ({len(obs_list)} events):")
            
            for i, obs in enumerate(obs_list[:5]):  # Limit to 5 examples per type
                if obs_type == "query_starts":
                    llm_analysis = obs.get('llm_analysis', {})
                    summary_parts.append(f"  Query: {obs.get('query', 'unknown')}")
                    summary_parts.append(f"    Type: {llm_analysis.get('query_type', 'unknown')}")
                    summary_parts.append(f"    Complexity: {llm_analysis.get('complexity_level', 'unknown')}")
                
                elif obs_type == "tool_usages":
                    effectiveness = obs.get('llm_effectiveness_analysis', {})
                    summary_parts.append(f"  Tool: {obs.get('tool_name', 'unknown')}")
                    summary_parts.append(f"    Quality: {effectiveness.get('quality_assessment', 'unknown')}")
                    summary_parts.append(f"    Score: {effectiveness.get('effectiveness_score', 'unknown')}")
                
                elif obs_type == "session_completions":
                    session_analysis = obs.get('llm_session_analysis', {})
                    summary_parts.append(f"  Query: {obs.get('query', 'unknown')}")
                    summary_parts.append(f"    Efficiency: {session_analysis.get('efficiency_score', 'unknown')}")
                    summary_parts.append(f"    Complexity: {session_analysis.get('complexity_assessment', 'unknown')}")
                
                if i >= 4:  # Limit examples
                    break
        
        return '\n'.join(summary_parts[:2000])  # Limit total length

# Factory function for easy initialization
def create_smart_methodology_logger(log_file: str = "methodology_observations.json") -> SmartMethodologyLogger:
    """Create smart methodology logger instance with LLM analysis in JSON format."""
    return SmartMethodologyLogger(log_file)

if __name__ == "__main__":
    # Test the smart methodology logger with JSON format
    print("Testing Smart Methodology Logger with LLM Analysis (JSON Format)...")
    
    try:
        logger = create_smart_methodology_logger("test_smart_methodology.json")
        
        # Test different types of logging with LLM analysis
        session_id = "test_session_json_123"
        
        # Test query start with LLM analysis
        logger.log_query_start(session_id, "Find Swedish researchers in explainable AI", is_followup=False)
        
        # Test tool usage with LLM analysis
        logger.log_tool_usage(session_id, "search_authors_by_country", "country:Sweden topic:explainable AI", 
                             success=True, 
                             result_content="Found 23 Swedish researchers working on explainable AI, including researchers from KTH, Chalmers, and Linköping University. Key publications in ICML, NeurIPS, and ICLR conferences.",
                             notes="Geographic search worked well, found comprehensive results")
        
        # Test replanning with LLM analysis
        logger.log_replanning_event(session_id, "Find Swedish researchers in explainable AI", 2,
                                   "Initial search returned too many general AI researchers, need to focus specifically on explainable AI", 
                                   "Broad AI researcher search in Sweden",
                                   "Specific explainable AI search with Swedish institution filter",
                                   research_context="Previous search found 847 AI researchers in Sweden, but most work on general machine learning rather than explainability specifically.")
        
        # Test session completion with LLM analysis
        logger.log_session_complete(session_id, "Find Swedish researchers in explainable AI", 
                                   total_steps=3, replanning_count=1,
                                   final_success="success", execution_time=67.5,
                                   full_research_results="Comprehensive analysis of Swedish explainable AI research landscape. Found 23 active researchers across 8 institutions. Key areas include interpretable machine learning, model transparency, and ethical AI. Leading institutions are KTH Royal Institute of Technology and Chalmers University of Technology.")
        
        # Test follow-up analysis with LLM analysis
        logger.log_followup_analysis(session_id, "Find Swedish researchers in explainable AI",
                                   "What are the most recent publications from these researchers?",
                                   context_usage_notes="Successfully used previous researcher list to search for recent publications",
                                   efficiency_observations="Follow-up query completed 70% faster due to existing researcher context")
        
        # Show summary stats
        stats = logger.get_summary_stats()
        print(f"\n📊 Summary Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Show recent observations
        recent = logger.get_recent_observations(limit=3)
        print(f"\n🔍 Recent Observations ({len(recent)}):")
        for obs in recent:
            print(f"  {obs['timestamp']}: {obs['observation_type']}")
        
        print("✅ Smart Methodology Logger with JSON format test completed!")
        print("📊 Check test_smart_methodology.json for human-readable structured observations")
        print("🧠 LLM analysis provides dynamic categorization and insights")
        print("📖 JSON format is much more readable and easier to browse")
        print("🎯 Ready for integration with your existing workflow!")
        
        # Test LLM insights summary
        print("\n🔍 Testing LLM insights generation...")
        insights = logger.generate_llm_insights_summary(days=1)
        print(f"📈 Generated insights: {len(insights)} categories analyzed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure LITELLM_API_KEY and LITELLM_BASE_URL are set in your environment")