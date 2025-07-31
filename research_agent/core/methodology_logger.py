"""
Standard Methodology Logger - Fast logging without LLM calls
Captures structured metrics and patterns for analysis without LLM overhead
FAST: No LLM calls during execution - only structured data logging
"""
import json
import os
import time
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import uuid
from collections import defaultdict

class StandardMethodologyLogger:
    """Fast methodology logger with structured metrics - no LLM calls during execution."""
    
    def __init__(self, log_file: str = "methodology_observations.json"):
        self.log_file = log_file
        self.observations_data = self._load_existing_data()
        
        print(f"⚡ Standard Methodology Logger initialized (no LLM overhead): {log_file}")
    
    def _load_existing_data(self) -> Dict[str, List]:
        """Load existing observations data or create new structure."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"📖 Loaded existing observations: {sum(len(v) for v in data.values() if isinstance(v, list))} total")
                    return data
            except Exception as e:
                print(f"⚠️ Error loading existing data: {e}")
        
        # Create new structure
        new_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "description": "Standard Methodology Log - Fast Structured Metrics",
                "version": "2.0",
                "total_observations": 0,
                "logger_type": "standard_fast"
            },
            "query_starts": [],
            "tool_usages": [],
            "replanning_events": [],
            "session_completions": [],
            "followup_analyses": []
        }
        print(f"📁 Created new standard methodology structure")
        return new_data
    
    def log_query_start(self, session_id: str, query: str, is_followup: bool = False, 
                       previous_context: str = ""):
        """Log query start with fast heuristic analysis."""
        
        # Fast heuristic analysis (no LLM calls)
        query_analysis = self._analyze_query_fast(query, is_followup, previous_context)
        
        observation = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "query": query,
            "query_length": len(query),
            "is_followup": is_followup,
            "has_previous_context": bool(previous_context),
            "context_length": len(previous_context),
            
            # Fast heuristic analysis
            "heuristic_analysis": query_analysis,
            
            # Raw metrics for later analysis
            "word_count": len(query.split()),
            "question_marks": query.count('?'),
            "contains_numbers": bool(re.search(r'\d', query)),
            "contains_quotes": '"' in query or "'" in query
        }
        
        self.observations_data["query_starts"].append(observation)
        self._save_data()
        print(f"📝 Logged query start: {query[:50]}... (type: {query_analysis['query_type']})")
    
    def log_tool_usage(self, session_id: str, tool_name: str, tool_input: str, 
                      success: bool, result_content: str = "", notes: str = ""):
        """Log tool usage with fast effectiveness metrics."""
        
        # Fast effectiveness analysis (no LLM calls)
        effectiveness_metrics = self._analyze_tool_effectiveness_fast(
            tool_name, tool_input, success, result_content, notes
        )
        
        observation = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "tool_name": tool_name,
            "tool_input_length": len(tool_input),
            "success": success,
            "result_length": len(result_content),
            "has_notes": bool(notes),
            
            # Fast effectiveness metrics
            "effectiveness_metrics": effectiveness_metrics,
            
            # Raw data for analysis
            "tool_input_preview": tool_input[:200],
            "result_preview": result_content[:300] if result_content else "",
            "notes": notes[:200] if notes else ""
        }
        
        self.observations_data["tool_usages"].append(observation)
        self._save_data()
        quality = effectiveness_metrics['quality_score']
        print(f"🔧 Logged tool usage: {tool_name} -> {quality}")
    
    def log_replanning_event(self, session_id: str, query: str, step_number: int,
                           reason: str, previous_approach: str, new_approach: str,
                           research_context: str = ""):
        """Log replanning with fast failure pattern analysis."""
        
        # Fast replanning analysis (no LLM calls)
        replanning_metrics = self._analyze_replanning_fast(
            query, reason, previous_approach, new_approach, research_context
        )
        
        observation = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "query_length": len(query),
            "step_number": step_number,
            "reason_length": len(reason),
            "context_available": bool(research_context),
            "context_length": len(research_context),
            
            # Fast replanning metrics
            "replanning_metrics": replanning_metrics,
            
            # Raw data for analysis
            "query": query,
            "reason": reason[:300],
            "previous_approach": previous_approach[:200],
            "new_approach": new_approach[:200],
            "research_context_preview": research_context[:300] if research_context else ""
        }
        
        self.observations_data["replanning_events"].append(observation)
        self._save_data()
        failure_type = replanning_metrics['failure_category']
        print(f"🔄 Logged replanning: {failure_type}")
    
    def log_session_complete(self, session_id: str, query: str, total_steps: int,
                           replanning_count: int, final_success: str, 
                           execution_time: float, full_research_results: str = ""):
        """Log session completion with fast performance metrics."""
        
        # Fast session analysis (no LLM calls)
        session_metrics = self._analyze_session_fast(
            query, total_steps, replanning_count, final_success, 
            execution_time, full_research_results
        )
        
        observation = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "query_length": len(query),
            "execution_summary": {
                "total_steps": total_steps,
                "replanning_count": replanning_count,
                "execution_time_seconds": execution_time,
                "final_success": final_success,
                "results_length": len(full_research_results),
                "steps_per_minute": (total_steps / (execution_time / 60)) if execution_time > 0 else 0
            },
            
            # Fast session metrics
            "session_metrics": session_metrics,
            
            # Raw data for analysis
            "query": query,
            "results_preview": full_research_results[:500] if full_research_results else ""
        }
        
        self.observations_data["session_completions"].append(observation)
        self._save_data()
        efficiency = session_metrics['efficiency_category']
        print(f"🎯 Logged session complete: {final_success} ({efficiency} efficiency)")
    
    def log_followup_analysis(self, session_id: str, original_query: str, 
                            followup_query: str, context_usage_notes: str = "",
                            efficiency_observations: str = ""):
        """Log follow-up analysis with fast context effectiveness metrics."""
        
        # Fast followup analysis (no LLM calls)
        followup_metrics = self._analyze_followup_fast(
            original_query, followup_query, context_usage_notes, efficiency_observations
        )
        
        observation = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "original_query_length": len(original_query),
            "followup_query_length": len(followup_query),
            "has_context_notes": bool(context_usage_notes),
            "has_efficiency_notes": bool(efficiency_observations),
            
            # Fast followup metrics
            "followup_metrics": followup_metrics,
            
            # Raw data for analysis
            "original_query": original_query,
            "followup_query": followup_query,
            "context_usage_notes": context_usage_notes[:300],
            "efficiency_observations": efficiency_observations[:300]
        }
        
        self.observations_data["followup_analyses"].append(observation)
        self._save_data()
        effectiveness = followup_metrics['context_effectiveness']
        print(f"🔗 Logged followup: {effectiveness} context effectiveness")
    
    # =============================================================================
    # FAST HEURISTIC ANALYSIS METHODS (NO LLM CALLS)
    # =============================================================================
    
    def _analyze_query_fast(self, query: str, is_followup: bool, previous_context: str) -> Dict[str, Any]:
        """Fast query analysis using heuristics instead of LLM."""
        
        query_lower = query.lower()
        
        # Determine query type based on keywords
        query_type = "other"
        if any(word in query_lower for word in ["who is", "author", "researcher", "professor"]):
            query_type = "author_lookup"
        elif any(word in query_lower for word in ["country", "sweden", "german", "university"]):
            query_type = "geographic_search"
        elif any(word in query_lower for word in ["topic", "field", "research area", "about"]):
            query_type = "topic_search"
        elif any(word in query_lower for word in ["collaborate", "work with", "team"]):
            query_type = "collaboration_search"
        elif any(word in query_lower for word in ["recent", "latest", "2023", "2024", "current"]):
            query_type = "temporal_search"
        elif any(word in query_lower for word in ["publication", "paper", "article", "journal"]):
            query_type = "publication_search"
        elif any(word in query_lower for word in ["compare", "versus", "difference", "between"]):
            query_type = "comparative_analysis"
        
        # Determine complexity based on length and structure
        complexity_level = "simple"
        if len(query.split()) > 15:
            complexity_level = "complex"
        elif len(query.split()) > 8:
            complexity_level = "moderate"
        
        # Predict challenges based on patterns
        challenges = []
        if len(query.split()) > 20:
            challenges.append("very_long_query")
        if query.count('?') > 1:
            challenges.append("multiple_questions")
        if '"' in query:
            challenges.append("specific_phrases")
        if not is_followup and len(query.split()) < 3:
            challenges.append("too_vague")
        
        return {
            "query_type": query_type,
            "complexity_level": complexity_level,
            "expected_challenges": challenges,
            "is_specific": len([w for w in query.split() if w.isupper()]) > 0,  # Has proper nouns
            "has_temporal_elements": any(word in query_lower for word in ["when", "recent", "2023", "2024", "latest"]),
            "followup_context_available": is_followup and bool(previous_context)
        }
    
    def _analyze_tool_effectiveness_fast(self, tool_name: str, tool_input: str, 
                                       success: bool, result_content: str, notes: str) -> Dict[str, Any]:
        """Fast tool effectiveness analysis using heuristics."""
        
        # Quality score based on success and result length
        if not success:
            quality_score = 0.0
        elif len(result_content) < 100:
            quality_score = 0.3
        elif len(result_content) < 500:
            quality_score = 0.6
        elif len(result_content) < 2000:
            quality_score = 0.8
        else:
            quality_score = 1.0
        
        # Quality category
        if quality_score >= 0.8:
            quality_category = "excellent"
        elif quality_score >= 0.6:
            quality_category = "good"
        elif quality_score >= 0.3:
            quality_category = "adequate"
        elif quality_score > 0:
            quality_category = "poor"
        else:
            quality_category = "failed"
        
        # Simple effectiveness indicators
        has_error_indicators = any(word in result_content.lower() for word in ["error", "failed", "not found", "no results"])
        has_success_indicators = any(word in result_content.lower() for word in ["found", "located", "identified", "discovered"])
        
        return {
            "quality_score": quality_score,
            "quality_category": quality_category,
            "input_length": len(tool_input),
            "output_length": len(result_content),
            "success": success,
            "has_error_indicators": has_error_indicators,
            "has_success_indicators": has_success_indicators,
            "efficiency_ratio": len(result_content) / len(tool_input) if len(tool_input) > 0 else 0,
            "tool_type": tool_name.split('_')[0] if '_' in tool_name else tool_name
        }
    
    def _analyze_replanning_fast(self, query: str, reason: str, previous_approach: str, 
                               new_approach: str, research_context: str) -> Dict[str, Any]:
        """Fast replanning analysis using heuristics."""
        
        reason_lower = reason.lower()
        
        # Categorize failure reason
        failure_category = "other"
        if any(word in reason_lower for word in ["no results", "not found", "empty"]):
            failure_category = "no_results"
        elif any(word in reason_lower for word in ["too broad", "too many", "generic"]):
            failure_category = "too_broad"
        elif any(word in reason_lower for word in ["irrelevant", "wrong", "incorrect"]):
            failure_category = "irrelevant_results"
        elif any(word in reason_lower for word in ["error", "failed", "timeout"]):
            failure_category = "tool_error"
        elif any(word in reason_lower for word in ["incomplete", "partial", "missing"]):
            failure_category = "incomplete_info"
        
        # Assess improvement potential
        approach_change_significant = len(set(previous_approach.split()) & set(new_approach.split())) < 0.5 * len(previous_approach.split())
        
        return {
            "failure_category": failure_category,
            "reason_length": len(reason),
            "approach_change_significant": approach_change_significant,
            "context_available": bool(research_context),
            "context_length": len(research_context),
            "step_evolution": len(new_approach.split()) - len(previous_approach.split()),
            "specificity_increase": "specific" in new_approach.lower() and "specific" not in previous_approach.lower()
        }
    
    def _analyze_session_fast(self, query: str, total_steps: int, replanning_count: int,
                            final_success: str, execution_time: float, full_results: str) -> Dict[str, Any]:
        """Fast session analysis using heuristics."""
        
        # Efficiency assessment
        steps_per_minute = (total_steps / (execution_time / 60)) if execution_time > 0 else 0
        
        if steps_per_minute > 2:
            efficiency_category = "high"
        elif steps_per_minute > 1:
            efficiency_category = "medium"
        else:
            efficiency_category = "low"
        
        # Success assessment
        success_score = 0.0
        if final_success == "success":
            success_score = 1.0
        elif final_success == "partial":
            success_score = 0.5
        
        # Complexity assessment based on steps and replanning
        if total_steps <= 2 and replanning_count == 0:
            complexity_category = "simple"
        elif total_steps <= 4 and replanning_count <= 1:
            complexity_category = "moderate"
        elif total_steps <= 6 and replanning_count <= 2:
            complexity_category = "complex"
        else:
            complexity_category = "very_complex"
        
        return {
            "efficiency_category": efficiency_category,
            "steps_per_minute": steps_per_minute,
            "success_score": success_score,
            "complexity_category": complexity_category,
            "replanning_ratio": replanning_count / total_steps if total_steps > 0 else 0,
            "results_per_step": len(full_results) / total_steps if total_steps > 0 else 0,
            "execution_efficiency": len(full_results) / execution_time if execution_time > 0 else 0
        }
    
    def _analyze_followup_fast(self, original_query: str, followup_query: str,
                             context_usage_notes: str, efficiency_observations: str) -> Dict[str, Any]:
        """Fast follow-up analysis using heuristics."""
        
        # Determine followup type based on patterns
        followup_lower = followup_query.lower()
        followup_type = "other"
        
        if any(word in followup_lower for word in ["more about", "tell me more", "details"]):
            followup_type = "expansion_request"
        elif any(word in followup_lower for word in ["recent", "latest", "current", "new"]):
            followup_type = "temporal_filter"
        elif any(word in followup_lower for word in ["collaborate", "work with", "colleagues"]):
            followup_type = "relationship_exploration"
        elif len(set(original_query.lower().split()) & set(followup_query.lower().split())) > 2:
            followup_type = "entity_continuation"
        
        # Context effectiveness based on notes
        context_effectiveness = "unused"
        if context_usage_notes:
            if any(word in context_usage_notes.lower() for word in ["successfully", "effectively", "good"]):
                context_effectiveness = "good"
            elif any(word in context_usage_notes.lower() for word in ["partially", "somewhat"]):
                context_effectiveness = "adequate"
            elif any(word in context_usage_notes.lower() for word in ["failed", "poorly", "didn't"]):
                context_effectiveness = "poor"
            else:
                context_effectiveness = "adequate"
        
        # Efficiency assessment
        efficiency_gain = "none"
        if efficiency_observations and any(word in efficiency_observations.lower() for word in ["faster", "quick", "efficient"]):
            efficiency_gain = "high"
        elif efficiency_observations and any(word in efficiency_observations.lower() for word in ["some", "moderate"]):
            efficiency_gain = "medium"
        
        return {
            "followup_type": followup_type,
            "context_effectiveness": context_effectiveness,
            "efficiency_gain": efficiency_gain,
            "query_similarity": len(set(original_query.lower().split()) & set(followup_query.lower().split())),
            "length_ratio": len(followup_query) / len(original_query) if len(original_query) > 0 else 0,
            "has_pronouns": any(word in followup_query.lower() for word in ["he", "she", "they", "it", "this", "that"]),
            "context_utilization_score": 1.0 if context_effectiveness == "good" else 0.5 if context_effectiveness == "adequate" else 0.0
        }
    
    # =============================================================================
    # DATA MANAGEMENT AND ANALYSIS
    # =============================================================================
    
    def _save_data(self):
        """Save observations data to JSON file."""
        try:
            # Update metadata
            self.observations_data["metadata"]["last_updated"] = datetime.now().isoformat()
            self.observations_data["metadata"]["total_observations"] = sum(
                len(v) for k, v in self.observations_data.items() if k != "metadata" and isinstance(v, list)
            )
            
            # Write with formatting
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.observations_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"⚠️ Error saving methodology observations: {e}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of logged observations."""
        metadata = self.observations_data.get("metadata", {})
        return {
            "total_observations": metadata.get("total_observations", 0),
            "query_starts": len(self.observations_data.get("query_starts", [])),
            "tool_usages": len(self.observations_data.get("tool_usages", [])),
            "replanning_events": len(self.observations_data.get("replanning_events", [])),
            "session_completions": len(self.observations_data.get("session_completions", [])),
            "followup_analyses": len(self.observations_data.get("followup_analyses", [])),
            "created_at": metadata.get("created_at", "Unknown"),
            "last_updated": metadata.get("last_updated", "Never"),
            "logger_type": "standard_fast"
        }
    
    def get_performance_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Get performance metrics from recent observations."""
        recent_sessions = self._get_recent_observations("session_completions", days)
        recent_tools = self._get_recent_observations("tool_usages", days)
        recent_queries = self._get_recent_observations("query_starts", days)
        
        if not recent_sessions:
            return {"error": "No recent session data available"}
        
        # Calculate metrics
        success_rate = sum(1 for s in recent_sessions if s.get("execution_summary", {}).get("final_success") == "success") / len(recent_sessions)
        avg_steps = sum(s.get("execution_summary", {}).get("total_steps", 0) for s in recent_sessions) / len(recent_sessions)
        avg_time = sum(s.get("execution_summary", {}).get("execution_time_seconds", 0) for s in recent_sessions) / len(recent_sessions)
        avg_replanning = sum(s.get("execution_summary", {}).get("replanning_count", 0) for s in recent_sessions) / len(recent_sessions)
        
        # Tool success rate
        tool_success_rate = sum(1 for t in recent_tools if t.get("success", False)) / len(recent_tools) if recent_tools else 0
        
        # Query complexity distribution
        query_complexity = defaultdict(int)
        for q in recent_queries:
            complexity = q.get("heuristic_analysis", {}).get("complexity_level", "unknown")
            query_complexity[complexity] += 1
        
        return {
            "period_days": days,
            "total_sessions": len(recent_sessions),
            "success_rate": round(success_rate, 3),
            "average_steps": round(avg_steps, 1),
            "average_execution_time": round(avg_time, 1),
            "average_replanning_events": round(avg_replanning, 1),
            "tool_success_rate": round(tool_success_rate, 3),
            "query_complexity_distribution": dict(query_complexity),
            "generated_at": datetime.now().isoformat()
        }
    
    def get_pattern_insights(self, days: int = 7) -> Dict[str, Any]:
        """Get pattern insights from recent data using heuristic analysis."""
        recent_data = {
            "queries": self._get_recent_observations("query_starts", days),
            "tools": self._get_recent_observations("tool_usages", days),
            "sessions": self._get_recent_observations("session_completions", days),
            "replanning": self._get_recent_observations("replanning_events", days)
        }
        
        insights = {
            "analysis_period": f"{days} days",
            "total_data_points": sum(len(v) for v in recent_data.values()),
            "generated_at": datetime.now().isoformat()
        }
        
        # Query type patterns
        query_types = defaultdict(int)
        for q in recent_data["queries"]:
            qtype = q.get("heuristic_analysis", {}).get("query_type", "unknown")
            query_types[qtype] += 1
        insights["query_type_patterns"] = dict(query_types)
        
        # Tool effectiveness patterns
        tool_quality = defaultdict(list)
        for t in recent_data["tools"]:
            tool_name = t.get("tool_name", "unknown")
            quality = t.get("effectiveness_metrics", {}).get("quality_score", 0)
            tool_quality[tool_name].append(quality)
        
        tool_avg_quality = {tool: round(sum(scores)/len(scores), 2) for tool, scores in tool_quality.items()}
        insights["tool_effectiveness_patterns"] = tool_avg_quality
        
        # Failure patterns
        failure_categories = defaultdict(int)
        for r in recent_data["replanning"]:
            category = r.get("replanning_metrics", {}).get("failure_category", "unknown")
            failure_categories[category] += 1
        insights["failure_patterns"] = dict(failure_categories)
        
        # Success patterns
        complexity_success = defaultdict(lambda: {"total": 0, "success": 0})
        for s in recent_data["sessions"]:
            complexity = s.get("session_metrics", {}).get("complexity_category", "unknown")
            success = s.get("execution_summary", {}).get("final_success", "") == "success"
            complexity_success[complexity]["total"] += 1
            if success:
                complexity_success[complexity]["success"] += 1
        
        success_by_complexity = {}
        for complexity, data in complexity_success.items():
            if data["total"] > 0:
                success_by_complexity[complexity] = round(data["success"] / data["total"], 3)
        insights["success_by_complexity"] = success_by_complexity
        
        return insights
    
    def _get_recent_observations(self, obs_type: str, days: int) -> List[Dict[str, Any]]:
        """Get observations from recent days."""
        cutoff = datetime.now() - timedelta(days=days)
        observations = self.observations_data.get(obs_type, [])
        
        recent = []
        for obs in observations:
            try:
                obs_time = datetime.fromisoformat(obs['timestamp'])
                if obs_time > cutoff:
                    recent.append(obs)
            except (KeyError, ValueError):
                continue
        
        return recent
    
    def get_recent_observations(self, observation_type: str = "all", limit: int = 10) -> List[Dict]:
        """Get recent observations of a specific type."""
        if observation_type == "all":
            # Combine all observations and sort by timestamp
            all_obs = []
            for obs_type, observations in self.observations_data.items():
                if obs_type != "metadata" and isinstance(observations, list):
                    for obs in observations:
                        obs_with_type = obs.copy()
                        obs_with_type["observation_type"] = obs_type
                        all_obs.append(obs_with_type)
            
            all_obs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return all_obs[:limit]
        else:
            # Get specific type
            observations = self.observations_data.get(observation_type, [])
            return sorted(observations, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]

# Factory function for easy initialization
def create_standard_methodology_logger(log_file: str = "methodology_observations.json") -> StandardMethodologyLogger:
    """Create standard methodology logger instance - fast with no LLM calls."""
    return StandardMethodologyLogger(log_file)

if __name__ == "__main__":
    # Test the standard methodology logger
    print("Testing Standard Methodology Logger (Fast - No LLM calls)...")
    
    try:
        logger = create_standard_methodology_logger("test_standard_methodology.json")
        
        # Test different types of logging with fast analysis
        session_id = "test_session_standard_123"
        
        # Test query start with fast analysis
        logger.log_query_start(session_id, "Find Swedish researchers in explainable AI", is_followup=False)
        
        # Test tool usage with fast analysis
        logger.log_tool_usage(session_id, "search_authors_by_country", "country:Sweden topic:explainable AI", 
                             success=True, 
                             result_content="Found 23 Swedish researchers working on explainable AI, including researchers from KTH, Chalmers, and Linköping University. Key publications in ICML, NeurIPS, and ICLR conferences.",
                             notes="Geographic search worked well, found comprehensive results")
        
        # Test replanning with fast analysis
        logger.log_replanning_event(session_id, "Find Swedish researchers in explainable AI", 2,
                                   "Initial search returned too many general AI researchers, need to focus specifically on explainable AI", 
                                   "Broad AI researcher search in Sweden",
                                   "Specific explainable AI search with Swedish institution filter",
                                   research_context="Previous search found 847 AI researchers in Sweden, but most work on general machine learning rather than explainability specifically.")
        
        # Test session completion with fast analysis
        logger.log_session_complete(session_id, "Find Swedish researchers in explainable AI", 
                                   total_steps=3, replanning_count=1,
                                   final_success="success", execution_time=67.5,
                                   full_research_results="Comprehensive analysis of Swedish explainable AI research landscape. Found 23 active researchers across 8 institutions. Key areas include interpretable machine learning, model transparency, and ethical AI. Leading institutions are KTH Royal Institute of Technology and Chalmers University of Technology.")
        
        # Test follow-up analysis with fast analysis
        logger.log_followup_analysis(session_id, "Find Swedish researchers in explainable AI",
                                   "What are the most recent publications from these researchers?",
                                   context_usage_notes="Successfully used previous researcher list to search for recent publications",
                                   efficiency_observations="Follow-up query completed 70% faster due to existing researcher context")
        
        # Show summary stats
        stats = logger.get_summary_stats()
        print(f"\n📊 Summary Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Show performance metrics
        performance = logger.get_performance_metrics(days=1)
        print(f"\n⚡ Performance Metrics:")
        for key, value in performance.items():
            print(f"  {key}: {value}")
        
        # Show pattern insights
        patterns = logger.get_pattern_insights(days=1)
        print(f"\n🔍 Pattern Insights:")
        for key, value in patterns.items():
            print(f"  {key}: {value}")
        
        print("✅ Standard Methodology Logger test completed!")
        print("⚡ Key benefits:")
        print("  - NO LLM calls during execution (much faster)")
        print("  - Rich heuristic analysis and pattern detection")
        print("  - Structured metrics for later analysis")
        print("  - Performance and efficiency tracking")
        print("  - Pattern insights without AI overhead")
        print("🚀 Ready for high-performance production use!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()