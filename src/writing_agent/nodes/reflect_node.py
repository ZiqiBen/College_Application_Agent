"""
Reflect Node - Self-evaluation and quality assessment

Enhanced with:
- Adaptive weights based on current quality score
- Dimension-level tracking for targeted improvements
- Intelligent stop conditions (stagnation detection, small improvement threshold)
- Keyword integration analysis
"""

from typing import Dict, Any, List
from ..state import WritingState, ReflectionScore
from ..prompts.reflection_prompts import (
    get_reflection_prompt, 
    parse_reflection_response,
    analyze_keyword_integration,
    get_adaptive_weights
)
from ..config import WritingAgentConfig
from ..llm_utils import get_llm, call_llm
from ..memory import get_memory
from datetime import datetime


# Threshold for considering improvement too small to continue
MIN_IMPROVEMENT_THRESHOLD = 0.02


def reflect_node(state: WritingState) -> Dict[str, Any]:
    """
    Reflection node for self-evaluation of generated content.
    
    Uses LLM to critically evaluate the generated document across
    multiple dimensions and provide actionable feedback.
    
    Enhanced features:
    - Adaptive weights based on quality level
    - Passes dimension scores to revision prompts
    - Intelligent stopping when improvement is minimal
    - Keyword integration analysis
    
    IMPORTANT: The decision to continue iterating is based on the quality_threshold
    set by the user, NOT the LLM's approve field. This ensures the iteration loop
    respects the user's quality requirements.
    """
    
    current_draft = state.get("current_draft", "")
    if not current_draft:
        # No draft to reflect on
        return {
            "overall_quality_score": 0.0,
            "reflection_feedback": "No draft available for reflection",
            "improvement_suggestions": ["Generate initial draft first"],
            "should_continue": True,
            "should_revise": True
        }
    
    document_type = state["document_type"].value
    profile = state["profile"]
    program_info = state["program_info"]
    required_keywords = state.get("required_keywords", [])
    current_iteration = state.get("current_iteration", 0)
    
    # Get previous score for adaptive weighting
    iteration_logs = state.get("iteration_logs", [])
    previous_score = iteration_logs[-1]["overall_score"] if iteration_logs else 0.0
    
    # Get LLM for reflection (use lower temperature for more consistent evaluation)
    llm = get_llm(
        provider=state.get("llm_provider", "openai"),
        model_name=state.get("model_name", WritingAgentConfig.get_model_name()),
        temperature=WritingAgentConfig.TEMPERATURE_REFLECTION
    )
    
    # Generate reflection prompt
    reflection_prompt = get_reflection_prompt(
        document=current_draft,
        document_type=document_type,
        profile=profile,
        program_info=program_info,
        required_keywords=required_keywords,
        iteration=current_iteration + 1
    )
    
    # Call LLM for reflection
    reflection_response = call_llm(llm, reflection_prompt)
    
    # Parse reflection response with adaptive weights
    parsed_reflection = parse_reflection_response(reflection_response, previous_score)
    
    # Analyze keyword integration
    keyword_analysis = analyze_keyword_integration(current_draft, required_keywords)
    
    # Convert scores dict to list of ReflectionScore objects
    reflection_scores: List[ReflectionScore] = []
    for dimension, score in parsed_reflection["scores"].items():
        reflection_scores.append(ReflectionScore(
            dimension=dimension,
            score=score,
            feedback=parsed_reflection.get("specific_issues", [""])[0] if parsed_reflection.get("specific_issues") else ""
        ))
    
    overall_score = parsed_reflection["overall_score"]
    feedback = parsed_reflection["feedback"]
    suggestions = parsed_reflection["improvement_suggestions"]
    weakest_dimensions = parsed_reflection.get("weakest_dimensions", [])
    weights_used = parsed_reflection.get("weights_used", {})
    
    # Determine if we should continue iterating
    max_iterations = state.get("max_iterations", 3)
    quality_threshold = state.get("quality_threshold", 0.85)
    current_iteration_next = current_iteration + 1
    
    # IMPROVED LOGIC: Consider multiple stopping conditions
    has_improvements = len(suggestions) > 0
    below_threshold = overall_score < quality_threshold
    within_iteration_limit = current_iteration_next < max_iterations
    
    should_continue = within_iteration_limit and below_threshold and has_improvements
    should_revise = should_continue
    is_complete = not should_continue
    
    # Check for intelligent stopping conditions
    learned_patterns = state.get("learned_patterns", {})
    stop_reason = None
    
    if current_iteration > 0 and len(iteration_logs) > 0:
        prev_score = iteration_logs[-1]["overall_score"]
        improvement = overall_score - prev_score
        
        learned_patterns["last_improvement"] = improvement
        
        # Condition 1: Very small improvement on high-quality draft
        if improvement < MIN_IMPROVEMENT_THRESHOLD and overall_score >= 0.90:
            print(f"[Reflect Node] Stopping: improvement ({improvement:.3f}) too small for high-quality draft ({overall_score:.3f})")
            should_continue = False
            should_revise = False
            is_complete = True
            stop_reason = "small_improvement_high_quality"
        
        # Condition 2: No improvement at all
        elif improvement <= 0:
            learned_patterns["stagnation_count"] = learned_patterns.get("stagnation_count", 0) + 1
            
            # If we're stagnating (no improvement for 2+ iterations), stop
            if learned_patterns.get("stagnation_count", 0) >= 2:
                print(f"[Reflect Node] Stopping due to stagnation (no improvement for 2 iterations)")
                should_continue = False
                should_revise = False
                is_complete = True
                stop_reason = "stagnation"
        else:
            # Reset stagnation count on improvement
            learned_patterns["stagnation_count"] = 0
            learned_patterns["successful_iteration"] = current_iteration_next
        
        # Condition 3: Same dimensions consistently weak (stuck dimensions)
        prev_weak_dims = learned_patterns.get("prev_weak_dimensions", [])
        if set(weakest_dimensions) == set(prev_weak_dims) and len(weakest_dimensions) > 0:
            stuck_count = learned_patterns.get("stuck_dimensions_count", 0) + 1
            learned_patterns["stuck_dimensions_count"] = stuck_count
            
            if stuck_count >= 2:
                print(f"[Reflect Node] Same dimensions stuck for 2 iterations: {weakest_dimensions}")
                # Don't stop, but flag this for potential strategy change
                learned_patterns["stuck_dimensions"] = weakest_dimensions
        else:
            learned_patterns["stuck_dimensions_count"] = 0
        
        learned_patterns["prev_weak_dimensions"] = weakest_dimensions
    
    # Log decision reasoning for debugging
    decision_log = {
        "current_iteration": current_iteration_next,
        "max_iterations": max_iterations,
        "overall_score": overall_score,
        "quality_threshold": quality_threshold,
        "below_threshold": below_threshold,
        "within_iteration_limit": within_iteration_limit,
        "has_improvements": has_improvements,
        "decision": "revise" if should_revise else "complete",
        "stop_reason": stop_reason,
        "weights_used": weights_used,
        "weakest_dimensions": weakest_dimensions,
        "keyword_coverage": keyword_analysis["overall_integration_quality"]
    }
    print(f"[Reflect Node] Decision: {decision_log}")
    
    # Create iteration log entry
    iteration_log = {
        "iteration": current_iteration_next,
        "timestamp": datetime.now().isoformat(),
        "draft_length": len(current_draft.split()),
        "reflection_scores": [
            {"dimension": rs["dimension"], "score": rs["score"], "feedback": rs["feedback"]}
            for rs in reflection_scores
        ],
        "dimension_scores": parsed_reflection["scores"],
        "overall_score": overall_score,
        "suggestions": suggestions,
        "weakest_dimensions": weakest_dimensions,
        "weights_used": weights_used,
        "keyword_analysis": keyword_analysis,
        "actions_taken": ["Generated draft", "Evaluated quality"],
        "decision_log": decision_log
    }
    
    # Update iteration logs
    iteration_logs = state.get("iteration_logs", [])
    iteration_logs.append(iteration_log)
    
    # Record patterns in memory for learning
    if is_complete and overall_score >= quality_threshold:
        memory = get_memory()
        # Record successful dimension strategies
        for dim, score in parsed_reflection["scores"].items():
            if score >= 0.85:
                memory.record_issue(
                    document_type=document_type,
                    issue_type=f"{dim}_success",
                    resolution=f"Achieved {score:.2f} on {dim}"
                )
    
    return {
        "reflection_scores": reflection_scores,
        "overall_quality_score": overall_score,
        "reflection_feedback": feedback,
        "improvement_suggestions": suggestions,
        "current_iteration": current_iteration_next,
        "iteration_logs": iteration_logs,
        "learned_patterns": learned_patterns,
        "should_continue": should_continue,
        "should_revise": should_revise,
        "is_complete": is_complete,
        # New fields for enhanced revision
        "dimension_scores": parsed_reflection["scores"],
        "weakest_dimensions": weakest_dimensions,
        "keyword_analysis": keyword_analysis,
        "generation_metadata": {
            **state.get("generation_metadata", {}),
            "reflection_completed": True,
            "current_score": overall_score,
            "approved_by_threshold": overall_score >= quality_threshold,
            "decision_log": decision_log,
            "weights_used": weights_used
        }
    }
