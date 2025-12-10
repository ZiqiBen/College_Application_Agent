"""
Reflect Node - Self-evaluation and quality assessment
"""

from typing import Dict, Any, List
from ..state import WritingState, ReflectionScore
from ..prompts.reflection_prompts import get_reflection_prompt, parse_reflection_response
from ..config import WritingAgentConfig
from ..llm_utils import get_llm, call_llm
from datetime import datetime


def reflect_node(state: WritingState) -> Dict[str, Any]:
    """
    Reflection node for self-evaluation of generated content.
    
    Uses LLM to critically evaluate the generated document across
    multiple dimensions and provide actionable feedback.
    
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
    
    # Parse reflection response
    parsed_reflection = parse_reflection_response(reflection_response)
    
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
    # Note: We intentionally IGNORE the LLM's approve field here
    # The decision should be based solely on quality_threshold
    
    # Determine if we should continue iterating
    max_iterations = state.get("max_iterations", 3)
    quality_threshold = state.get("quality_threshold", 0.85)
    current_iteration_next = current_iteration + 1
    
    # FIXED LOGIC: Continue iterating if:
    # 1. We haven't reached max iterations
    # 2. The quality score is below the threshold
    # 3. There are improvement suggestions available
    has_improvements = len(suggestions) > 0
    below_threshold = overall_score < quality_threshold
    within_iteration_limit = current_iteration_next < max_iterations
    
    should_continue = within_iteration_limit and below_threshold and has_improvements
    should_revise = should_continue
    is_complete = not should_continue
    
    # Log decision reasoning for debugging
    decision_log = {
        "current_iteration": current_iteration_next,
        "max_iterations": max_iterations,
        "overall_score": overall_score,
        "quality_threshold": quality_threshold,
        "below_threshold": below_threshold,
        "within_iteration_limit": within_iteration_limit,
        "has_improvements": has_improvements,
        "decision": "revise" if should_revise else "complete"
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
        "overall_score": overall_score,
        "suggestions": suggestions,
        "actions_taken": ["Generated draft", "Evaluated quality"],
        "decision_log": decision_log
    }
    
    # Update iteration logs
    iteration_logs = state.get("iteration_logs", [])
    iteration_logs.append(iteration_log)
    
    # Check for improvement from previous iteration
    learned_patterns = state.get("learned_patterns", {})
    if current_iteration > 0 and len(iteration_logs) > 1:
        prev_score = iteration_logs[-2]["overall_score"]
        improvement = overall_score - prev_score
        
        if improvement > 0:
            learned_patterns["last_improvement"] = improvement
            learned_patterns["successful_iteration"] = current_iteration_next
        else:
            learned_patterns["stagnation_count"] = learned_patterns.get("stagnation_count", 0) + 1
            # If we're stagnating (no improvement for 2+ iterations), stop
            if learned_patterns.get("stagnation_count", 0) >= 2:
                print(f"[Reflect Node] Stopping due to stagnation (no improvement for 2 iterations)")
                should_continue = False
                should_revise = False
                is_complete = True
    
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
        "generation_metadata": {
            **state.get("generation_metadata", {}),
            "reflection_completed": True,
            "current_score": overall_score,
            "approved_by_threshold": overall_score >= quality_threshold,
            "decision_log": decision_log
        }
    }
