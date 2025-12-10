"""
Revise Node - Content improvement based on reflection feedback

Enhanced with:
- Dimension-level scores passed to prompts for targeted improvement
- Previous draft comparison for regression prevention
- Adaptive revision strategies based on current score
- Memory integration for learning from past improvements
"""

from typing import Dict, Any
from ..state import WritingState, DocumentType
from ..prompts import (
    get_ps_revision_prompt,
    get_resume_revision_prompt,
    get_rl_revision_prompt
)
from ..config import WritingAgentConfig
from ..llm_utils import get_llm, call_llm
from ..memory import get_memory


def revise_node(state: WritingState) -> Dict[str, Any]:
    """
    Revise node for improving generated content based on reflection feedback.
    
    Takes the current draft and reflection feedback, then generates
    an improved version addressing the identified issues.
    
    Enhanced features:
    - Passes dimension scores for targeted improvement focus
    - Includes previous draft for comparison (prevents regression)
    - Uses adaptive revision strategies based on current quality
    - Integrates memory for learning from past successes
    """
    
    current_draft = state.get("current_draft", "")
    reflection_feedback = state.get("reflection_feedback", "")
    improvement_suggestions = state.get("improvement_suggestions", [])
    current_iteration = state.get("current_iteration", 0)
    overall_score = state.get("overall_quality_score", 0.0)
    
    # Get enhanced information from reflect node
    dimension_scores = state.get("dimension_scores", {})
    weakest_dimensions = state.get("weakest_dimensions", [])
    
    # Get previous draft for comparison (to prevent regression)
    draft_history = state.get("draft_history", [])
    previous_draft = draft_history[-2] if len(draft_history) >= 2 else None
    
    print(f"[Revise Node] Starting revision iteration {current_iteration}")
    print(f"[Revise Node] Current score: {overall_score:.3f}")
    print(f"[Revise Node] Weakest dimensions: {weakest_dimensions}")
    print(f"[Revise Node] Feedback: {reflection_feedback[:200]}...")
    print(f"[Revise Node] Suggestions: {improvement_suggestions}")
    
    if not current_draft:
        print("[Revise Node] No draft available to revise")
        return {
            "error_message": "No draft available to revise",
            "should_continue": False,
            "is_complete": True
        }
    
    # Even if no specific suggestions, we can still try to improve based on feedback
    if not improvement_suggestions and not reflection_feedback:
        print("[Revise Node] No suggestions or feedback, marking complete")
        return {
            "final_document": current_draft,
            "should_continue": False,
            "is_complete": True
        }
    
    # If we have feedback but no suggestions, create targeted suggestions based on weak dimensions
    if not improvement_suggestions and reflection_feedback:
        improvement_suggestions = _generate_targeted_suggestions(weakest_dimensions)
        print(f"[Revise Node] Created targeted suggestions: {improvement_suggestions}")
    
    # Get relevant patterns from memory
    memory = get_memory()
    document_type_str = state["document_type"].value
    relevant_patterns = memory.get_relevant_patterns(document_type_str, overall_score)
    memory_suggestions = memory.suggest_strategies(document_type_str, improvement_suggestions)
    
    # Combine with memory-based suggestions if available
    if memory_suggestions:
        print(f"[Revise Node] Memory suggests: {memory_suggestions[:3]}")
        # Add memory suggestions that aren't already in the list
        for suggestion in memory_suggestions[:2]:
            if suggestion not in improvement_suggestions:
                improvement_suggestions.append(f"[From past success] {suggestion}")
    
    profile = state["profile"]
    program_info = state["program_info"]
    document_type = state["document_type"]
    
    # Get LLM for revision
    llm = get_llm(
        provider=state.get("llm_provider", "openai"),
        model_name=state.get("model_name", WritingAgentConfig.get_model_name()),
        temperature=state.get("temperature", WritingAgentConfig.TEMPERATURE)
    )
    
    # Generate revision prompt based on document type with enhanced parameters
    if document_type == DocumentType.PERSONAL_STATEMENT:
        revision_prompt = get_ps_revision_prompt(
            current_draft=current_draft,
            reflection_feedback=reflection_feedback,
            improvement_suggestions=improvement_suggestions,
            profile=profile,
            program_info=program_info,
            iteration=current_iteration,
            current_score=overall_score,
            dimension_scores=dimension_scores,
            weakest_dimensions=weakest_dimensions,
            previous_draft=previous_draft
        )
    elif document_type == DocumentType.RESUME_BULLETS:
        revision_prompt = get_resume_revision_prompt(
            current_draft=current_draft,
            reflection_feedback=reflection_feedback,
            improvement_suggestions=improvement_suggestions,
            profile=profile,
            required_keywords=state.get("required_keywords", []),
            iteration=current_iteration,
            current_score=overall_score,
            dimension_scores=dimension_scores,
            weakest_dimensions=weakest_dimensions
        )
    else:  # RECOMMENDATION_LETTER
        revision_prompt = get_rl_revision_prompt(
            current_draft=current_draft,
            reflection_feedback=reflection_feedback,
            improvement_suggestions=improvement_suggestions,
            profile=profile,
            program_info=program_info,
            iteration=current_iteration,
            current_score=overall_score,
            dimension_scores=dimension_scores,
            weakest_dimensions=weakest_dimensions
        )
    
    # Call LLM to generate revised version
    revised_content = call_llm(llm, revision_prompt)
    
    # Add to draft history
    draft_history = state.get("draft_history", [])
    draft_history.append(revised_content)
    
    # Update learned patterns
    learned_patterns = state.get("learned_patterns", {})
    learned_patterns["revision_count"] = learned_patterns.get("revision_count", 0) + 1
    learned_patterns["last_revision_focus"] = weakest_dimensions
    
    # Track which suggestions were addressed in this revision
    iteration_logs = state.get("iteration_logs", [])
    if iteration_logs:
        iteration_logs[-1]["actions_taken"].append(
            f"Revised addressing {len(improvement_suggestions)} suggestions (focus: {weakest_dimensions})"
        )
        iteration_logs[-1]["revision_focus"] = {
            "weakest_dimensions": weakest_dimensions,
            "dimension_scores": dimension_scores,
            "revision_mode": _get_revision_mode(overall_score)
        }
    
    return {
        "current_draft": revised_content,
        "draft_history": draft_history,
        "learned_patterns": learned_patterns,
        "iteration_logs": iteration_logs,
        "should_revise": False,  # Will be re-evaluated by reflect node
        "generation_metadata": {
            **state.get("generation_metadata", {}),
            "revision_completed": True,
            "revision_count": learned_patterns.get("revision_count", 1),
            "revision_mode": _get_revision_mode(overall_score),
            "focused_dimensions": weakest_dimensions
        }
    }


def _generate_targeted_suggestions(weakest_dimensions: list) -> list:
    """
    Generate targeted improvement suggestions based on weakest dimensions.
    
    Args:
        weakest_dimensions: List of dimension names that need improvement
    
    Returns:
        List of actionable suggestions
    """
    suggestions = []
    
    dimension_suggestions = {
        "keyword_coverage": "Naturally integrate more required keywords into existing sentences",
        "personalization": "Add specific details, numbers, and concrete examples from your experience",
        "coherence": "Improve transitions between paragraphs and ensure logical flow",
        "program_alignment": "Add specific references to program courses, faculty, or unique features",
        "persuasiveness": "Strengthen impact statements and add compelling evidence"
    }
    
    for dim in weakest_dimensions:
        if dim in dimension_suggestions:
            suggestions.append(dimension_suggestions[dim])
    
    # Add a general fallback if no specific dimensions identified
    if not suggestions:
        suggestions = [
            "Improve overall quality based on feedback",
            "Enhance personalization and specificity",
            "Strengthen program alignment"
        ]
    
    return suggestions


def _get_revision_mode(score: float) -> str:
    """
    Determine revision mode based on current score.
    
    Args:
        score: Current quality score
    
    Returns:
        Revision mode string
    """
    if score >= 0.85:
        return "fine_tuning"
    elif score >= 0.70:
        return "targeted"
    else:
        return "comprehensive"
