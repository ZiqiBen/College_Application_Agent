"""
Revise Node - Content improvement based on reflection feedback
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


def revise_node(state: WritingState) -> Dict[str, Any]:
    """
    Revise node for improving generated content based on reflection feedback.
    
    Takes the current draft and reflection feedback, then generates
    an improved version addressing the identified issues.
    """
    
    current_draft = state.get("current_draft", "")
    reflection_feedback = state.get("reflection_feedback", "")
    improvement_suggestions = state.get("improvement_suggestions", [])
    current_iteration = state.get("current_iteration", 0)
    
    print(f"[Revise Node] Starting revision iteration {current_iteration}")
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
    
    # If we have feedback but no suggestions, create generic improvement suggestions
    if not improvement_suggestions and reflection_feedback:
        improvement_suggestions = [
            "Improve overall quality based on feedback",
            "Enhance personalization and specificity",
            "Strengthen program alignment"
        ]
        print(f"[Revise Node] Created generic suggestions: {improvement_suggestions}")
    
    profile = state["profile"]
    program_info = state["program_info"]
    document_type = state["document_type"]
    
    # Get LLM for revision
    llm = get_llm(
        provider=state.get("llm_provider", "openai"),
        model_name=state.get("model_name", WritingAgentConfig.get_model_name()),
        temperature=state.get("temperature", WritingAgentConfig.TEMPERATURE)
    )
    
    # Generate revision prompt based on document type
    if document_type == DocumentType.PERSONAL_STATEMENT:
        revision_prompt = get_ps_revision_prompt(
            current_draft=current_draft,
            reflection_feedback=reflection_feedback,
            improvement_suggestions=improvement_suggestions,
            profile=profile,
            program_info=program_info,
            iteration=current_iteration
        )
    elif document_type == DocumentType.RESUME_BULLETS:
        revision_prompt = get_resume_revision_prompt(
            current_draft=current_draft,
            reflection_feedback=reflection_feedback,
            improvement_suggestions=improvement_suggestions,
            profile=profile,
            required_keywords=state.get("required_keywords", []),
            iteration=current_iteration
        )
    else:  # RECOMMENDATION_LETTER
        revision_prompt = get_rl_revision_prompt(
            current_draft=current_draft,
            reflection_feedback=reflection_feedback,
            improvement_suggestions=improvement_suggestions,
            profile=profile,
            program_info=program_info,
            iteration=current_iteration
        )
    
    # Call LLM to generate revised version
    revised_content = call_llm(llm, revision_prompt)
    
    # Add to draft history
    draft_history = state.get("draft_history", [])
    draft_history.append(revised_content)
    
    # Update learned patterns
    learned_patterns = state.get("learned_patterns", {})
    learned_patterns["revision_count"] = learned_patterns.get("revision_count", 0) + 1
    
    # Track which suggestions were addressed in this revision
    iteration_logs = state.get("iteration_logs", [])
    if iteration_logs:
        iteration_logs[-1]["actions_taken"].append(f"Revised addressing {len(improvement_suggestions)} suggestions")
    
    return {
        "current_draft": revised_content,
        "draft_history": draft_history,
        "learned_patterns": learned_patterns,
        "iteration_logs": iteration_logs,
        "should_revise": False,  # Will be re-evaluated by reflect node
        "generation_metadata": {
            **state.get("generation_metadata", {}),
            "revision_completed": True,
            "revision_count": learned_patterns.get("revision_count", 1)
        }
    }
