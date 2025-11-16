"""
ReAct Node - Reasoning and Acting with tool calls for content generation
"""

from typing import Dict, Any
from ..state import WritingState, DocumentType
from ..tools.match_calculator import calculate_match_score_direct
from ..tools.requirement_checker import check_program_requirements_direct
from ..prompts import (
    get_ps_generation_prompt,
    get_resume_generation_prompt,
    get_rl_generation_prompt
)
from ..config import WritingAgentConfig
from ..llm_utils import get_llm, call_llm


def react_node(state: WritingState) -> Dict[str, Any]:
    """
    ReAct (Reasoning + Acting) node for content generation.
    
    This node:
    1. Calls tools to gather necessary information
    2. Reasons about what content to generate
    3. Generates the initial draft or refined version
    """
    
    profile = state["profile"]
    program_info = state["program_info"]
    document_type = state["document_type"]
    program_keywords = state.get("program_keywords", [])
    retrieved_chunks = state.get("retrieved_chunks", [])
    
    # Tool Phase: Call tools to gather information
    tool_results = {}
    
    # Tool 1: Calculate match score
    match_result = calculate_match_score_direct(
        profile=profile,
        program_info=program_info,
        program_keywords=program_keywords
    )
    tool_results["match_score"] = match_result
    
    # Tool 2: Check program requirements
    requirements_result = check_program_requirements_direct(
        program_info=program_info
    )
    tool_results["requirements"] = requirements_result
    
    # Tool 3: Determine required keywords
    required_keywords = program_keywords[:12]  # Use top keywords
    
    # Log tool calls
    tool_call_history = state.get("tool_call_history", [])
    tool_call_history.append({
        "iteration": state.get("current_iteration", 0),
        "tools_called": ["match_calculator", "requirement_checker", "keyword_extractor"],
        "results_summary": {
            "match_score": match_result.get("overall_score"),
            "requirements_found": len(requirements_result.get("must_mention", [])),
            "keywords_count": len(required_keywords)
        }
    })
    
    # Generation Phase: Generate content using LLM
    llm = get_llm(
        provider=state.get("llm_provider", "openai"),
        model_name=state.get("model_name", WritingAgentConfig.get_model_name()),
        temperature=state.get("temperature", WritingAgentConfig.TEMPERATURE)
    )
    
    # Select appropriate prompt based on document type
    if document_type == DocumentType.PERSONAL_STATEMENT:
        prompt = get_ps_generation_prompt(
            profile=profile,
            program_info=program_info,
            retrieved_context=retrieved_chunks,
            match_analysis=match_result,
            required_keywords=required_keywords
        )
    elif document_type == DocumentType.RESUME_BULLETS:
        prompt = get_resume_generation_prompt(
            profile=profile,
            required_keywords=required_keywords,
            match_analysis=match_result
        )
    else:  # RECOMMENDATION_LETTER
        prompt = get_rl_generation_prompt(
            profile=profile,
            program_info=program_info,
            retrieved_context=retrieved_chunks,
            required_keywords=required_keywords
        )
    
    # Call LLM to generate content
    generated_content = call_llm(llm, prompt)
    
    # Add to draft history
    draft_history = state.get("draft_history", [])
    draft_history.append(generated_content)
    
    return {
        "current_draft": generated_content,
        "draft_history": draft_history,
        "match_score": match_result.get("overall_score"),
        "required_keywords": required_keywords,
        "special_requirements": requirements_result,
        "tool_call_history": tool_call_history,
        "should_revise": False,  # Will be determined by reflect node
        "generation_metadata": {
            **state.get("generation_metadata", {}),
            "generation_completed": True,
            "draft_count": len(draft_history)
        }
    }
