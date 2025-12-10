"""
Prompt templates for the Writing Agent

Enhanced with:
- Adaptive weighting system for reflection
- Score-based revision strategies (fine-tuning, targeted, comprehensive)
- Keyword integration analysis
"""

from .ps_prompts import get_ps_generation_prompt, get_ps_revision_prompt
from .resume_prompts import get_resume_generation_prompt, get_resume_revision_prompt
from .rl_prompts import get_rl_generation_prompt, get_rl_revision_prompt
from .reflection_prompts import (
    get_reflection_prompt,
    parse_reflection_response,
    get_adaptive_weights,
    analyze_keyword_integration
)

__all__ = [
    "get_ps_generation_prompt",
    "get_ps_revision_prompt",
    "get_resume_generation_prompt",
    "get_resume_revision_prompt",
    "get_rl_generation_prompt",
    "get_rl_revision_prompt",
    "get_reflection_prompt",
    "parse_reflection_response",
    "get_adaptive_weights",
    "analyze_keyword_integration"
]
