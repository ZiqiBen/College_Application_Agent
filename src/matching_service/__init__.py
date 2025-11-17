"""
Program Matching Service

Analyzes student background and matches them with the most suitable programs
based on multiple dimensions: academic background, skills, experience, goals, and requirements.
"""

from .matcher import ProgramMatcher
from .scorer import DimensionScorer
from .explainer import MatchExplainer
from .models import (
    ProgramMatch,
    MatchingResult,
    MatchLevel,
    MatchDimension,
    DimensionScore,
    MatchingRequest,
    MatchingResponse,
    ProgramMatchResponse,
    DimensionScoreResponse
)

__version__ = "1.0.0"

__all__ = [
    "ProgramMatcher",
    "DimensionScorer",
    "MatchExplainer",
    "ProgramMatch",
    "MatchingResult",
    "MatchLevel",
    "MatchDimension",
    "DimensionScore",
    "MatchingRequest",
    "MatchingResponse",
    "ProgramMatchResponse",
    "DimensionScoreResponse"
]