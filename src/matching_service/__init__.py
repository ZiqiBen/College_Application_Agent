"""
Program Matching Service

Analyzes student background and matches them with the most suitable programs
based on multiple dimensions: academic background, skills, experience, goals, and requirements.

Supports two dataset versions:
- V1 (Legacy): Original corpus format with sections structure
- V2 (New): Enhanced dataset with extracted_fields, courses with descriptions, etc.
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

# V2 imports for new dataset
from .matcher_v2 import ProgramMatcherV2, convert_match_to_response
from .scorer_v2 import DimensionScorerV2
from .models_v2 import (
    ProgramDataV2,
    ProgramMatchV2,
    MatchingResultV2,
    MatchLevelV2,
    MatchDimensionV2,
    DimensionScoreV2,
    MatchingRequestV2,
    MatchingResponseV2,
    ProgramMatchResponseV2,
    DimensionScoreResponseV2,
    CourseInfo,
    ApplicationRequirementsV2,
    TrainingOutcomesV2,
    ProgramBackgroundV2
)

__version__ = "2.0.0"

__all__ = [
    # V1 (Legacy)
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
    "DimensionScoreResponse",
    # V2 (New dataset)
    "ProgramMatcherV2",
    "DimensionScorerV2",
    "convert_match_to_response",
    "ProgramDataV2",
    "ProgramMatchV2",
    "MatchingResultV2",
    "MatchLevelV2",
    "MatchDimensionV2",
    "DimensionScoreV2",
    "MatchingRequestV2",
    "MatchingResponseV2",
    "ProgramMatchResponseV2",
    "DimensionScoreResponseV2",
    "CourseInfo",
    "ApplicationRequirementsV2",
    "TrainingOutcomesV2",
    "ProgramBackgroundV2"
]