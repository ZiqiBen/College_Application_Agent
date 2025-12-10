"""
Data models for Program Matching Service
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class MatchDimension(str, Enum):
    """Matching evaluation dimensions"""
    ACADEMIC = "academic"
    SKILLS = "skills"
    EXPERIENCE = "experience"
    GOALS = "goals"
    REQUIREMENTS = "requirements"


class MatchLevel(str, Enum):
    """Match quality levels"""
    EXCELLENT = "excellent"  # 0.8+
    GOOD = "good"           # 0.6-0.8
    MODERATE = "moderate"   # 0.4-0.6
    WEAK = "weak"          # <0.4


@dataclass
class DimensionScore:
    """Individual dimension score details"""
    dimension: str
    score: float  # 0-1
    weight: float
    details: Dict[str, Any] = field(default_factory=dict)
    contributing_factors: List[str] = field(default_factory=list)


@dataclass
class ProgramMatch:
    """Program matching result"""
    program_id: str
    program_name: str
    university: str
    overall_score: float  # 0-1
    match_level: MatchLevel
    
    # Dimension scores
    dimension_scores: Dict[str, DimensionScore]
    
    # Student strengths and gaps
    strengths: List[str]
    gaps: List[str]
    fit_reasons: List[str]  # Personalized "Why This Program Fits You" reasons
    recommendations: List[str]
    
    # Detailed explanation
    explanation: str
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchingResult:
    """Complete matching result"""
    student_profile_summary: Dict[str, Any]
    total_programs_evaluated: int
    matches: List[ProgramMatch]
    top_k: int
    min_score_threshold: float
    matching_timestamp: str
    
    # Global insights
    overall_insights: Dict[str, Any] = field(default_factory=dict)


# Pydantic models for API requests/responses

class MatchingRequest(BaseModel):
    """Matching request model"""
    profile: Dict[str, Any] = Field(..., description="Student profile information")
    top_k: int = Field(5, ge=1, le=20, description="Return top K matching programs")
    min_score: float = Field(0.5, ge=0.0, le=1.0, description="Minimum match score threshold")
    
    # Optional: custom weights
    custom_weights: Optional[Dict[str, float]] = Field(None, description="Custom dimension weights")
    
    # Optional: filters
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters (e.g., country, tuition range)")
    
    # Optional: use LLM-enhanced explanation
    use_llm_explanation: bool = Field(True, description="Use LLM to generate detailed explanations")


class DimensionScoreResponse(BaseModel):
    """Dimension score response"""
    dimension: str
    score: float
    weight: float
    details: Dict[str, Any] = {}
    contributing_factors: List[str] = []


class ProgramMatchResponse(BaseModel):
    """Single program match response"""
    program_id: str
    program_name: str
    university: str
    overall_score: float
    match_level: str
    
    dimension_scores: Dict[str, DimensionScoreResponse]
    
    strengths: List[str]
    gaps: List[str]
    fit_reasons: List[str] = []  # Personalized "Why This Program Fits You" reasons
    recommendations: List[str]
    
    explanation: str
    
    metadata: Dict[str, Any] = {}
    
    # NEW: Complete program details for Writing Agent
    program_details: Optional[Dict[str, Any]] = None


class MatchingResponse(BaseModel):
    """Complete matching response"""
    success: bool
    message: str
    
    student_profile_summary: Dict[str, Any]
    total_programs_evaluated: int
    matches: List[ProgramMatchResponse]
    
    overall_insights: Dict[str, Any] = {}
    
    matching_timestamp: str
    processing_time_seconds: float


# Program data model

class ProgramData(BaseModel):
    """Program data structure (loaded from corpus)"""
    program_id: str
    name: str
    university: str
    degree_type: str = "Master"  # Master, PhD, etc.
    field: str
    
    # Requirements
    min_gpa: float = 3.0
    required_skills: List[str] = []
    prerequisite_courses: List[str] = []
    language_requirements: Dict[str, Any] = {}
    
    # Program features
    focus_areas: List[str] = []
    core_courses: List[str] = []
    career_outcomes: str = ""
    duration: str = ""
    
    # Other info
    tuition: Optional[str] = None
    location: Optional[str] = None
    ranking: Optional[int] = None
    
    # Raw text (for embeddings)
    description_text: str = ""
    
    class Config:
        extra = "allow"  # Allow extra fields