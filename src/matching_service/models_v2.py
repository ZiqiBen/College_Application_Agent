"""
Data models for Program Matching Service V2 (New Dataset)

This module defines data models for the new V2 dataset which has a different
structure compared to the legacy corpus format. The V2 dataset includes:
- More detailed program information (courses with descriptions, etc.)
- Nested extracted_fields structure with application_requirements, program_background, etc.
- Chunks for text segments with embeddings support
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class MatchDimensionV2(str, Enum):
    """Matching evaluation dimensions for V2"""
    ACADEMIC = "academic"
    SKILLS = "skills"
    EXPERIENCE = "experience"
    GOALS = "goals"
    REQUIREMENTS = "requirements"
    CURRICULUM = "curriculum"  # New: curriculum alignment


class MatchLevelV2(str, Enum):
    """Match quality levels"""
    EXCELLENT = "excellent"  # 0.85+
    GOOD = "good"            # 0.70-0.85
    MODERATE = "moderate"    # 0.55-0.70
    FAIR = "fair"            # 0.40-0.55
    WEAK = "weak"            # <0.40


@dataclass
class CourseInfo:
    """Course information from V2 dataset"""
    name: str
    description: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Any) -> Optional['CourseInfo']:
        """Create CourseInfo from dict or string"""
        if data is None:
            return None
        if isinstance(data, str):
            return cls(name=data, description=None)
        if isinstance(data, dict):
            return cls(
                name=data.get("name", "Unknown Course"),
                description=data.get("description")
            )
        return None


@dataclass
class ApplicationRequirementsV2:
    """Application requirements from V2 dataset"""
    academic_background: Optional[str] = None
    prerequisites: Optional[str] = None
    gre: Optional[str] = None
    english_tests: Optional[str] = None
    research_experience: Optional[str] = None
    work_experience: Optional[str] = None
    documents: Optional[str] = None
    summary: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> 'ApplicationRequirementsV2':
        """Create from dictionary with null handling"""
        if data is None:
            return cls()
        return cls(
            academic_background=data.get("academic_background"),
            prerequisites=data.get("prerequisites"),
            gre=data.get("gre"),
            english_tests=data.get("english_tests"),
            research_experience=data.get("research_experience"),
            work_experience=data.get("work_experience"),
            documents=data.get("documents"),
            summary=data.get("summary")
        )


@dataclass
class ProgramBackgroundV2:
    """Program background information from V2 dataset"""
    mission: Optional[str] = None
    environment: Optional[str] = None
    faculty: Optional[str] = None
    resources: Optional[str] = None
    summary: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> 'ProgramBackgroundV2':
        """Create from dictionary with null handling"""
        if data is None:
            return cls()
        return cls(
            mission=data.get("mission"),
            environment=data.get("environment"),
            faculty=data.get("faculty"),
            resources=data.get("resources"),
            summary=data.get("summary")
        )


@dataclass
class TrainingOutcomesV2:
    """Training outcomes from V2 dataset"""
    goals: Optional[str] = None
    career_paths: Optional[str] = None
    research_orientation: Optional[str] = None
    professional_orientation: Optional[str] = None
    summary: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> 'TrainingOutcomesV2':
        """Create from dictionary with null handling"""
        if data is None:
            return cls()
        return cls(
            goals=data.get("goals"),
            career_paths=data.get("career_paths"),
            research_orientation=data.get("research_orientation"),
            professional_orientation=data.get("professional_orientation"),
            summary=data.get("summary")
        )


@dataclass
class ChunkInfo:
    """Text chunk information for RAG"""
    chunk_id: str
    text: str
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0
    vector_id: Optional[str] = None


@dataclass
class DimensionScoreV2:
    """Individual dimension score details for V2"""
    dimension: str
    score: float  # 0-1
    weight: float
    details: Dict[str, Any] = field(default_factory=dict)
    contributing_factors: List[str] = field(default_factory=list)
    matched_items: List[str] = field(default_factory=list)  # New: specific matched items
    missing_items: List[str] = field(default_factory=list)  # New: what's missing


@dataclass
class ProgramDataV2:
    """Program data structure for V2 dataset"""
    # Core identifiers
    program_id: str
    source_url: str
    crawl_date: str
    title: Optional[str] = None
    
    # Extracted fields
    program_name: Optional[str] = None
    school: Optional[str] = None
    department: Optional[str] = None
    duration: Optional[str] = None
    tuition: Optional[str] = None
    contact_email: Optional[str] = None
    language: Optional[str] = None
    
    # Structured data
    courses: List[CourseInfo] = field(default_factory=list)
    application_requirements: ApplicationRequirementsV2 = field(default_factory=ApplicationRequirementsV2)
    program_background: ProgramBackgroundV2 = field(default_factory=ProgramBackgroundV2)
    training_outcomes: TrainingOutcomesV2 = field(default_factory=TrainingOutcomesV2)
    
    # Text content
    raw_text: str = ""
    chunks: List[ChunkInfo] = field(default_factory=list)
    
    # Metadata
    others: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'ProgramDataV2':
        """Create ProgramDataV2 from JSON data with comprehensive null handling"""
        extracted = data.get("extracted_fields", {}) or {}
        
        # Parse courses
        courses_data = extracted.get("courses") or []
        courses = []
        for c in courses_data:
            course_info = CourseInfo.from_dict(c)
            if course_info:
                courses.append(course_info)
        
        # Parse chunks
        chunks_data = data.get("chunks", []) or []
        chunks = []
        for chunk in chunks_data:
            if chunk:
                chunks.append(ChunkInfo(
                    chunk_id=chunk.get("chunk_id", ""),
                    text=chunk.get("text", ""),
                    char_start=chunk.get("char_start", 0),
                    char_end=chunk.get("char_end", 0),
                    token_count=chunk.get("token_count", 0),
                    vector_id=chunk.get("vector_id")
                ))
        
        return cls(
            program_id=data.get("id", ""),
            source_url=data.get("source_url", ""),
            crawl_date=data.get("crawl_date", ""),
            title=data.get("title"),
            program_name=extracted.get("program_name"),
            school=extracted.get("school"),
            department=extracted.get("department"),
            duration=extracted.get("duration"),
            tuition=extracted.get("tuition"),
            contact_email=extracted.get("contact_email"),
            language=extracted.get("language"),
            courses=courses,
            application_requirements=ApplicationRequirementsV2.from_dict(
                extracted.get("application_requirements")
            ),
            program_background=ProgramBackgroundV2.from_dict(
                extracted.get("program_background")
            ),
            training_outcomes=TrainingOutcomesV2.from_dict(
                extracted.get("training_outcomes")
            ),
            raw_text=data.get("raw_text", ""),
            chunks=chunks,
            others=extracted.get("others")
        )
    
    def get_display_name(self) -> str:
        """Get best display name for the program"""
        if self.program_name:
            return self.program_name
        if self.title:
            return self.title
        return "Unknown Program"
    
    def get_school_name(self) -> str:
        """Get best school name"""
        if self.school:
            return self.school
        # Try to extract from source_url
        if self.source_url:
            url = self.source_url.lower()
            for pattern, name in UNIVERSITY_PATTERNS.items():
                if pattern in url:
                    return name
        return "Unknown University"
    
    def get_full_text(self) -> str:
        """Get concatenated text content for analysis"""
        parts = []
        
        if self.raw_text:
            parts.append(self.raw_text)
        
        # Add structured information
        if self.program_background.mission:
            parts.append(f"Mission: {self.program_background.mission}")
        if self.program_background.summary:
            parts.append(f"Program Summary: {self.program_background.summary}")
        if self.training_outcomes.goals:
            parts.append(f"Training Goals: {self.training_outcomes.goals}")
        if self.training_outcomes.career_paths:
            parts.append(f"Career Paths: {self.training_outcomes.career_paths}")
        if self.application_requirements.summary:
            parts.append(f"Requirements: {self.application_requirements.summary}")
        
        # Add course names
        if self.courses:
            course_names = [c.name for c in self.courses if c.name]
            if course_names:
                parts.append(f"Courses: {', '.join(course_names)}")
        
        return "\n\n".join(parts)
    
    def get_course_descriptions(self) -> str:
        """Get all course descriptions concatenated"""
        descriptions = []
        for course in self.courses:
            if course.description:
                descriptions.append(f"{course.name}: {course.description}")
        return "\n".join(descriptions)


@dataclass
class ProgramMatchV2:
    """Program matching result for V2"""
    program_id: str
    program_name: str
    university: str
    overall_score: float  # 0-1
    match_level: MatchLevelV2
    
    # Dimension scores
    dimension_scores: Dict[str, DimensionScoreV2]
    
    # Student fit analysis
    strengths: List[str]
    gaps: List[str]
    fit_reasons: List[str]  # "Why This Program Fits You"
    recommendations: List[str]
    
    # Curriculum alignment (V2 specific)
    matched_courses: List[str] = field(default_factory=list)
    relevant_courses: List[str] = field(default_factory=list)
    
    # Detailed explanation
    explanation: str = ""
    
    # Full program data for Writing Agent
    program_data: Optional[ProgramDataV2] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchingResultV2:
    """Complete matching result for V2"""
    student_profile_summary: Dict[str, Any]
    total_programs_evaluated: int
    matches: List[ProgramMatchV2]
    top_k: int
    min_score_threshold: float
    matching_timestamp: str
    dataset_version: str = "v2"
    
    # Global insights
    overall_insights: Dict[str, Any] = field(default_factory=dict)


# University name patterns for extraction from URL
UNIVERSITY_PATTERNS = {
    "stanford": "Stanford University",
    "mit.edu": "MIT",
    "cmu.edu": "Carnegie Mellon University",
    "berkeley": "UC Berkeley",
    "columbia": "Columbia University",
    "harvard": "Harvard University",
    "yale": "Yale University",
    "princeton": "Princeton University",
    "cornell": "Cornell University",
    "upenn": "University of Pennsylvania",
    "duke": "Duke University",
    "northwestern": "Northwestern University",
    "nyu.edu": "New York University",
    "brown": "Brown University",
    "dartmouth": "Dartmouth College",
    "caltech": "California Institute of Technology",
    "rice": "Rice University",
    "vanderbilt": "Vanderbilt University",
    "jhu.edu": "Johns Hopkins University",
    "uchicago": "University of Chicago",
    "ucla": "UCLA",
    "ucsd": "UC San Diego",
    "ucsb": "UC Santa Barbara",
    "utexas": "University of Texas at Austin",
    "uw.edu": "University of Washington",
    "purdue": "Purdue University",
    "psu.edu": "Penn State University",
}


# Pydantic models for API requests/responses

class MatchingRequestV2(BaseModel):
    """Matching request model for V2"""
    profile: Dict[str, Any] = Field(..., description="Student profile information")
    top_k: int = Field(10, ge=1, le=50, description="Return top K matching programs")
    min_score: float = Field(0.4, ge=0.0, le=1.0, description="Minimum match score threshold")
    
    # V2 specific options
    include_curriculum_analysis: bool = Field(True, description="Include curriculum alignment analysis")
    include_course_recommendations: bool = Field(True, description="Include course recommendations")
    
    # Optional: custom weights
    custom_weights: Optional[Dict[str, float]] = Field(None, description="Custom dimension weights")
    
    # Optional: filters
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters (e.g., university, field)")
    
    # Optional: use LLM-enhanced explanation
    use_llm_explanation: bool = Field(False, description="Use LLM to generate detailed explanations")


class DimensionScoreResponseV2(BaseModel):
    """Dimension score response for V2"""
    dimension: str
    score: float
    weight: float
    details: Dict[str, Any] = {}
    contributing_factors: List[str] = []
    matched_items: List[str] = []
    missing_items: List[str] = []


class ProgramMatchResponseV2(BaseModel):
    """Single program match response for V2"""
    program_id: str
    program_name: str
    university: str
    department: Optional[str] = None
    overall_score: float
    match_level: str
    
    dimension_scores: Dict[str, DimensionScoreResponseV2]
    
    strengths: List[str]
    gaps: List[str]
    fit_reasons: List[str] = []
    recommendations: List[str]
    
    # V2 specific
    matched_courses: List[str] = []
    relevant_courses: List[str] = []
    
    explanation: str = ""
    
    # Program details for Writing Agent
    program_details: Optional[Dict[str, Any]] = None
    
    metadata: Dict[str, Any] = {}


class MatchingResponseV2(BaseModel):
    """Complete matching response for V2"""
    success: bool
    message: str
    dataset_version: str = "v2"
    
    student_profile_summary: Dict[str, Any]
    total_programs_evaluated: int
    matches: List[ProgramMatchResponseV2]
    
    overall_insights: Dict[str, Any] = {}
    
    matching_timestamp: str
    processing_time_seconds: float
