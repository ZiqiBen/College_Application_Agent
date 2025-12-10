"""
State definitions for the Writing Agent LangGraph workflow
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal
from enum import Enum
from datetime import datetime


class DocumentType(str, Enum):
    """Types of documents that can be generated"""
    PERSONAL_STATEMENT = "personal_statement"
    RESUME_BULLETS = "resume_bullets"
    RECOMMENDATION_LETTER = "recommendation_letter"


class ReflectionScore(TypedDict):
    """Individual reflection dimension score"""
    dimension: str
    score: float  # 0-1
    feedback: str


class IterationLog(TypedDict):
    """Log entry for each iteration"""
    iteration: int
    timestamp: str
    draft_length: int
    reflection_scores: List[ReflectionScore]
    dimension_scores: Dict[str, float]  # Raw dimension scores for tracking
    overall_score: float
    suggestions: List[str]
    weakest_dimensions: List[str]  # Dimensions needing most improvement
    weights_used: Dict[str, float]  # Adaptive weights used
    keyword_analysis: Dict[str, Any]  # Keyword integration analysis
    actions_taken: List[str]
    decision_log: Dict[str, Any]  # Decision reasoning


class WritingState(TypedDict):
    """
    Main state for the Writing Agent workflow.
    This state flows through all nodes in the LangGraph.
    
    Enhanced with:
    - Dimension-level score tracking
    - Keyword integration analysis
    - Adaptive weighting information
    """
    
    # ===== Input Information =====
    profile: Dict[str, Any]  # Applicant profile
    program_info: Dict[str, Any]  # Target program information
    document_type: DocumentType  # Type of document to generate
    corpus: Optional[Dict[str, str]]  # Program corpus (chunk_id -> text)
    
    # ===== Configuration =====
    max_iterations: int  # Maximum refinement iterations
    quality_threshold: float  # Quality score threshold (0-1)
    llm_provider: Literal["openai", "anthropic", "qwen"]  # LLM provider
    model_name: str  # Model name
    temperature: float  # LLM temperature
    
    # ===== RAG Retrieval Results =====
    retrieved_chunks: List[str]  # Retrieved text chunks from corpus
    retrieved_chunk_ids: List[str]  # Chunk IDs for provenance
    matched_experiences: List[Dict[str, Any]]  # Relevant experiences from profile
    program_keywords: List[str]  # Extracted program keywords
    
    # ===== ReAct Tool Results =====
    match_score: Optional[float]  # Profile-program match score (0-1)
    required_keywords: List[str]  # Must-have keywords for the document
    special_requirements: Dict[str, Any]  # Program-specific requirements
    tool_call_history: List[Dict[str, Any]]  # History of tool calls
    
    # ===== Generation Content =====
    plan: Optional[str]  # Generation plan (ReWOO)
    current_draft: Optional[str]  # Current version of document
    draft_history: List[str]  # All previous drafts
    
    # ===== Reflection & Evaluation =====
    reflection_scores: List[ReflectionScore]  # Multi-dimensional scores
    overall_quality_score: float  # Overall quality (0-1)
    reflection_feedback: str  # Detailed feedback from reflection
    improvement_suggestions: List[str]  # Specific suggestions for improvement
    dimension_scores: Dict[str, float]  # Individual dimension scores
    weakest_dimensions: List[str]  # Dimensions needing most improvement
    keyword_analysis: Dict[str, Any]  # Keyword integration analysis
    
    # ===== Reflexion Memory =====
    current_iteration: int  # Current iteration number
    iteration_logs: List[IterationLog]  # Complete iteration history
    learned_patterns: Dict[str, Any]  # Patterns learned from previous iterations
    
    # ===== Control Flow =====
    should_continue: bool  # Whether to continue iterating
    should_revise: bool  # Whether revision is needed
    is_complete: bool  # Whether generation is complete
    error_message: Optional[str]  # Error message if any
    
    # ===== Output & Metadata =====
    final_document: Optional[str]  # Final approved document
    generation_metadata: Dict[str, Any]  # Metadata about generation process
    quality_report: Dict[str, Any]  # Final quality report


def create_initial_state(
    profile: Dict[str, Any],
    program_info: Dict[str, Any],
    document_type: DocumentType,
    corpus: Optional[Dict[str, str]] = None,
    max_iterations: int = 3,
    quality_threshold: float = 0.85,
    llm_provider: Literal["openai", "anthropic", "qwen"] = "openai",
    model_name: str = "gpt-4-turbo-preview",
    temperature: float = 0.7
) -> WritingState:
    """
    Create initial state for the writing workflow
    
    Args:
        profile: Applicant profile dictionary
        program_info: Program information dictionary
        document_type: Type of document to generate
        corpus: Optional program corpus
        max_iterations: Maximum refinement iterations
        quality_threshold: Quality threshold for approval
        llm_provider: LLM provider to use
        model_name: Model name
        temperature: LLM temperature
    
    Returns:
        Initial WritingState
    """
    return WritingState(
        # Input
        profile=profile,
        program_info=program_info,
        document_type=document_type,
        corpus=corpus or {},
        
        # Configuration
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
        llm_provider=llm_provider,
        model_name=model_name,
        temperature=temperature,
        
        # RAG Results (empty initially)
        retrieved_chunks=[],
        retrieved_chunk_ids=[],
        matched_experiences=[],
        program_keywords=[],
        
        # Tool Results (empty initially)
        match_score=None,
        required_keywords=[],
        special_requirements={},
        tool_call_history=[],
        
        # Generation Content (empty initially)
        plan=None,
        current_draft=None,
        draft_history=[],
        
        # Reflection (empty initially)
        reflection_scores=[],
        overall_quality_score=0.0,
        reflection_feedback="",
        improvement_suggestions=[],
        dimension_scores={},
        weakest_dimensions=[],
        keyword_analysis={},
        
        # Reflexion Memory
        current_iteration=0,
        iteration_logs=[],
        learned_patterns={},
        
        # Control Flow
        should_continue=True,
        should_revise=False,
        is_complete=False,
        error_message=None,
        
        # Output
        final_document=None,
        generation_metadata={
            "start_time": datetime.now().isoformat(),
            "llm_provider": llm_provider,
            "model_name": model_name
        },
        quality_report={}
    )
