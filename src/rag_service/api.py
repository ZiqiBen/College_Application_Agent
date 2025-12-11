from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn, os, json, re, time, hashlib
import requests
from bs4 import BeautifulSoup

# Import both generator systems
from .generator import generate_all  # Original simple generator (fallback)
from .multi_agent_generator import generate_all_multi_agent  # New multi-agent system
from .ingest import ingest_corpus
from .retriever_bert import build_query, retrieve_topk

# Import Matching Service (V1 - Legacy)
from ..matching_service import (
    ProgramMatcher,
    MatchingRequest,
    MatchingResponse,
    ProgramMatchResponse,
    DimensionScoreResponse
)
from ..matching_service.explainer import MatchExplainer

# Import Matching Service V2 (New Dataset)
from ..matching_service import (
    ProgramMatcherV2,
    MatchingRequestV2,
    MatchingResponseV2,
    ProgramMatchResponseV2,
    DimensionScoreResponseV2,
    convert_match_to_response
)

# Import new Writing Agent (LangGraph-based)
try:
    from ..writing_agent import create_writing_graph
    from ..writing_agent.state import DocumentType, create_initial_state
    from ..writing_agent.graph import generate_document
    WRITING_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Writing Agent not available: {e}")
    WRITING_AGENT_AVAILABLE = False

app = FastAPI(title="College App Helper API V4.0 - Enhanced with Dual Dataset Support")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# Dataset directories
CORPUS_DIR_V1 = "data/corpus"  # Legacy corpus
CORPUS_DIR_V2 = "data_preparation/dataset/graduate_programs"  # New V2 dataset
OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)

# LLM provider for V2 explainer (can be configured via environment variable)
V2_LLM_PROVIDER = os.environ.get("V2_LLM_PROVIDER", "openai")

# Initialize Program Matchers for both datasets
# V1 Matcher (Legacy)
try:
    PROGRAM_MATCHER = ProgramMatcher(corpus_dir=CORPUS_DIR_V1)
    MATCHER_AVAILABLE = True
    print(f"✅ V1 Program Matcher initialized with {len(PROGRAM_MATCHER.programs)} programs")
except Exception as e:
    print(f"⚠️  Warning: V1 Program matcher initialization failed: {e}")
    MATCHER_AVAILABLE = False

# V2 Matcher (New Dataset) - with LLM-based explainer for personalized fit reasons
try:
    PROGRAM_MATCHER_V2 = ProgramMatcherV2(
        corpus_dir=CORPUS_DIR_V2,
        use_llm_explainer=True,
        llm_provider=V2_LLM_PROVIDER
    )
    MATCHER_V2_AVAILABLE = True
    print(f"✅ V2 Program Matcher initialized with {len(PROGRAM_MATCHER_V2.programs)} programs")
    print(f"✅ V2 Matcher LLM Explainer: {'enabled' if PROGRAM_MATCHER_V2.explainer else 'disabled (fallback to rules)'}")
except Exception as e:
    print(f"⚠️  Warning: V2 Program matcher initialization failed: {e}")
    MATCHER_V2_AVAILABLE = False

class Profile(BaseModel):
    name: str
    major: str
    goals: str
    email: Optional[str] = None
    gpa: Optional[float] = None
    courses: Optional[List[str]] = []
    skills: Optional[List[str]] = []
    experiences: Optional[List[dict]] = []

class GenerateRequest(BaseModel):
    profile: Profile
    resume_text: str
    program_text: Optional[str] = None
    program_url: Optional[str] = None
    topk: int = 3
    # Multi-agent system parameters
    use_multi_agent: bool = True  # Default to using multi-agent system
    max_iterations: int = 3
    critique_threshold: float = 0.8
    fallback_on_error: bool = True  # Fallback to simple generator if multi-agent fails


class WritingAgentRequest(BaseModel):
    """Request model for new Writing Agent (LangGraph-based)"""
    profile: Profile
    resume_text: str
    program_text: Optional[str] = None
    program_url: Optional[str] = None
    document_type: str = "personal_statement"  # "personal_statement", "resume_bullets", "recommendation_letter"
    # LLM configuration
    llm_provider: str = "openai"  # "openai", "anthropic", "qwen"
    model_name: Optional[str] = None
    temperature: float = 0.7
    # Generation parameters
    max_iterations: int = 3
    quality_threshold: float = 0.85
    # RAG parameters
    use_corpus: bool = True
    retrieval_topk: int = 5

def cache_path(url: str) -> str:
    """Generate cache file path for a given URL using SHA1 hash."""
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return os.path.join(OUT_DIR, f"fetch_{h}.txt")

def polite_fetch(url: str, timeout=15) -> str:
    """
    Fetch the content of a URL with polite caching (7 days).
    Returns cached content if available and fresh, otherwise fetches and caches.
    Raises exception on failure.
    """
    cp = cache_path(url)
    cp = cache_path(url)
    if os.path.exists(cp) and time.time() - os.path.getmtime(cp) < 7 * 24 * 3600:
        return open(cp, "r", encoding="utf-8", errors="ignore").read()
    
    
    headers = {"User-Agent": "CollegeAppHelperBot/0.1 (research; contact: admin@example.com)"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{2,}", "\n", text).strip()
        
        with open(cp, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    except Exception as e:
        # Log the error but don't raise it - let caller handle fallback
        print(f"Failed to fetch {url}: {e}")
        raise

def must_keywords_from_program(program_text: str) -> List[str]:
    """Extract relevant keywords from the program text for use in generation."""
    seeds = [
        "Machine Learning", "Statistical Inference", "Data Management",
        "Data Visualization", "Data Science", "Analytics", "Statistics",
        "Python", "R", "SQL", "Deep Learning", "AI", "Artificial Intelligence",
        "Big Data", "Database", "Algorithms", "Research", "Ethics"
    ]
    found = [s for s in seeds if s.lower() in program_text.lower()]
    return found if found else seeds[:8]  # Return top 8 if none found

def validate_profile(profile: Profile) -> List[str]:
    """Validate the profile data and return a list of warnings if any issues are found."""
    warnings = []
    
    if not profile.experiences:
        warnings.append("No work experiences provided - this may limit content quality")
    
    if not profile.skills:
        warnings.append("No skills listed - consider adding relevant technical skills")
    
    if profile.gpa and (profile.gpa < 0 or profile.gpa > 4.0):
        warnings.append("GPA appears to be outside normal range (0-4.0)")
    
    if not profile.goals or len(profile.goals) < 20:
        warnings.append("Goals statement is very brief - more detail could improve results")
    
    return warnings

@app.post("/generate")
def generate(req: GenerateRequest):
    """
    Main generation endpoint.
    Handles both multi-agent and simple generator systems, with fallback logic.
    Validates input, fetches program text, retrieves evidence, generates content, and returns results.
    """
    try:
        # Validate input
        profile_warnings = validate_profile(req.profile)
        
        # 1) Get program text
        program_text = req.program_text or ""
        fetch_error = None
        
        if (not program_text) and req.program_url:
            try:
                program_text = polite_fetch(req.program_url)
            except Exception as e:
                fetch_error = str(e)
                program_text = ""
        
        if not program_text:
            error_msg = "No program text provided."
            if req.program_url:
                error_msg += f" URL fetch failed: {fetch_error}"
            error_msg += " Please provide program_text or a valid program_url."
            raise HTTPException(status_code=400, detail=error_msg)
        
        # 2) Ingest corpus and retrieve evidence
        try:
            corpus = ingest_corpus(CORPUS_DIR)
            prof_kws = (req.profile.courses or []) + (req.profile.skills or [])
            q = build_query(program_text, prof_kws)
            top = retrieve_topk(corpus, q, k=req.topk)
            evidence_ids = [tid for tid, _ in top]
        except Exception as e:
            # Fallback: use program text as evidence
            print(f"Corpus retrieval failed: {e}")
            corpus = {"program_text": program_text}
            evidence_ids = ["program_text"]
        
        # 3) Extract keywords
        must_kws = must_keywords_from_program(program_text)
        
        # 4) Generate content using selected system
        generation_error = None
        texts = None
        report = None
        system_used = "unknown"
        
        if req.use_multi_agent:
            try:
                # Try multi-agent system first
                texts, report = generate_all_multi_agent(
                    corpus, evidence_ids, req.profile.dict(),
                    req.resume_text, program_text, must_kws,
                    req.max_iterations, req.critique_threshold
                )
                system_used = "multi_agent"
            except Exception as e:
                generation_error = f"Multi-agent system failed: {str(e)}"
                print(generation_error)
                
                if req.fallback_on_error:
                    # Fallback to simple generator
                    try:
                        texts, report = generate_all(
                            corpus, evidence_ids, req.profile.dict(),
                            req.resume_text, program_text, must_kws
                        )
                        system_used = "simple_fallback"
                        report["fallback_reason"] = generation_error
                    except Exception as fallback_error:
                        raise HTTPException(
                            status_code=500, 
                            detail=f"Both systems failed. Multi-agent: {generation_error}. Simple: {str(fallback_error)}"
                        )
                else:
                    raise HTTPException(status_code=500, detail=generation_error)
        else:
            # Use simple generator by request
            try:
                texts, report = generate_all(
                    corpus, evidence_ids, req.profile.dict(),
                    req.resume_text, program_text, must_kws
                )
                system_used = "simple_requested"
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Simple generator failed: {str(e)}")
        
        # 5) Enhance report with metadata
        enhanced_report = {
            **report,
            "system_used": system_used,
            "generation_metadata": {
                "program_url": req.program_url,
                "program_text_length": len(program_text),
                "evidence_count": len(evidence_ids),
                "keywords_extracted": len(must_kws),
                "profile_warnings": profile_warnings,
                "fetch_error": fetch_error,
                "generation_error": generation_error
            }
        }
        
        # 6) Persist results with enhanced naming
        timestamp = int(time.time())
        file_suffix = f"_{system_used}_{timestamp}"
        
        try:
            with open(os.path.join(OUT_DIR, f"personal_statement{file_suffix}.md"), "w", encoding="utf-8") as f:
                f.write(texts["personal_statement"])
            with open(os.path.join(OUT_DIR, f"resume_bullets{file_suffix}.md"), "w", encoding="utf-8") as f:
                f.write(texts["resume_bullets"])
            with open(os.path.join(OUT_DIR, f"reco_template{file_suffix}.md"), "w", encoding="utf-8") as f:
                f.write(texts["reco_template"])
            with open(os.path.join(OUT_DIR, f"report{file_suffix}.json"), "w", encoding="utf-8") as f:
                json.dump(enhanced_report, f, indent=2)
        except Exception as e:
            print(f"Failed to persist files: {e}")
            # Don't fail the request for file write errors
        
        return {
            "texts": texts, 
            "report": enhanced_report, 
            "evidence_ids": evidence_ids,
            "system_used": system_used,
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint to verify API status and available systems."""
    return {"status": "healthy", "systems": ["simple_generator", "multi_agent_generator"]}

@app.get("/systems/info")
def systems_info():
    """Endpoint to provide information about available generation systems and recommendations."""
    return {
        "available_systems": {
            "simple": {
                "name": "Simple Generator",
                "description": "Fast, template-based content generation",
                "best_for": "Quick results, basic requirements",
                "average_time": "< 1 second"
            },
            "multi_agent": {
                "name": "Multi-Agent Generator", 
                "description": "Iterative improvement using writer and critic agents",
                "best_for": "High-quality results, detailed feedback",
                "average_time": "3-10 seconds",
                "parameters": {
                    "max_iterations": "Maximum refinement cycles (1-5)",
                    "critique_threshold": "Quality threshold for approval (0.0-1.0)"
                }
            }
        },
        "recommendations": {
            "use_multi_agent_for": [
                "Important applications (graduate school, jobs)",
                "When you have detailed profile information", 
                "When quality is more important than speed"
            ],
            "use_simple_for": [
                "Quick drafts and iterations",
                "When you have limited profile information",
                "When speed is critical"
            ]
        }
    }

@app.post("/generate/simple")
def generate_simple(req: GenerateRequest):
    """Endpoint to force use of the simple generator system for content generation."""
    req.use_multi_agent = False
    return generate(req)

@app.post("/generate/multi-agent") 
def generate_multi_agent(req: GenerateRequest):
    """Endpoint to force use of the multi-agent generator system for content generation."""
    req.use_multi_agent = True
    req.fallback_on_error = False  # Don't fallback for this explicit endpoint
    return generate(req)

@app.get("/")
def root():
    """Root endpoint providing API information, available endpoints, and features."""
    return {
        "message": "College Application Helper API V4.0 - Dual Dataset Support",
        "version": "4.0",
        "endpoints": {
            "/generate": "Main generation endpoint with system selection",
            "/generate/simple": "Force simple generator",
            "/generate/multi-agent": "Force multi-agent generator",
            "/generate/writing-agent": "LangGraph-based Writing Agent",
            "/systems/info": "Information about available systems",
            "/health": "Health check",
            # V1 (Legacy) matching endpoints
            "/match/programs": "Match programs using V1 (legacy) corpus",
            "/match/info": "V1 matching service info",
            "/match/program/{id}/details": "Get V1 program details",
            "/match/programs/list": "List all V1 programs",
            # V2 (New dataset) matching endpoints
            "/v2/match/programs": "NEW: Match programs using V2 enhanced dataset",
            "/v2/match/info": "V2 matching service info",
            "/v2/match/program/{id}/details": "Get V2 program details",
            "/v2/match/programs/list": "List all V2 programs",
        },
        "features": [
            "Dual dataset support (V1 legacy + V2 enhanced)",
            "V2 dataset with richer program information and course descriptions",
            "Multi-agent content generation with iterative improvement",
            "LangGraph-based Writing Agent with RAG, ReAct, Reflection",
            "6-dimension matching for V2 (academic, skills, experience, goals, requirements, curriculum)",
            "Course-level curriculum analysis in V2",
            "Fallback to simple generator for reliability",
            "Detailed quality reports and feedback"
        ],
        "datasets": {
            "v1_legacy": {
                "available": MATCHER_AVAILABLE,
                "programs_count": len(PROGRAM_MATCHER.programs) if MATCHER_AVAILABLE else 0,
                "description": "Original corpus with sections-based structure"
            },
            "v2_enhanced": {
                "available": MATCHER_V2_AVAILABLE,
                "programs_count": len(PROGRAM_MATCHER_V2.programs) if MATCHER_V2_AVAILABLE else 0,
                "description": "New dataset with extracted_fields, courses with descriptions, etc."
            }
        },
        "writing_agent_available": WRITING_AGENT_AVAILABLE,
        "matching_service_available": MATCHER_AVAILABLE,
        "matching_service_v2_available": MATCHER_V2_AVAILABLE
    }


@app.post("/generate/writing-agent")
def generate_with_writing_agent(req: WritingAgentRequest):
    """
    Generate documents using LangGraph-based Writing Agent.
    
    This endpoint uses advanced AI workflows including:
    - RAG (Retrieval-Augmented Generation)
    - ReAct (Reasoning + Acting with tools)
    - Reflection (Self-evaluation)
    - Reflexion (Memory-enhanced learning)
    - ReWOO (Plan-Tool-Solve workflow)
    """
    
    if not WRITING_AGENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Writing Agent is not available. Please install required dependencies: langchain, langgraph, langchain-openai"
        )
    
    try:
        # 1) Get program text
        program_text = req.program_text or ""
        fetch_error = None
        
        if (not program_text) and req.program_url:
            try:
                program_text = polite_fetch(req.program_url)
            except Exception as e:
                fetch_error = str(e)
                program_text = ""
        
        if not program_text:
            error_msg = "No program text provided."
            if req.program_url:
                error_msg += f" URL fetch failed: {fetch_error}"
            error_msg += " Please provide program_text or a valid program_url."
            raise HTTPException(status_code=400, detail=error_msg)
        
        # 2) Prepare program_info dict
        program_info = {
            "program_name": "Target Program",  # Extract from program_text if needed
            "features": program_text[:2000],  # Use first part as features
            "application_requirements": "",
            "courses": []
        }
        
        # Try to ingest corpus for RAG
        corpus = None
        if req.use_corpus:
            try:
                full_corpus = ingest_corpus(CORPUS_DIR)
                # Simple filtering: use chunks relevant to program_text
                corpus = {}
                for chunk_id, chunk_text in list(full_corpus.items())[:50]:  # Limit corpus size
                    corpus[chunk_id] = chunk_text
            except Exception as e:
                print(f"Corpus ingestion failed: {e}")
                corpus = {"program_text": program_text}
        
        # 3) Map document type
        doc_type_map = {
            "personal_statement": DocumentType.PERSONAL_STATEMENT,
            "resume_bullets": DocumentType.RESUME_BULLETS,
            "recommendation_letter": DocumentType.RECOMMENDATION_LETTER
        }
        
        document_type = doc_type_map.get(req.document_type, DocumentType.PERSONAL_STATEMENT)
        
        # 4) Generate using Writing Agent
        start_time = time.time()
        
        result = generate_document(
            profile=req.profile.dict(),
            program_info=program_info,
            document_type=document_type,
            corpus=corpus,
            llm_provider=req.llm_provider,
            model_name=req.model_name,
            temperature=req.temperature,
            max_iterations=req.max_iterations,
            quality_threshold=req.quality_threshold
        )
        
        generation_time = time.time() - start_time
        
        # 5) Format response
        final_document = result.get("final_document", "")
        quality_report = result.get("quality_report", {})
        metadata = result.get("metadata", {})
        
        # 6) Save output
        timestamp = int(time.time())
        doc_type_str = req.document_type
        
        output_file = os.path.join(OUT_DIR, f"{doc_type_str}_writing_agent_{timestamp}.md")
        report_file = os.path.join(OUT_DIR, f"{doc_type_str}_report_{timestamp}.json")
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(final_document)
            
            full_report = {
                "quality_report": quality_report,
                "metadata": metadata,
                "generation_time_seconds": generation_time,
                "document_type": req.document_type,
                "llm_provider": req.llm_provider,
                "model_name": req.model_name,
                "iterations": result.get("iterations", 0)
            }
            
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(full_report, f, indent=2)
        except Exception as e:
            print(f"Failed to save output files: {e}")
        
        return {
            "success": True,
            "document": final_document,
            "document_type": req.document_type,
            "quality_report": quality_report,
            "metadata": metadata,
            "generation_time_seconds": round(generation_time, 2),
            "iterations": result.get("iterations", 0),
            "draft_history_length": len(result.get("draft_history", [])),
            "output_files": {
                "document": output_file,
                "report": report_file
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        raise HTTPException(status_code=500, detail=error_detail)

# ============================================================
# MATCHING SERVICE ENDPOINTS 
# ============================================================

@app.post("/match/programs", response_model=MatchingResponse)
def match_programs(request: MatchingRequest):
    """
    Match student profile with suitable programs
    
    Analyzes student background across 5 dimensions:
    - Academic (GPA, major, coursework)
    - Skills (technical skills match)
    - Experience (work/project relevance)
    - Goals (career alignment)
    - Requirements (hard requirements compliance)
    
    Returns ranked programs with detailed scores and recommendations.
    """
    
    if not MATCHER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Program matching service is not available. Please ensure corpus is loaded."
        )
    
    try:
        start_time = time.time()
        
        # Validate profile
        profile = request.profile
        if not profile.get("major") or not profile.get("gpa"):
            raise HTTPException(
                status_code=400,
                detail="Profile must include at least 'major' and 'gpa' fields"
            )
        
        # Perform matching
        result = PROGRAM_MATCHER.match_programs(
            profile=profile,
            top_k=request.top_k,
            min_score=request.min_score,
            custom_weights=request.custom_weights,
            filters=request.filters
        )
        
        # Optionally enhance explanations with LLM
        if request.use_llm_explanation:
            try:
                explainer = MatchExplainer(
                    llm_provider=os.getenv("WRITING_AGENT_LLM_PROVIDER", "openai"),
                    use_llm=True
                )
                
                for match in result.matches:
                    enhanced_explanation = explainer.generate_detailed_explanation(
                        profile=profile,
                        program={
                            "name": match.program_name,
                            "university": match.university,
                            "focus_areas": [],
                        },
                        overall_score=match.overall_score,
                        dimension_scores=match.dimension_scores,
                        strengths=match.strengths,
                        gaps=match.gaps
                    )
                    match.explanation = enhanced_explanation
            except Exception as e:
                print(f"Warning: LLM explanation enhancement failed: {e}")
        
        processing_time = time.time() - start_time
        
        # Convert to response format with complete program details
        matches_response = []
        for m in result.matches:
            # Get full program details for Writing Agent
            program_details = None
            if m.program_id in PROGRAM_MATCHER.programs:
                prog = PROGRAM_MATCHER.programs[m.program_id]
                program_details = {
                    "program_id": m.program_id,
                    "program_name": prog.get("name", m.program_name),
                    "university": prog.get("university", m.university),
                    "field": prog.get("field", ""),
                    "description_text": prog.get("description_text", ""),
                    "features": prog.get("career_outcomes", ""),
                    "core_courses": prog.get("core_courses", []),
                    "focus_areas": prog.get("focus_areas", []),
                    "required_skills": prog.get("required_skills", []),
                    "min_gpa": prog.get("min_gpa", 3.0),
                    "duration": prog.get("duration", ""),
                    "source_url": prog.get("source_url", "")
                }
            
            match_response = ProgramMatchResponse(
                program_id=m.program_id,
                program_name=m.program_name,
                university=m.university,
                overall_score=round(m.overall_score, 3),
                match_level=m.match_level.value,
                dimension_scores={
                    dim: DimensionScoreResponse(
                        dimension=score.dimension,
                        score=round(score.score, 3),
                        weight=score.weight,
                        details=score.details,
                        contributing_factors=score.contributing_factors
                    )
                    for dim, score in m.dimension_scores.items()
                },
                strengths=m.strengths,
                gaps=m.gaps,
                fit_reasons=getattr(m, 'fit_reasons', []),  # Personalized fit reasons
                recommendations=m.recommendations,
                explanation=m.explanation,
                metadata=m.metadata,
                program_details=program_details
            )
            matches_response.append(match_response)
        
        response = MatchingResponse(
            success=True,
            message=f"Found {len(result.matches)} matching programs",
            student_profile_summary=result.student_profile_summary,
            total_programs_evaluated=result.total_programs_evaluated,
            matches=matches_response,
            overall_insights=result.overall_insights,
            matching_timestamp=result.matching_timestamp,
            processing_time_seconds=round(processing_time, 2)
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error in program matching: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Program matching failed: {str(e)}"
        )


@app.get("/match/info")
def matching_info():
    """
    Get information about the matching service
    
    Returns:
    - Number of programs available
    - Matching dimensions and weights
    - Example usage
    """
    
    if not MATCHER_AVAILABLE:
        return {
            "available": False,
            "message": "Matching service not initialized"
        }
    
    return {
        "available": True,
        "programs_loaded": len(PROGRAM_MATCHER.programs),
        "matching_dimensions": {
            "academic": {
                "weight": 0.30,
                "factors": ["GPA", "Major relevance", "Coursework"]
            },
            "skills": {
                "weight": 0.25,
                "factors": ["Required skills coverage", "Skill breadth"]
            },
            "experience": {
                "weight": 0.20,
                "factors": ["Experience quantity", "Relevance", "Impact"]
            },
            "goals": {
                "weight": 0.15,
                "factors": ["Goal alignment", "Clarity"]
            },
            "requirements": {
                "weight": 0.10,
                "factors": ["GPA requirement", "Language tests", "Prerequisites"]
            }
        },
        "default_top_k": 5,
        "default_min_score": 0.5,
        "supports_custom_weights": True,
        "supports_llm_explanations": True,
        "example_request": {
            "profile": {
                "name": "Jane Doe",
                "major": "Data Science",
                "gpa": 3.7,
                "skills": ["Python", "Machine Learning", "SQL"],
                "experiences": [
                    {
                        "title": "Data Analyst",
                        "org": "Tech Corp",
                        "impact": "Improved model accuracy by 15%",
                        "skills": ["Python", "TensorFlow"]
                    }
                ],
                "goals": "Apply ML to healthcare analytics"
            },
            "top_k": 5,
            "min_score": 0.6,
            "use_llm_explanation": True
        }
    }


@app.post("/match/compare")
def compare_programs(profile: dict, program_ids: List[str]):
    """
    Compare specific programs for a student profile
    
    Args:
    - profile: Student profile dictionary
    - program_ids: List of program IDs to compare
    
    Returns:
    - Side-by-side comparison of programs
    """
    
    if not MATCHER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Matching service not available"
        )
    
    try:
        comparisons = []
        
        for program_id in program_ids:
            if program_id not in PROGRAM_MATCHER.programs:
                continue
            
            program = PROGRAM_MATCHER.programs[program_id]
            
            # Score this specific program
            dimension_scores = {
                "academic": PROGRAM_MATCHER.scorer.score_academic(profile, program),
                "skills": PROGRAM_MATCHER.scorer.score_skills(profile, program),
                "experience": PROGRAM_MATCHER.scorer.score_experience(profile, program),
                "goals": PROGRAM_MATCHER.scorer.score_goals(profile, program),
                "requirements": PROGRAM_MATCHER.scorer.score_requirements(profile, program)
            }
            
            overall_score = PROGRAM_MATCHER.scorer.compute_overall_score(dimension_scores)
            
            comparisons.append({
                "program_id": program_id,
                "program_name": program["name"],
                "university": program["university"],
                "overall_score": round(overall_score, 3),
                "dimension_scores": {
                    dim: {
                        "score": round(score.score, 3),
                        "factors": score.contributing_factors[:2]
                    }
                    for dim, score in dimension_scores.items()
                }
            })
        
        # Sort by score
        comparisons.sort(key=lambda x: x["overall_score"], reverse=True)
        
        return {
            "success": True,
            "comparisons": comparisons,
            "best_match": comparisons[0] if comparisons else None
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )

# ============================================================
# NEW: GET PROGRAM DETAILS ENDPOINT
# ============================================================

@app.get("/match/program/{program_id}/details")
def get_program_details(program_id: str):
    """
    Get complete program details for a specific program.
    
    This endpoint returns all information needed for the Writing Agent,
    including full description, courses, focus areas, etc.
    
    Args:
        program_id: The unique identifier of the program
    
    Returns:
        Complete program information dictionary
    """
    
    if not MATCHER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Matching service not available"
        )
    
    if program_id not in PROGRAM_MATCHER.programs:
        raise HTTPException(
            status_code=404,
            detail=f"Program '{program_id}' not found in corpus"
        )
    
    try:
        program = PROGRAM_MATCHER.programs[program_id]
        
        # Build comprehensive program info for Writing Agent
        program_details = {
            "program_id": program_id,
            "program_name": program.get("name", "Unknown Program"),
            "university": program.get("university", "Unknown University"),
            "field": program.get("field", ""),
            "degree_type": program.get("degree_type", "Master"),
            
            # Full description for Writing Agent
            "description_text": program.get("description_text", ""),
            "features": program.get("career_outcomes", ""),
            
            # Academic details
            "core_courses": program.get("core_courses", []),
            "prerequisite_courses": program.get("prerequisite_courses", []),
            "focus_areas": program.get("focus_areas", []),
            
            # Requirements
            "min_gpa": program.get("min_gpa", 3.0),
            "required_skills": program.get("required_skills", []),
            "language_requirements": program.get("language_requirements", {}),
            
            # Program info
            "duration": program.get("duration", ""),
            "tuition": program.get("tuition"),
            "source_url": program.get("source_url", ""),
            
            # Career outcomes
            "career_outcomes": program.get("career_outcomes", "")
        }
        
        return {
            "success": True,
            "program": program_details
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get program details: {str(e)}"
        )


@app.get("/match/programs/list")
def list_all_programs():
    """
    List all available programs in the V1 (legacy) corpus.
    
    Returns:
        List of programs with basic info (id, name, university)
    """
    
    if not MATCHER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="V1 Matching service not available"
        )
    
    try:
        programs_list = []
        
        for program_id, program in PROGRAM_MATCHER.programs.items():
            programs_list.append({
                "program_id": program_id,
                "program_name": program.get("name", "Unknown"),
                "university": program.get("university", "Unknown"),
                "field": program.get("field", ""),
                "focus_areas": program.get("focus_areas", [])[:3]
            })
        
        return {
            "success": True,
            "dataset_version": "v1",
            "total_programs": len(programs_list),
            "programs": programs_list
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list programs: {str(e)}"
        )


# ============================================================
# V2 MATCHING SERVICE ENDPOINTS (New Enhanced Dataset)
# ============================================================

class MatchingRequestV2API(BaseModel):
    """Matching request model for V2 API"""
    profile: Dict[str, Any] = Field(..., description="Student profile information")
    top_k: int = Field(10, ge=1, le=50, description="Return top K matching programs")
    min_score: float = Field(0.4, ge=0.0, le=1.0, description="Minimum match score threshold")
    include_curriculum_analysis: bool = Field(True, description="Include curriculum alignment analysis")
    include_course_recommendations: bool = Field(True, description="Include course recommendations")
    custom_weights: Optional[Dict[str, float]] = Field(None, description="Custom dimension weights")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters (e.g., university, field)")
    use_llm_explanation: bool = Field(False, description="Use LLM for detailed explanations")


@app.post("/v2/match/programs")
def match_programs_v2(request: MatchingRequestV2API):
    """
    Match student profile with programs using V2 (enhanced) dataset.
    
    V2 Features:
    - 6-dimension analysis (including curriculum alignment)
    - Rich course information with descriptions
    - Detailed application requirements parsing
    - Training outcomes and career path analysis
    
    Returns ranked programs with detailed scores and recommendations.
    """
    
    if not MATCHER_V2_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="V2 Program matching service is not available. Please ensure V2 corpus is loaded."
        )
    
    try:
        start_time = time.time()
        
        # Validate profile
        profile = request.profile
        if not profile.get("major") or not profile.get("gpa"):
            raise HTTPException(
                status_code=400,
                detail="Profile must include at least 'major' and 'gpa' fields"
            )
        
        # Perform matching
        result = PROGRAM_MATCHER_V2.match_programs(
            profile=profile,
            top_k=request.top_k,
            min_score=request.min_score,
            custom_weights=request.custom_weights,
            filters=request.filters,
            include_curriculum_analysis=request.include_curriculum_analysis
        )
        
        processing_time = time.time() - start_time
        
        # Convert to response format
        matches_response = []
        for m in result.matches:
            # Get program details for Writing Agent
            program_details = PROGRAM_MATCHER_V2.get_program_details_for_writing(m.program_id)
            
            match_response = {
                "program_id": m.program_id,
                "program_name": m.program_name,
                "university": m.university,
                "department": m.program_data.department if m.program_data else None,
                "overall_score": round(m.overall_score, 3),
                "match_level": m.match_level.value,
                "dimension_scores": {
                    dim: {
                        "dimension": score.dimension,
                        "score": round(score.score, 3),
                        "weight": score.weight,
                        "details": score.details,
                        "contributing_factors": score.contributing_factors,
                        "matched_items": score.matched_items,
                        "missing_items": score.missing_items
                    }
                    for dim, score in m.dimension_scores.items()
                },
                "strengths": m.strengths,
                "gaps": m.gaps,
                "fit_reasons": m.fit_reasons,
                "recommendations": m.recommendations,
                "matched_courses": m.matched_courses,
                "relevant_courses": m.relevant_courses,
                "explanation": m.explanation,
                "metadata": m.metadata,
                "program_details": program_details
            }
            matches_response.append(match_response)
        
        return {
            "success": True,
            "message": f"Found {len(result.matches)} matching programs from V2 dataset",
            "dataset_version": "v2",
            "student_profile_summary": result.student_profile_summary,
            "total_programs_evaluated": result.total_programs_evaluated,
            "matches": matches_response,
            "overall_insights": result.overall_insights,
            "matching_timestamp": result.matching_timestamp,
            "processing_time_seconds": round(processing_time, 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error in V2 program matching: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"V2 Program matching failed: {str(e)}"
        )


@app.get("/v2/match/info")
def matching_info_v2():
    """
    Get information about the V2 matching service.
    
    Returns:
    - Number of programs available
    - Matching dimensions and weights
    - V2-specific features
    """
    
    if not MATCHER_V2_AVAILABLE:
        return {
            "available": False,
            "dataset_version": "v2",
            "message": "V2 Matching service not initialized"
        }
    
    return {
        "available": True,
        "dataset_version": "v2",
        "programs_loaded": len(PROGRAM_MATCHER_V2.programs),
        "matching_dimensions": {
            "academic": {
                "weight": 0.25,
                "factors": ["GPA", "Major relevance", "Coursework alignment"]
            },
            "skills": {
                "weight": 0.20,
                "factors": ["Required skills coverage", "Course-skill alignment", "Skill breadth"]
            },
            "experience": {
                "weight": 0.15,
                "factors": ["Experience quantity", "Relevance", "Research/Professional orientation"]
            },
            "goals": {
                "weight": 0.20,
                "factors": ["Career goals alignment", "Mission alignment", "Career path match"]
            },
            "requirements": {
                "weight": 0.10,
                "factors": ["Prerequisites", "Test scores", "Documents"]
            },
            "curriculum": {
                "weight": 0.10,
                "factors": ["Course interest alignment", "Relevant courses", "Curriculum diversity"],
                "note": "V2-specific: Analyzes course descriptions and content"
            }
        },
        "v2_features": [
            "Rich course information with descriptions",
            "Nested extracted_fields structure",
            "Application requirements with detailed fields",
            "Training outcomes with career paths",
            "Curriculum alignment scoring",
            "Course-level matching"
        ],
        "default_top_k": 10,
        "default_min_score": 0.4
    }


@app.get("/v2/match/program/{program_id}/details")
def get_program_details_v2(program_id: str):
    """
    Get complete program details from V2 dataset.
    
    Returns all information needed for the Writing Agent,
    including courses with descriptions, application requirements,
    training outcomes, and text chunks for RAG.
    """
    
    if not MATCHER_V2_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="V2 Matching service not available"
        )
    
    program = PROGRAM_MATCHER_V2.get_program(program_id)
    if not program:
        raise HTTPException(
            status_code=404,
            detail=f"Program '{program_id}' not found in V2 corpus"
        )
    
    try:
        program_details = PROGRAM_MATCHER_V2.get_program_details_for_writing(program_id)
        
        return {
            "success": True,
            "dataset_version": "v2",
            "program": program_details
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get V2 program details: {str(e)}"
        )


@app.get("/v2/match/programs/list")
def list_all_programs_v2():
    """
    List all available programs in the V2 (enhanced) corpus.
    
    Returns:
        List of programs with basic info including course availability
    """
    
    if not MATCHER_V2_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="V2 Matching service not available"
        )
    
    try:
        programs_list = PROGRAM_MATCHER_V2.list_programs()
        
        return {
            "success": True,
            "dataset_version": "v2",
            "total_programs": len(programs_list),
            "programs": programs_list
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list V2 programs: {str(e)}"
        )


@app.post("/v2/generate/writing-agent")
def generate_with_writing_agent_v2(req: WritingAgentRequest, program_id: Optional[str] = None):
    """
    Generate documents using Writing Agent with V2 dataset program details.
    
    If program_id is provided, automatically fetches rich program information
    from the V2 dataset including course descriptions, requirements, etc.
    """
    
    if not WRITING_AGENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Writing Agent is not available"
        )
    
    try:
        # If program_id provided, get V2 program details
        program_text = req.program_text or ""
        program_info = {}
        
        if program_id and MATCHER_V2_AVAILABLE:
            program_details = PROGRAM_MATCHER_V2.get_program_details_for_writing(program_id)
            if program_details:
                # Build rich program text from V2 data
                parts = []
                parts.append(f"Program: {program_details.get('program_name', 'Graduate Program')}")
                parts.append(f"University: {program_details.get('university', 'University')}")
                
                if program_details.get('program_background', {}).get('mission'):
                    parts.append(f"Mission: {program_details['program_background']['mission']}")
                
                if program_details.get('description_text'):
                    parts.append(program_details['description_text'][:2000])
                
                if program_details.get('courses'):
                    course_info = []
                    for c in program_details['courses'][:10]:
                        if isinstance(c, dict):
                            course_str = c.get('name', '')
                            if c.get('description'):
                                course_str += f": {c['description'][:100]}"
                            course_info.append(course_str)
                    parts.append("Courses:\n" + "\n".join(course_info))
                
                if program_details.get('training_outcomes', {}).get('career_paths'):
                    parts.append(f"Career Paths: {program_details['training_outcomes']['career_paths']}")
                
                program_text = "\n\n".join(parts)
                
                # Set program_info for writing agent
                program_info = {
                    "program_name": program_details.get('program_name', ''),
                    "features": program_text[:2000],
                    "application_requirements": program_details.get('application_requirements', {}).get('summary', ''),
                    "courses": [c.get('name', '') if isinstance(c, dict) else c for c in program_details.get('courses', [])]
                }
        
        # Fallback to URL fetch if no program text
        if not program_text and req.program_url:
            try:
                program_text = polite_fetch(req.program_url)
            except Exception as e:
                pass
        
        if not program_text:
            raise HTTPException(
                status_code=400,
                detail="No program information provided. Provide program_id, program_text, or program_url."
            )
        
        # Use default program_info if not set
        if not program_info:
            program_info = {
                "program_name": "Target Program",
                "features": program_text[:2000],
                "application_requirements": "",
                "courses": []
            }
        
        # Get corpus for RAG
        corpus = None
        if req.use_corpus and program_id and MATCHER_V2_AVAILABLE:
            program = PROGRAM_MATCHER_V2.get_program(program_id)
            if program and program.chunks:
                corpus = {c.chunk_id: c.text for c in program.chunks}
        
        if not corpus:
            corpus = {"program_text": program_text}
        
        # Map document type
        doc_type_map = {
            "personal_statement": DocumentType.PERSONAL_STATEMENT,
            "resume_bullets": DocumentType.RESUME_BULLETS,
            "recommendation_letter": DocumentType.RECOMMENDATION_LETTER
        }
        document_type = doc_type_map.get(req.document_type, DocumentType.PERSONAL_STATEMENT)
        
        # Generate document
        start_time = time.time()
        
        result = generate_document(
            profile=req.profile.dict(),
            program_info=program_info,
            document_type=document_type,
            corpus=corpus,
            llm_provider=req.llm_provider,
            model_name=req.model_name,
            temperature=req.temperature,
            max_iterations=req.max_iterations,
            quality_threshold=req.quality_threshold
        )
        
        generation_time = time.time() - start_time
        
        return {
            "success": True,
            "document": result.get("final_document", ""),
            "document_type": req.document_type,
            "quality_report": result.get("quality_report", {}),
            "metadata": result.get("metadata", {}),
            "generation_time_seconds": round(generation_time, 2),
            "iterations": result.get("iterations", 0),
            "dataset_version": "v2" if program_id else "custom",
            "program_id": program_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "traceback": traceback.format_exc()}
        )


# ============================================================
# END OF V2 MATCHING SERVICE ENDPOINTS
# ============================================================

if __name__ == "__main__":
    # Entry point for running the FastAPI app with uvicorn.
    uvicorn.run(app, host="0.0.0.0", port=8000)