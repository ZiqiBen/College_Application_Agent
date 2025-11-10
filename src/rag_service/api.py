from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn, os, json, re, time, hashlib
import requests
from bs4 import BeautifulSoup

# Import both generator systems
from .generator import generate_all  # Original simple generator (fallback)
from .multi_agent_generator import generate_all_multi_agent  # New multi-agent system
# Import both generator systems
from .generator import generate_all  # Original simple generator (fallback)
from .multi_agent_generator import generate_all_multi_agent  # New multi-agent system
from .ingest import ingest_corpus
from .retriever_bert import build_query, retrieve_topk
from .retriever_bert import build_query, retrieve_topk

app = FastAPI(title="College App Helper API - Enhanced Multi-Agent System")
app = FastAPI(title="College App Helper API - Enhanced Multi-Agent System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],

)

CORPUS_DIR = "data/corpus"
OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)

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
    # Multi-agent system parameters
    use_multi_agent: bool = True  # Default to using multi-agent system
    max_iterations: int = 3
    critique_threshold: float = 0.8
    fallback_on_error: bool = True  # Fallback to simple generator if multi-agent fails

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
        "message": "College Application Helper API - Enhanced Multi-Agent System",
        "version": "2.0",
        "endpoints": {
            "/generate": "Main generation endpoint with system selection",
            "/generate/simple": "Force simple generator",
            "/generate/multi-agent": "Force multi-agent generator", 
            "/systems/info": "Information about available systems",
            "/health": "Health check"
        },
        "features": [
            "Multi-agent content generation with iterative improvement",
            "Fallback to simple generator for reliability",
            "Detailed quality reports and feedback",
            "Automatic keyword extraction and optimization",
            "Profile validation and warnings"
        ]
    }

if __name__ == "__main__":
    # Entry point for running the FastAPI app with uvicorn.
    uvicorn.run(app, host="0.0.0.0", port=8000)