"""
College Application Helper - Full-Featured Streamlit Frontend (v3.0)

This comprehensive Streamlit app provides a complete user interface for the 
College Application Helper system with three main workflows:

1. **Program Matching**: Match student profile with suitable graduate programs
2. **Document Generation**: Generate application documents using AI Writing Agent
3. **End-to-End Flow**: Full pipeline from matching to document generation

Key Features:
- Integration with Matching Service for intelligent program recommendations
- LangGraph-based Writing Agent for high-quality document generation
- Support for multiple LLM providers (OpenAI, Anthropic, Qwen)
- Iterative refinement with quality scoring
- RAG-enhanced generation with corpus retrieval
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, List, Optional

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="College Application Helper - Full Suite",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONSTANTS AND API CONFIGURATION
# =============================================================================

API_BASE_URL = "http://localhost:8000"

# Document type options
DOCUMENT_TYPES = {
    "personal_statement": "📝 Personal Statement",
    "resume_bullets": "📋 Resume Bullets", 
    "recommendation_letter": "📨 Recommendation Letter"
}

# LLM Provider options
LLM_PROVIDERS = {
    "openai": "OpenAI (GPT-4)",
    "anthropic": "Anthropic (Claude)",
    "qwen": "Qwen (通义千问)"
}

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables"""
    if "profile" not in st.session_state:
        st.session_state.profile = None
    if "matched_programs" not in st.session_state:
        st.session_state.matched_programs = []
    if "selected_program" not in st.session_state:
        st.session_state.selected_program = None
    if "generated_documents" not in st.session_state:
        st.session_state.generated_documents = {}
    if "current_step" not in st.session_state:
        st.session_state.current_step = 1

init_session_state()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_api_health() -> Dict:
    """Check API health and available services"""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            return {"status": "healthy", "data": resp.json()}
        return {"status": "error", "message": f"Status code: {resp.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Cannot connect to API server"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_api_info() -> Dict:
    """Get API information including available services"""
    try:
        resp = requests.get(f"{API_BASE_URL}/", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return {}
    except:
        return {}

def build_profile_dict(name, major, goals, email, gpa, courses, skills, experiences) -> Dict:
    """Build profile dictionary from form inputs"""
    return {
        "name": name,
        "major": major,
        "goals": goals,
        "email": email or None,
        "gpa": float(gpa) if gpa else None,
        "courses": [c.strip() for c in courses.split(",") if c.strip()],
        "skills": [s.strip() for s in skills.split(",") if s.strip()],
        "experiences": experiences
    }

# =============================================================================
# UI COMPONENTS - PROFILE INPUT
# =============================================================================

def render_profile_form() -> Dict:
    """Render the profile input form and return profile data"""
    st.subheader("👤 Your Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Name", value="Alice Zhang", key="profile_name")
        major = st.text_input("Major/Field", value="Data Science", key="profile_major")
        goals = st.text_area(
            "Career Goals",
            value="Apply ML to real-world analytics and product data science. Lead data-driven initiatives in tech companies.",
            height=100,
            help="Be specific about your career aspirations",
            key="profile_goals"
        )
        
        with st.expander("📧 Contact Information"):
            email = st.text_input("Email", value="alice@example.com", key="profile_email")
            gpa = st.text_input("GPA (0-4.0)", value="3.78", key="profile_gpa")
    
    with col2:
        courses = st.text_area(
            "Relevant Courses",
            value="Machine Learning, Deep Learning, Statistical Inference, Data Management, Algorithms",
            help="Comma-separated list",
            key="profile_courses"
        )
        skills = st.text_area(
            "Technical Skills",
            value="Python, SQL, PyTorch, TensorFlow, R, Tableau, AWS, Git",
            help="Comma-separated list",
            key="profile_skills"
        )
    
    # Work experiences
    st.subheader("💼 Work Experience")
    num_experiences = st.number_input("Number of Experiences", min_value=0, max_value=5, value=2, key="num_exp")
    
    experiences = []
    exp_cols = st.columns(min(int(num_experiences), 2)) if num_experiences > 0 else []
    
    for i in range(int(num_experiences)):
        col_idx = i % 2
        with exp_cols[col_idx] if exp_cols else st.container():
            with st.expander(f"Experience #{i+1}", expanded=(i < 2)):
                exp_title = st.text_input(
                    f"Title", 
                    value="Data Science Intern" if i == 0 else "Research Assistant",
                    key=f"exp_title_{i}"
                )
                exp_org = st.text_input(
                    f"Organization",
                    value="Tech Company" if i == 0 else "University Lab",
                    key=f"exp_org_{i}"
                )
                exp_impact = st.text_area(
                    f"Impact/Achievement",
                    value="Built ML model improving prediction accuracy by 15%" if i == 0 else "Analyzed large datasets and created visualizations for research publication",
                    height=60,
                    key=f"exp_impact_{i}"
                )
                exp_skills = st.text_input(
                    f"Skills Used",
                    value="Python, Scikit-learn, Pandas",
                    key=f"exp_skills_{i}"
                )
                
                if exp_title and exp_org:
                    experiences.append({
                        "title": exp_title,
                        "org": exp_org,
                        "impact": exp_impact,
                        "skills": [s.strip() for s in exp_skills.split(",") if s.strip()]
                    })
    
    return build_profile_dict(name, major, goals, email, gpa, courses, skills, experiences)

def render_resume_input() -> str:
    """Render resume text input"""
    return st.text_area(
        "📄 Current Resume (plain text)",
        height=250,
        value="""University of Michigan — B.S. in Data Science (GPA 3.78)

EXPERIENCE
• Data Science Intern, TechCorp (Summer 2023)
  - Built ML models for customer segmentation, improving targeting by 25%
  - Developed automated data pipelines processing 1M+ records daily
  
• Research Assistant, UM Data Lab (2022-2023) 
  - Analyzed healthcare datasets using Python and R
  - Created interactive dashboards for research visualization

PROJECTS
• Sentiment Analysis System: Built transformer model achieving F1=0.86 on Twitter data
• Customer Churn Analysis: Used SQL and Tableau to identify key churn factors

SKILLS: Python, R, SQL, PyTorch, TensorFlow, AWS, Git, Tableau""",
        help="Your current resume content - this helps inform the generation",
        key="resume_text"
    )

# =============================================================================
# PAGE 1: PROGRAM MATCHING
# =============================================================================

def render_matching_page():
    """Render the Program Matching page"""
    st.header("🎯 Step 1: Find Your Best-Fit Programs")
    st.markdown("*Let our AI analyze your profile and recommend the most suitable graduate programs*")
    
    # Check API status
    api_info = get_api_info()
    matcher_available = api_info.get("matching_service_available", False)
    
    if not matcher_available:
        st.warning("⚠️ Matching service is not available. Please ensure the API server is running and corpus is loaded.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        profile = render_profile_form()
    
    with col2:
        st.subheader("⚙️ Matching Settings")
        
        top_k = st.slider(
            "Number of Programs to Return",
            min_value=1,
            max_value=20,
            value=5,
            help="How many top matching programs to show"
        )
        
        min_score = st.slider(
            "Minimum Match Score",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Only show programs with score above this threshold"
        )
        
        use_llm_explanation = st.checkbox(
            "Use LLM for Detailed Explanations",
            value=False,
            help="Generate AI-powered explanations for each match (slower but more insightful)"
        )
        
        st.markdown("---")
        st.subheader("📊 Dimension Weights")
        st.caption("Adjust how much each factor matters in matching")
        
        academic_weight = st.slider("Academic", 0.0, 1.0, 0.30, 0.05)
        skills_weight = st.slider("Skills", 0.0, 1.0, 0.25, 0.05)
        experience_weight = st.slider("Experience", 0.0, 1.0, 0.20, 0.05)
        goals_weight = st.slider("Goals", 0.0, 1.0, 0.15, 0.05)
        requirements_weight = st.slider("Requirements", 0.0, 1.0, 0.10, 0.05)
        
        # Normalize weights
        total = academic_weight + skills_weight + experience_weight + goals_weight + requirements_weight
        if total > 0:
            custom_weights = {
                "academic": academic_weight / total,
                "skills": skills_weight / total,
                "experience": experience_weight / total,
                "goals": goals_weight / total,
                "requirements": requirements_weight / total
            }
        else:
            custom_weights = None
    
    # Match button
    if st.button("🔍 Find Matching Programs", use_container_width=True, type="primary"):
        if not profile.get("major") or not profile.get("gpa"):
            st.error("Please provide at least your Major and GPA")
            return
        
        with st.spinner("Analyzing your profile and finding matches..."):
            try:
                payload = {
                    "profile": profile,
                    "top_k": top_k,
                    "min_score": min_score,
                    "use_llm_explanation": use_llm_explanation,
                    "custom_weights": custom_weights
                }
                
                resp = requests.post(
                    f"{API_BASE_URL}/match/programs",
                    json=payload,
                    timeout=60
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    
                    if data.get("success"):
                        st.success(f"✅ {data.get('message', 'Matching complete!')}")
                        
                        # Store results in session state
                        st.session_state.matched_programs = data.get("matches", [])
                        st.session_state.profile = profile
                        
                        # Display results
                        render_matching_results(data)
                    else:
                        st.error("Matching failed: " + data.get("message", "Unknown error"))
                else:
                    st.error(f"API Error {resp.status_code}: {resp.text}")
                    
            except requests.exceptions.Timeout:
                st.error("Request timed out. Try disabling LLM explanations.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the server is running.")
            except Exception as e:
                st.error(f"Error: {e}")

def render_matching_results(data: Dict):
    """Render matching results"""
    matches = data.get("matches", [])
    
    if not matches:
        st.warning("No programs found matching your criteria. Try lowering the minimum score.")
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Programs Evaluated", data.get("total_programs_evaluated", 0))
    with col2:
        st.metric("Programs Matched", len(matches))
    with col3:
        avg_score = sum(m.get("overall_score", 0) for m in matches) / len(matches) if matches else 0
        st.metric("Average Match Score", f"{avg_score:.2f}")
    
    # Overall insights
    if data.get("overall_insights"):
        with st.expander("💡 Overall Insights", expanded=True):
            for insight in data["overall_insights"]:
                st.info(insight)
    
    st.markdown("---")
    st.subheader("🏆 Top Matching Programs")
    
    for i, match in enumerate(matches):
        with st.expander(
            f"#{i+1} {match.get('university', 'Unknown')} - {match.get('program_name', 'Unknown Program')} "
            f"(Score: {match.get('overall_score', 0):.2f})",
            expanded=(i < 3)
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Match level badge
                match_level = match.get("match_level", "moderate")
                level_colors = {"excellent": "🟢", "good": "🟡", "moderate": "🟠", "poor": "🔴"}
                st.markdown(f"**Match Level:** {level_colors.get(match_level, '⚪')} {match_level.title()}")
                
                # Dimension scores
                st.markdown("**Dimension Scores:**")
                dim_scores = match.get("dimension_scores", {})
                for dim, score_info in dim_scores.items():
                    score_val = score_info.get("score", 0) if isinstance(score_info, dict) else score_info
                    progress_val = min(score_val, 1.0)
                    st.progress(progress_val, text=f"{dim.title()}: {score_val:.2f}")
                
                # Strengths and gaps
                if match.get("strengths"):
                    st.markdown("**✅ Your Strengths:**")
                    for s in match["strengths"][:3]:
                        st.write(f"• {s}")
                
                if match.get("gaps"):
                    st.markdown("**⚠️ Areas to Improve:**")
                    for g in match["gaps"][:3]:
                        st.write(f"• {g}")
            
            with col2:
                # Recommendations
                if match.get("recommendations"):
                    st.markdown("**📋 Recommendations:**")
                    for r in match["recommendations"][:3]:
                        st.caption(f"• {r}")
                
                # Explanation
                if match.get("explanation"):
                    st.markdown("**💬 AI Analysis:**")
                    st.caption(match["explanation"][:500] + "..." if len(match.get("explanation", "")) > 500 else match.get("explanation", ""))
            
            # Select button
            if st.button(f"Select this program", key=f"select_{i}"):
                st.session_state.selected_program = match
                st.success(f"Selected: {match.get('program_name')}")
                st.info("Go to 'Step 2: Generate Documents' to create your application materials!")

# =============================================================================
# PAGE 2: DOCUMENT GENERATION (Writing Agent)
# =============================================================================

def render_generation_page():
    """Render the Document Generation page"""
    st.header("✍️ Step 2: Generate Application Documents")
    st.markdown("*Use our AI Writing Agent to create high-quality application materials*")
    
    # Check API status
    api_info = get_api_info()
    writing_agent_available = api_info.get("writing_agent_available", False)
    
    if not writing_agent_available:
        st.warning("⚠️ LangGraph Writing Agent is not available. Falling back to basic generator.")
    
    # Show selected program if any
    if st.session_state.selected_program:
        program = st.session_state.selected_program
        st.info(f"🎯 **Target Program:** {program.get('university', '')} - {program.get('program_name', '')}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Profile section
        profile = render_profile_form()
        
        st.markdown("---")
        
        # Resume section
        resume_text = render_resume_input()
        
        st.markdown("---")
        
        # Program section
        st.subheader("🎯 Target Program Information")
        
        program_url = st.text_input(
            "Program URL (optional)",
            placeholder="https://example.edu/programs/data-science-ms",
            help="We'll automatically extract program information"
        )
        
        # Pre-fill program text if selected from matching
        default_program_text = ""
        if st.session_state.selected_program:
            prog = st.session_state.selected_program
            default_program_text = f"""{prog.get('program_name', 'Graduate Program')}
at {prog.get('university', 'University')}

This program focuses on training students in advanced methodologies and practical applications.
"""
        
        program_text = st.text_area(
            "Program Description",
            height=200,
            value=default_program_text or """Master of Science in Data Science

Program Mission: Train the next generation of data scientists to solve complex real-world problems using statistical methods, machine learning, and ethical data practices.

Core Curriculum:
- Statistical Inference and Modeling  
- Machine Learning and Deep Learning
- Data Management and Big Data Systems
- Data Visualization and Communication
- Ethics in Data Science
- Capstone Project

The program emphasizes hands-on experience with real datasets, collaboration with industry partners, and development of both technical and communication skills.""",
            help="Paste the program description here if no URL provided"
        )
    
    with col2:
        st.subheader("⚙️ Generation Settings")
        
        # Document type selection
        st.markdown("**📑 Document Type**")
        doc_type = st.selectbox(
            "Select document to generate",
            options=list(DOCUMENT_TYPES.keys()),
            format_func=lambda x: DOCUMENT_TYPES[x],
            key="doc_type_select"
        )
        
        st.markdown("---")
        
        # Generation system selection
        st.markdown("**🤖 Generation System**")
        generation_system = st.radio(
            "Choose generation method",
            options=["writing_agent", "multi_agent", "simple"],
            format_func=lambda x: {
                "writing_agent": "🚀 LangGraph Writing Agent (Best Quality)",
                "multi_agent": "⚡ Multi-Agent System (Balanced)",
                "simple": "💨 Simple Generator (Fastest)"
            }[x],
            help="Writing Agent uses advanced AI with RAG and reflection. Multi-Agent uses iterative refinement. Simple is template-based."
        )
        
        # LLM settings for Writing Agent
        if generation_system == "writing_agent":
            st.markdown("---")
            st.markdown("**🔧 LLM Configuration**")
            
            llm_provider = st.selectbox(
                "LLM Provider",
                options=list(LLM_PROVIDERS.keys()),
                format_func=lambda x: LLM_PROVIDERS[x]
            )
            
            model_name = st.text_input(
                "Model Name (optional)",
                placeholder="gpt-4, claude-3-opus, qwen-plus",
                help="Leave empty to use default model"
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher = more creative, Lower = more focused"
            )
            
            max_iterations = st.slider(
                "Max Refinement Iterations",
                min_value=1,
                max_value=5,
                value=3,
                help="More iterations = higher quality but slower"
            )
            
            quality_threshold = st.slider(
                "Quality Threshold",
                min_value=0.5,
                max_value=1.0,
                value=0.85,
                step=0.05,
                help="Content must reach this score to be approved"
            )
            
            use_corpus = st.checkbox(
                "Use RAG Corpus",
                value=True,
                help="Retrieve relevant examples from corpus to enhance generation"
            )
        
        elif generation_system == "multi_agent":
            st.markdown("---")
            st.markdown("**🔧 Multi-Agent Settings**")
            
            max_iterations = st.slider(
                "Max Iterations",
                min_value=1,
                max_value=5,
                value=3
            )
            
            quality_threshold = st.slider(
                "Critique Threshold",
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                step=0.05
            )
            
            fallback_enabled = st.checkbox("Enable Fallback", value=True)
        
        else:  # simple
            st.info("Simple generator uses templates without LLM calls.")
        
        st.markdown("---")
        topk = st.slider("Evidence Chunks (RAG)", 1, 10, 5)
    
    # Generate button
    if st.button("✨ Generate Document", use_container_width=True, type="primary"):
        if not profile.get("name"):
            st.error("Please provide your name")
            return
        
        if not program_text.strip() and not program_url.strip():
            st.error("Please provide program information (URL or description)")
            return
        
        with st.spinner(f"Generating {DOCUMENT_TYPES[doc_type]} using {generation_system}..."):
            try:
                start_time = time.time()
                
                if generation_system == "writing_agent":
                    # Use Writing Agent endpoint
                    payload = {
                        "profile": profile,
                        "resume_text": resume_text,
                        "program_text": program_text.strip() if program_text.strip() else None,
                        "program_url": program_url.strip() if program_url.strip() else None,
                        "document_type": doc_type,
                        "llm_provider": llm_provider,
                        "model_name": model_name if model_name else None,
                        "temperature": temperature,
                        "max_iterations": max_iterations,
                        "quality_threshold": quality_threshold,
                        "use_corpus": use_corpus,
                        "retrieval_topk": topk
                    }
                    
                    resp = requests.post(
                        f"{API_BASE_URL}/generate/writing-agent",
                        json=payload,
                        timeout=180
                    )
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        render_writing_agent_result(data, doc_type, time.time() - start_time)
                    else:
                        st.error(f"API Error {resp.status_code}: {resp.text}")
                
                else:
                    # Use original /generate endpoint
                    payload = {
                        "profile": profile,
                        "resume_text": resume_text,
                        "program_text": program_text.strip() if program_text.strip() else None,
                        "program_url": program_url.strip() if program_url.strip() else None,
                        "topk": topk,
                        "use_multi_agent": (generation_system == "multi_agent"),
                        "max_iterations": max_iterations if generation_system == "multi_agent" else 3,
                        "critique_threshold": quality_threshold if generation_system == "multi_agent" else 0.8,
                        "fallback_on_error": fallback_enabled if generation_system == "multi_agent" else True
                    }
                    
                    resp = requests.post(
                        f"{API_BASE_URL}/generate",
                        json=payload,
                        timeout=120
                    )
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        render_generation_result(data, time.time() - start_time)
                    else:
                        st.error(f"API Error {resp.status_code}: {resp.text}")
                        
            except requests.exceptions.Timeout:
                st.error("Request timed out. Try using a simpler generation method.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the server is running on http://localhost:8000")
            except Exception as e:
                st.error(f"Error: {e}")

def render_writing_agent_result(data: Dict, doc_type: str, gen_time: float):
    """Render Writing Agent generation result"""
    if not data.get("success"):
        st.error("Generation failed")
        return
    
    st.success(f"✅ Generated successfully in {gen_time:.1f}s!")
    
    # Quality metrics
    quality_report = data.get("quality_report", {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Generation Time", f"{data.get('generation_time_seconds', gen_time):.1f}s")
    with col2:
        st.metric("Iterations", data.get("iterations", 0))
    with col3:
        quality_score = quality_report.get("overall_score", 0)
        st.metric("Quality Score", f"{quality_score:.2f}")
    with col4:
        st.metric("Draft History", data.get("draft_history_length", 0))
    
    # Quality details
    if quality_report:
        with st.expander("📊 Quality Analysis", expanded=True):
            cols = st.columns(3)
            scores = quality_report.get("dimension_scores", {})
            for i, (dim, score) in enumerate(scores.items()):
                with cols[i % 3]:
                    st.progress(min(score, 1.0), text=f"{dim}: {score:.2f}")
            
            if quality_report.get("feedback"):
                st.markdown("**Feedback:**")
                st.caption(quality_report["feedback"])
            
            if quality_report.get("suggestions"):
                st.markdown("**Suggestions for improvement:**")
                for s in quality_report["suggestions"]:
                    st.caption(f"• {s}")
    
    # Document content
    st.markdown("---")
    st.subheader(f"{DOCUMENT_TYPES[doc_type]}")
    
    document = data.get("document", "")
    st.code(document, language="markdown")
    
    # Download button
    st.download_button(
        f"📥 Download {DOCUMENT_TYPES[doc_type]}",
        document.encode("utf-8"),
        file_name=f"{doc_type}.md",
        mime="text/markdown",
        use_container_width=True
    )
    
    # Store in session state
    st.session_state.generated_documents[doc_type] = document

def render_generation_result(data: Dict, gen_time: float):
    """Render standard generation result (multi-agent or simple)"""
    if "error" in data:
        st.error(f"Generation Error: {data['error']}")
        return
    
    system_used = data.get("system_used", "unknown")
    st.success(f"✅ Generated successfully using {system_used} system in {gen_time:.1f}s!")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("System Used", system_used.replace("_", " ").title())
    with col2:
        keywords_count = data.get("report", {}).get("generation_metadata", {}).get("keywords_extracted", 0)
        st.metric("Keywords Extracted", keywords_count)
    with col3:
        if system_used == "multi_agent":
            total_iterations = data.get("report", {}).get("overall_quality", {}).get("total_iterations", 0)
            st.metric("Total Iterations", total_iterations)
    
    # Content tabs
    texts = data.get("texts", {})
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Personal Statement", "📋 Resume Bullets", "📨 Recommendation", "📊 Report"])
    
    with tab1:
        st.code(texts.get("personal_statement", ""), language="markdown")
        st.download_button(
            "📥 Download",
            texts.get("personal_statement", "").encode("utf-8"),
            file_name="personal_statement.md",
            mime="text/markdown"
        )
    
    with tab2:
        st.code(texts.get("resume_bullets", ""), language="markdown")
        st.download_button(
            "📥 Download",
            texts.get("resume_bullets", "").encode("utf-8"),
            file_name="resume_bullets.md",
            mime="text/markdown"
        )
    
    with tab3:
        st.code(texts.get("reco_template", ""), language="markdown")
        st.download_button(
            "📥 Download",
            texts.get("reco_template", "").encode("utf-8"),
            file_name="recommendation.md",
            mime="text/markdown"
        )
    
    with tab4:
        st.json(data.get("report", {}))
    
    # Store all documents
    st.session_state.generated_documents = texts

# =============================================================================
# PAGE 3: END-TO-END FLOW
# =============================================================================

def render_e2e_page():
    """Render the End-to-End workflow page"""
    st.header("🚀 End-to-End Application Helper")
    st.markdown("*Complete workflow: Profile → Program Matching → Document Generation*")
    
    # Progress indicator
    steps = ["1️⃣ Enter Profile", "2️⃣ Match Programs", "3️⃣ Select Program", "4️⃣ Generate Documents"]
    current = st.session_state.current_step
    
    cols = st.columns(4)
    for i, step in enumerate(steps):
        with cols[i]:
            if i + 1 < current:
                st.success(step + " ✓")
            elif i + 1 == current:
                st.info(step + " ◀")
            else:
                st.caption(step)
    
    st.markdown("---")
    
    # Step 1: Profile
    if current == 1:
        st.subheader("Step 1: Enter Your Profile")
        profile = render_profile_form()
        resume = render_resume_input()
        
        if st.button("Next: Find Programs →", type="primary"):
            if profile.get("major") and profile.get("gpa"):
                st.session_state.profile = profile
                st.session_state.resume_text = resume
                st.session_state.current_step = 2
                st.rerun()
            else:
                st.error("Please provide at least Major and GPA")
    
    # Step 2: Match
    elif current == 2:
        st.subheader("Step 2: Finding Your Best-Fit Programs")
        
        profile = st.session_state.profile
        st.info(f"Profile: {profile.get('name', 'Unknown')} | {profile.get('major', 'Unknown')} | GPA: {profile.get('gpa', 'N/A')}")
        
        if st.button("🔍 Find Matching Programs", type="primary"):
            with st.spinner("Analyzing your profile..."):
                try:
                    payload = {
                        "profile": profile,
                        "top_k": 5,
                        "min_score": 0.5,
                        "use_llm_explanation": False
                    }
                    
                    resp = requests.post(f"{API_BASE_URL}/match/programs", json=payload, timeout=60)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.matched_programs = data.get("matches", [])
                        st.session_state.current_step = 3
                        st.rerun()
                    else:
                        st.error(f"Matching failed: {resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.button("← Back to Profile"):
            st.session_state.current_step = 1
            st.rerun()
    
    # Step 3: Select Program
    elif current == 3:
        st.subheader("Step 3: Select Your Target Program")
        
        matches = st.session_state.matched_programs
        
        if not matches:
            st.warning("No programs matched. Going back...")
            st.session_state.current_step = 2
            st.rerun()
        
        for i, match in enumerate(matches):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{i+1}. {match.get('university', '')} - {match.get('program_name', '')}**")
                st.caption(f"Match Score: {match.get('overall_score', 0):.2f} | Level: {match.get('match_level', 'N/A')}")
            with col2:
                if st.button("Select", key=f"e2e_select_{i}"):
                    st.session_state.selected_program = match
                    st.session_state.current_step = 4
                    st.rerun()
        
        if st.button("← Back to Matching"):
            st.session_state.current_step = 2
            st.rerun()
    
    # Step 4: Generate
    elif current == 4:
        st.subheader("Step 4: Generate Application Documents")
        
        program = st.session_state.selected_program
        profile = st.session_state.profile
        resume = st.session_state.get("resume_text", "")
        
        st.success(f"🎯 Target: {program.get('university', '')} - {program.get('program_name', '')}")
        
        doc_type = st.selectbox(
            "Document to Generate",
            options=list(DOCUMENT_TYPES.keys()),
            format_func=lambda x: DOCUMENT_TYPES[x]
        )
        
        if st.button("✨ Generate Document", type="primary"):
            with st.spinner("Generating..."):
                try:
                    payload = {
                        "profile": profile,
                        "resume_text": resume,
                        "program_text": f"{program.get('program_name', '')} at {program.get('university', '')}",
                        "document_type": doc_type,
                        "llm_provider": "openai",
                        "max_iterations": 2,
                        "quality_threshold": 0.8
                    }
                    
                    resp = requests.post(f"{API_BASE_URL}/generate/writing-agent", json=payload, timeout=120)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        st.markdown("---")
                        st.subheader(f"Generated {DOCUMENT_TYPES[doc_type]}")
                        st.code(data.get("document", ""), language="markdown")
                        
                        st.download_button(
                            "📥 Download",
                            data.get("document", "").encode("utf-8"),
                            file_name=f"{doc_type}.md",
                            mime="text/markdown"
                        )
                    else:
                        st.error(f"Generation failed: {resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Program Selection"):
                st.session_state.current_step = 3
                st.rerun()
        with col2:
            if st.button("🔄 Start Over"):
                st.session_state.current_step = 1
                st.session_state.matched_programs = []
                st.session_state.selected_program = None
                st.rerun()

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/graduation-cap.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Choose a workflow:",
            options=["🎯 Program Matching", "✍️ Document Generation", "🚀 End-to-End Flow", "📊 Dashboard"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # API Status
        st.subheader("🔌 API Status")
        health = check_api_health()
        
        if health["status"] == "healthy":
            st.success("API Connected ✓")
            
            api_info = get_api_info()
            if api_info.get("writing_agent_available"):
                st.caption("✓ Writing Agent")
            else:
                st.caption("✗ Writing Agent")
            
            if api_info.get("matching_service_available"):
                st.caption("✓ Matching Service")
            else:
                st.caption("✗ Matching Service")
        else:
            st.error("API Disconnected")
            st.caption(health.get("message", ""))
            st.caption("Run: `python -m src.rag_service.api`")
        
        st.markdown("---")
        st.caption("College Application Helper v3.0")
        st.caption("Enhanced with AI Matching & Writing Agents")
    
    # Main content
    if "Program Matching" in page:
        render_matching_page()
    elif "Document Generation" in page:
        render_generation_page()
    elif "End-to-End" in page:
        render_e2e_page()
    elif "Dashboard" in page:
        render_dashboard()

def render_dashboard():
    """Render a dashboard showing session state and generated content"""
    st.header("📊 Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Current Profile")
        if st.session_state.profile:
            st.json(st.session_state.profile)
        else:
            st.caption("No profile entered yet")
    
    with col2:
        st.subheader("🎯 Selected Program")
        if st.session_state.selected_program:
            prog = st.session_state.selected_program
            st.markdown(f"**{prog.get('university', '')}**")
            st.markdown(f"{prog.get('program_name', '')}")
            st.metric("Match Score", f"{prog.get('overall_score', 0):.2f}")
        else:
            st.caption("No program selected yet")
    
    st.markdown("---")
    
    st.subheader("📝 Generated Documents")
    docs = st.session_state.generated_documents
    
    if docs:
        tabs = st.tabs(list(docs.keys()))
        for i, (doc_type, content) in enumerate(docs.items()):
            with tabs[i]:
                st.code(content[:1000] + "..." if len(content) > 1000 else content, language="markdown")
    else:
        st.caption("No documents generated yet")
    
    st.markdown("---")
    
    st.subheader("🏆 Matched Programs")
    if st.session_state.matched_programs:
        for i, match in enumerate(st.session_state.matched_programs[:5]):
            st.caption(f"{i+1}. {match.get('university', '')} - {match.get('program_name', '')} (Score: {match.get('overall_score', 0):.2f})")
    else:
        st.caption("No programs matched yet")

if __name__ == "__main__":
    main()
