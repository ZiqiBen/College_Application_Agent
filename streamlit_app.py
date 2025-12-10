"""
College Application Helper - Full-Featured Streamlit Frontend (v3.1)

This comprehensive Streamlit app provides a complete user interface for the 
College Application Helper system with the following workflows:

1. **Smart Matching + Generation**: Complete flow from profile → matching → select program → generate documents
2. **Quick Generation (Manual Input)**: Direct document generation with manual program input
3. **Dashboard**: View history and generated content

Key Features:
- Integration with Matching Service for intelligent program recommendations
- Seamless flow from matching to document generation with auto-filled program info
- LangGraph-based Writing Agent for high-quality document generation
- Support for multiple LLM providers (OpenAI, Anthropic, Qwen)
- Manual input option for users who know their target program
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
    page_title="College Application Helper",
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
    "qwen": "Qwen (Tongyi Qianwen)"
}

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        "profile": None,
        "resume_text": "",
        "matched_programs": [],
        "selected_program": None,
        "selected_program_details": None,
        "generated_documents": {},
        "generation_mode": None,  # "matched" or "manual"
        "flow_step": "profile",  # "profile", "matching", "select", "generate"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

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

def get_program_details(program_id: str) -> Optional[Dict]:
    """Fetch complete program details from API"""
    try:
        resp = requests.get(f"{API_BASE_URL}/match/program/{program_id}/details", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                return data.get("program")
        return None
    except Exception as e:
        print(f"Error fetching program details: {e}")
        return None

# =============================================================================
# UI COMPONENTS - PROFILE INPUT
# =============================================================================

def render_profile_form(key_prefix: str = "") -> Dict:
    """Render the profile input form and return profile data"""
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Name", value="Alice Zhang", key=f"{key_prefix}name")
        major = st.text_input("Major/Field", value="Data Science", key=f"{key_prefix}major")
        goals = st.text_area(
            "Career Goals",
            value="Apply ML to real-world analytics and product data science. Lead data-driven initiatives in tech companies.",
            height=100,
            help="Be specific about your career aspirations",
            key=f"{key_prefix}goals"
        )
        
        with st.expander("📧 Contact & Academic Info"):
            email = st.text_input("Email", value="alice@example.com", key=f"{key_prefix}email")
            gpa = st.text_input("GPA (0-4.0)", value="3.78", key=f"{key_prefix}gpa")
    
    with col2:
        courses = st.text_area(
            "Relevant Courses (comma-separated)",
            value="Machine Learning, Deep Learning, Statistical Inference, Data Management, Algorithms",
            key=f"{key_prefix}courses"
        )
        skills = st.text_area(
            "Technical Skills (comma-separated)",
            value="Python, SQL, PyTorch, TensorFlow, R, Tableau, AWS, Git",
            key=f"{key_prefix}skills"
        )
    
    # Work experiences
    st.markdown("##### 💼 Work Experience")
    num_experiences = st.number_input("Number of Experiences", min_value=0, max_value=5, value=2, key=f"{key_prefix}num_exp")
    
    experiences = []
    if num_experiences > 0:
        exp_cols = st.columns(min(int(num_experiences), 2))
        for i in range(int(num_experiences)):
            col_idx = i % 2
            with exp_cols[col_idx]:
                with st.expander(f"Experience #{i+1}", expanded=(i < 2)):
                    exp_title = st.text_input(
                        "Title", 
                        value="Data Science Intern" if i == 0 else "Research Assistant",
                        key=f"{key_prefix}exp_title_{i}"
                    )
                    exp_org = st.text_input(
                        "Organization",
                        value="Tech Company" if i == 0 else "University Lab",
                        key=f"{key_prefix}exp_org_{i}"
                    )
                    exp_impact = st.text_area(
                        "Impact/Achievement",
                        value="Built ML model improving prediction accuracy by 15%" if i == 0 else "Analyzed large datasets and created visualizations for research publication",
                        height=60,
                        key=f"{key_prefix}exp_impact_{i}"
                    )
                    exp_skills = st.text_input(
                        "Skills Used",
                        value="Python, Scikit-learn, Pandas",
                        key=f"{key_prefix}exp_skills_{i}"
                    )
                    
                    if exp_title and exp_org:
                        experiences.append({
                            "title": exp_title,
                            "org": exp_org,
                            "impact": exp_impact,
                            "skills": [s.strip() for s in exp_skills.split(",") if s.strip()]
                        })
    
    return build_profile_dict(name, major, goals, email, gpa, courses, skills, experiences)

def render_resume_input(key_prefix: str = "") -> str:
    """Render resume text input"""
    return st.text_area(
        "📄 Current Resume (plain text)",
        height=200,
        value="""University of Michigan — B.S. in Data Science (GPA 3.78)

EXPERIENCE
• Data Science Intern, TechCorp (Summer 2023)
  - Built ML models for customer segmentation, improving targeting by 25%
  - Developed automated data pipelines processing 1M+ records daily
  
• Research Assistant, UM Data Lab (2022-2023) 
  - Analyzed healthcare datasets using Python and R
  - Created interactive dashboards for research visualization

PROJECTS
• Sentiment Analysis System: Built transformer model achieving F1=0.86
• Customer Churn Analysis: Used SQL and Tableau to identify key churn factors

SKILLS: Python, R, SQL, PyTorch, TensorFlow, AWS, Git, Tableau""",
        help="Your current resume content",
        key=f"{key_prefix}resume"
    )

# =============================================================================
# PAGE 1: Smart Matching + Generation (MAIN WORKFLOW)
# =============================================================================

def render_smart_matching_page():
    """Render the Smart Matching + Generation page"""
    st.header("🎯 Smart Matching + Document Generation")
    st.markdown("*Complete workflow: Input Profile → Smart Match Programs → Select Program → Generate Application Documents*")
    
    # Progress indicator
    steps = {
        "profile": "1️⃣ Input Info",
        "matching": "2️⃣ Match Results", 
        "select": "3️⃣ Select Program",
        "generate": "4️⃣ Generate Docs"
    }
    
    current_step = st.session_state.flow_step
    step_cols = st.columns(4)
    step_order = ["profile", "matching", "select", "generate"]
    current_idx = step_order.index(current_step)
    
    for i, (step_key, step_name) in enumerate(steps.items()):
        with step_cols[i]:
            if i < current_idx:
                st.success(step_name + " ✓")
            elif i == current_idx:
                st.info(step_name + " ◀")
            else:
                st.caption(step_name)
    
    st.markdown("---")
    
    # Render current step
    if current_step == "profile":
        render_profile_step()
    elif current_step == "matching":
        render_matching_step()
    elif current_step == "select":
        render_select_step()
    elif current_step == "generate":
        render_generate_step()

def render_profile_step():
    """Step 1: Profile input"""
    st.subheader("👤 Enter Your Application Information")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        profile = render_profile_form(key_prefix="smart_")
        st.markdown("---")
        resume_text = render_resume_input(key_prefix="smart_")
    
    with col2:
        st.markdown("##### ⚙️ Matching Settings")
        top_k = st.slider("Number of Programs to Return", 1, 15, 5, key="smart_topk")
        min_score = st.slider("Minimum Match Score", 0.0, 1.0, 0.4, 0.05, key="smart_min_score")
        
        st.markdown("---")
        st.markdown("##### 📊 Dimension Weights")
        st.caption("Adjust the importance of each factor")
        
        academic_weight = st.slider("Academic Background", 0.0, 1.0, 0.30, 0.05, key="w_academic")
        skills_weight = st.slider("Skills Match", 0.0, 1.0, 0.25, 0.05, key="w_skills")
        experience_weight = st.slider("Work Experience", 0.0, 1.0, 0.20, 0.05, key="w_experience")
        goals_weight = st.slider("Goals Alignment", 0.0, 1.0, 0.15, 0.05, key="w_goals")
        requirements_weight = st.slider("Application Requirements", 0.0, 1.0, 0.10, 0.05, key="w_requirements")
        
        # Normalize weights
        total = academic_weight + skills_weight + experience_weight + goals_weight + requirements_weight
        custom_weights = None
        if total > 0:
            custom_weights = {
                "academic": academic_weight / total,
                "skills": skills_weight / total,
                "experience": experience_weight / total,
                "goals": goals_weight / total,
                "requirements": requirements_weight / total
            }
    
    # Action button
    if st.button("🔍 Start Matching Programs", type="primary", use_container_width=True):
        if not profile.get("major") or not profile.get("gpa"):
            st.error("Please fill in at least Major and GPA")
            return
        
        # Save to session state
        st.session_state.profile = profile
        st.session_state.resume_text = resume_text
        
        # Perform matching
        with st.spinner("Analyzing your background and matching the best programs..."):
            try:
                payload = {
                    "profile": profile,
                    "top_k": top_k,
                    "min_score": min_score,
                    "use_llm_explanation": False,
                    "custom_weights": custom_weights
                }
                
                resp = requests.post(f"{API_BASE_URL}/match/programs", json=payload, timeout=60)
                
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("success"):
                        st.session_state.matched_programs = data.get("matches", [])
                        st.session_state.flow_step = "matching"
                        st.rerun()
                    else:
                        st.error("Matching failed: " + data.get("message", "Unknown error"))
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API server. Please ensure backend service is running.")
            except Exception as e:
                st.error(f"Error: {e}")

def render_matching_step():
    """Step 2: Show matching results"""
    st.subheader("🏆 Matching Results")
    
    matches = st.session_state.matched_programs
    
    if not matches:
        st.warning("No programs found matching criteria. Please try lowering the minimum score threshold.")
        if st.button("← Back to Modify"):
            st.session_state.flow_step = "profile"
            st.rerun()
        return
    
    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Matched Programs", len(matches))
    with col2:
        avg_score = sum(m.get("overall_score", 0) for m in matches) / len(matches)
        st.metric("Average Match Score", f"{avg_score:.2f}")
    with col3:
        top_score = max(m.get("overall_score", 0) for m in matches)
        st.metric("Highest Match Score", f"{top_score:.2f}")
    
    st.markdown("---")
    st.markdown("**Click 'Select This Program' in a program card to proceed to document generation**")
    
    # Display matches
    for i, match in enumerate(matches):
        match_level = match.get("match_level", "moderate")
        level_icons = {"excellent": "🟢", "good": "🟡", "moderate": "🟠", "weak": "🔴"}
        
        with st.expander(
            f"{level_icons.get(match_level, '⚪')} #{i+1} {match.get('university', '')} - {match.get('program_name', '')} "
            f"(Score: {match.get('overall_score', 0):.2f})",
            expanded=(i < 3)
        ):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Dimension scores
                st.markdown("**Dimension Scores:**")
                dim_scores = match.get("dimension_scores", {})
                score_cols = st.columns(5)
                dim_names = {"academic": "Academic", "skills": "Skills", "experience": "Experience", "goals": "Goals", "requirements": "Requirements"}
                
                for j, (dim, score_info) in enumerate(dim_scores.items()):
                    score_val = score_info.get("score", 0) if isinstance(score_info, dict) else score_info
                    with score_cols[j % 5]:
                        st.metric(dim_names.get(dim, dim), f"{score_val:.2f}")
                
                # Why This Program Fits You (personalized reasons)
                if match.get("fit_reasons"):
                    st.markdown("**🎯 Why This Program Fits You:**")
                    for reason in match["fit_reasons"][:3]:
                        st.caption(f"• {reason}")
                elif match.get("strengths"):
                    # Fallback to old strengths if fit_reasons not available
                    st.markdown("**✅ Your Strengths:**")
                    for s in match["strengths"][:3]:
                        st.caption(f"• {s}")
            
            with col2:
                # Program details preview
                details = match.get("program_details", {})
                if details:
                    if details.get("focus_areas"):
                        st.markdown("**Research Areas:**")
                        st.caption(", ".join(details["focus_areas"][:3]))
                    if details.get("core_courses"):
                        st.markdown("**Core Courses:**")
                        st.caption(", ".join(details["core_courses"][:3]))
                
                # Select button
                st.markdown("---")
                if st.button("✅ Select This Program", key=f"select_match_{i}", use_container_width=True):
                    st.session_state.selected_program = match
                    st.session_state.selected_program_details = match.get("program_details")
                    st.session_state.generation_mode = "matched"
                    st.session_state.flow_step = "select"
                    st.rerun()
    
    # Navigation
    st.markdown("---")
    if st.button("← Back to Modify Profile"):
        st.session_state.flow_step = "profile"
        st.rerun()

def render_select_step():
    """Step 3: Confirm selected program"""
    st.subheader("📋 Confirm Selected Program")
    
    program = st.session_state.selected_program
    details = st.session_state.selected_program_details
    
    if not program:
        st.warning("No program selected")
        st.session_state.flow_step = "matching"
        st.rerun()
        return
    
    # Display selected program
    st.success(f"🎯 Selected: **{program.get('university', '')} - {program.get('program_name', '')}**")
    st.metric("Match Score", f"{program.get('overall_score', 0):.2f}")
    
    # Show program details
    if details:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Program Information")
            if details.get("description_text"):
                st.text_area(
                    "Program Description",
                    value=details["description_text"][:1500] + "..." if len(details.get("description_text", "")) > 1500 else details.get("description_text", ""),
                    height=200,
                    disabled=True
                )
            
            if details.get("focus_areas"):
                st.markdown(f"**Research Areas:** {', '.join(details['focus_areas'])}")
            
            if details.get("core_courses"):
                st.markdown(f"**Core Courses:** {', '.join(details['core_courses'][:5])}")
        
        with col2:
            st.markdown("##### Application Requirements")
            if details.get("min_gpa"):
                st.markdown(f"**Minimum GPA:** {details['min_gpa']}")
            if details.get("required_skills"):
                st.markdown(f"**Required Skills:** {', '.join(details['required_skills'][:5])}")
            if details.get("source_url"):
                st.markdown(f"**Program Link:** [View Official Site]({details['source_url']})")
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← Back to Select Other Program"):
            st.session_state.flow_step = "matching"
            st.rerun()
    with col2:
        pass
    with col3:
        if st.button("Continue to Generate Documents →", type="primary"):
            st.session_state.flow_step = "generate"
            st.rerun()

def render_generate_step():
    """Step 4: Generate documents"""
    st.subheader("✨ Generate Application Documents")
    
    program = st.session_state.selected_program
    details = st.session_state.selected_program_details
    profile = st.session_state.profile
    resume_text = st.session_state.resume_text
    
    if not program or not profile:
        st.warning("Information incomplete, please return to previous steps")
        return
    
    # Show context
    st.info(f"🎯 **Target Program:** {program.get('university', '')} - {program.get('program_name', '')}")
    st.info(f"👤 **Applicant:** {profile.get('name', '')} | {profile.get('major', '')} | GPA: {profile.get('gpa', 'N/A')}")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("##### 📑 Select Document to Generate")
        doc_type = st.radio(
            "Document Type",
            options=list(DOCUMENT_TYPES.keys()),
            format_func=lambda x: DOCUMENT_TYPES[x],
            horizontal=True,
            key="gen_doc_type"
        )
    
    with col2:
        st.markdown("##### ⚙️ Generation Settings")
        
        llm_provider = st.selectbox(
            "LLM Provider",
            options=list(LLM_PROVIDERS.keys()),
            format_func=lambda x: LLM_PROVIDERS[x],
            key="gen_llm"
        )
        
        temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.7, 0.1, key="gen_temp")
        max_iterations = st.slider("Max Iterations", 1, 5, 3, key="gen_iter")
        quality_threshold = st.slider("Quality Threshold", 0.5, 1.0, 0.85, 0.05, key="gen_quality")
    
    st.markdown("---")
    
    # Generate button
    if st.button("✨ Start Generation", type="primary", use_container_width=True):
        # Build program text from details
        program_text = ""
        if details:
            program_text = f"""
{details.get('program_name', program.get('program_name', 'Graduate Program'))}
at {details.get('university', program.get('university', 'University'))}

{details.get('description_text', '')}

Focus Areas: {', '.join(details.get('focus_areas', []))}
Core Courses: {', '.join(details.get('core_courses', []))}
Required Skills: {', '.join(details.get('required_skills', []))}
"""
        else:
            program_text = f"{program.get('program_name', '')} at {program.get('university', '')}"
        
        with st.spinner(f"Generating {DOCUMENT_TYPES[doc_type]} using {LLM_PROVIDERS[llm_provider]}..."):
            try:
                payload = {
                    "profile": profile,
                    "resume_text": resume_text,
                    "program_text": program_text,
                    "document_type": doc_type,
                    "llm_provider": llm_provider,
                    "temperature": temperature,
                    "max_iterations": max_iterations,
                    "quality_threshold": quality_threshold,
                    "use_corpus": True
                }
                
                resp = requests.post(
                    f"{API_BASE_URL}/generate/writing-agent",
                    json=payload,
                    timeout=180
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    render_generation_result(data, doc_type)
                else:
                    error_text = resp.text
                    try:
                        error_json = resp.json()
                        if "detail" in error_json:
                            error_text = str(error_json["detail"])
                    except:
                        pass
                    st.error(f"Generation failed: {error_text}")
                    
            except requests.exceptions.Timeout:
                st.error("Request timeout, please try reducing iterations.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API server.")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← Back to Confirm Program"):
            st.session_state.flow_step = "select"
            st.rerun()
    with col2:
        if st.button("🔄 Start Over"):
            st.session_state.flow_step = "profile"
            st.session_state.matched_programs = []
            st.session_state.selected_program = None
            st.session_state.selected_program_details = None
            st.rerun()

def render_generation_result(data: Dict, doc_type: str):
    """Render the generation result"""
    if not data.get("success"):
        st.error("Generation failed")
        return
    
    st.success("✅ Document generated successfully!")
    
    # Quality metrics
    quality_report = data.get("quality_report", {})
    iterations = data.get("iterations", 0)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Generation Time", f"{data.get('generation_time_seconds', 0):.1f}s")
    with col2:
        st.metric("Iterations", iterations)
    with col3:
        quality = quality_report.get("final_score", 0)
        st.metric("Final Quality Score", f"{quality:.2f}")
    with col4:
        approved = quality_report.get("approved", False)
        st.metric("Status", "✅ Approved" if approved else "⚠️ Below Threshold")
    
    # Show iteration history if available
    iteration_history = quality_report.get("iteration_history", [])
    if iteration_history and len(iteration_history) > 1:
        st.markdown("---")
        st.markdown("##### 📈 Quality Improvement Progress")
        
        # Create a simple progress visualization
        progress_cols = st.columns(len(iteration_history))
        for idx, log in enumerate(iteration_history):
            with progress_cols[idx]:
                iter_score = log.get("overall_score", 0)
                st.metric(
                    f"Iteration {idx + 1}",
                    f"{iter_score:.2f}",
                    delta=f"+{iter_score - iteration_history[idx-1].get('overall_score', 0):.2f}" if idx > 0 else None
                )
        
        # Show suggestions used for improvement
        with st.expander("View Improvement Details"):
            for idx, log in enumerate(iteration_history):
                st.markdown(f"**Iteration {idx + 1}** (Score: {log.get('overall_score', 0):.2f})")
                suggestions = log.get("suggestions", [])
                if suggestions:
                    for s in suggestions[:3]:
                        st.caption(f"  → {s}")
                st.markdown("---")
    
    # Document content
    document = data.get("document", "")
    
    st.markdown("---")
    st.markdown(f"##### {DOCUMENT_TYPES[doc_type]}")
    st.code(document, language="markdown")
    
    # Download
    st.download_button(
        f"📥 Download {DOCUMENT_TYPES[doc_type]}",
        document.encode("utf-8"),
        file_name=f"{doc_type}.md",
        mime="text/markdown",
        use_container_width=True
    )
    
    # Save to session
    st.session_state.generated_documents[doc_type] = document

# =============================================================================
# PAGE 2: Quick Generation (Manual Input)
# =============================================================================

def render_manual_generation_page():
    """Render the Manual Generation page for users who know their target program"""
    st.header("✍️ Quick Generation (Manual Input)")
    st.markdown("*Already know your target program? Enter program info directly to generate application documents*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("##### 👤 Applicant Information")
        profile = render_profile_form(key_prefix="manual_")
        
        st.markdown("---")
        resume_text = render_resume_input(key_prefix="manual_")
        
        st.markdown("---")
        st.markdown("##### 🎯 Target Program Information")
        
        input_method = st.radio(
            "Input Method",
            options=["url", "text"],
            format_func=lambda x: "📎 Fetch via URL" if x == "url" else "📝 Direct Text Input",
            horizontal=True,
            key="manual_input_method"
        )
        
        program_url = ""
        program_text = ""
        
        if input_method == "url":
            program_url = st.text_input(
                "Program URL",
                placeholder="https://example.edu/programs/data-science-ms",
                help="We will automatically extract program information",
                key="manual_url"
            )
        else:
            program_text = st.text_area(
                "Program Description",
                height=200,
                placeholder="Paste program description, curriculum, application requirements, etc...",
                value="""Master of Science in Data Science

Program Mission: Train the next generation of data scientists to solve complex real-world problems using statistical methods, machine learning, and ethical data practices.

Core Curriculum:
- Statistical Inference and Modeling  
- Machine Learning and Deep Learning
- Data Management and Big Data Systems
- Data Visualization and Communication
- Ethics in Data Science

The program emphasizes hands-on experience with real datasets, collaboration with industry partners, and development of both technical and communication skills.""",
                key="manual_text"
            )
    
    with col2:
        st.markdown("##### 📑 Document Type")
        doc_type = st.radio(
            "Select document to generate",
            options=list(DOCUMENT_TYPES.keys()),
            format_func=lambda x: DOCUMENT_TYPES[x],
            key="manual_doc_type"
        )
        
        st.markdown("---")
        st.markdown("##### ⚙️ Generation Settings")
        
        generation_system = st.radio(
            "Generation System",
            options=["writing_agent", "multi_agent", "simple"],
            format_func=lambda x: {
                "writing_agent": "🚀 Writing Agent (Highest Quality)",
                "multi_agent": "⚡ Multi-Agent (Balanced)",
                "simple": "💨 Simple Mode (Fastest)"
            }[x],
            key="manual_system"
        )
        
        if generation_system == "writing_agent":
            llm_provider = st.selectbox(
                "LLM Provider",
                options=list(LLM_PROVIDERS.keys()),
                format_func=lambda x: LLM_PROVIDERS[x],
                key="manual_llm"
            )
            temperature = st.slider("Creativity", 0.0, 1.0, 0.7, 0.1, key="manual_temp")
            max_iterations = st.slider("Max Iterations", 1, 5, 3, key="manual_iter")
            quality_threshold = st.slider("Quality Threshold", 0.5, 1.0, 0.85, 0.05, key="manual_quality")
        
        elif generation_system == "multi_agent":
            max_iterations = st.slider("Max Iterations", 1, 5, 3, key="manual_ma_iter")
            quality_threshold = st.slider("Quality Threshold", 0.5, 1.0, 0.8, 0.05, key="manual_ma_quality")
        
        topk = st.slider("Number of Evidence Chunks", 1, 10, 5, key="manual_topk")
    
    # Generate button
    if st.button("✨ Generate Document", type="primary", use_container_width=True):
        if not profile.get("name"):
            st.error("Please fill in name")
            return
        
        if input_method == "url" and not program_url.strip():
            st.error("Please enter program URL")
            return
        elif input_method == "text" and not program_text.strip():
            st.error("Please enter program description")
            return
        
        with st.spinner(f"Generating {DOCUMENT_TYPES[doc_type]}..."):
            try:
                if generation_system == "writing_agent":
                    payload = {
                        "profile": profile,
                        "resume_text": resume_text,
                        "program_text": program_text.strip() if program_text.strip() else None,
                        "program_url": program_url.strip() if program_url.strip() else None,
                        "document_type": doc_type,
                        "llm_provider": llm_provider,
                        "temperature": temperature,
                        "max_iterations": max_iterations,
                        "quality_threshold": quality_threshold,
                        "use_corpus": True,
                        "retrieval_topk": topk
                    }
                    
                    resp = requests.post(f"{API_BASE_URL}/generate/writing-agent", json=payload, timeout=180)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        render_generation_result(data, doc_type)
                    else:
                        st.error(f"Generation failed: {resp.text}")
                
                else:
                    # Use standard /generate endpoint
                    payload = {
                        "profile": profile,
                        "resume_text": resume_text,
                        "program_text": program_text.strip() if program_text.strip() else None,
                        "program_url": program_url.strip() if program_url.strip() else None,
                        "topk": topk,
                        "use_multi_agent": (generation_system == "multi_agent"),
                        "max_iterations": max_iterations if generation_system == "multi_agent" else 3,
                        "critique_threshold": quality_threshold if generation_system == "multi_agent" else 0.8,
                        "fallback_on_error": True
                    }
                    
                    resp = requests.post(f"{API_BASE_URL}/generate", json=payload, timeout=120)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        render_standard_result(data)
                    else:
                        st.error(f"Generation failed: {resp.text}")
                        
            except requests.exceptions.Timeout:
                st.error("Request timeout, please try using a simpler generation mode.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API server.")
            except Exception as e:
                st.error(f"Error: {e}")

def render_standard_result(data: Dict):
    """Render standard generation result (from /generate endpoint)"""
    if "error" in data:
        st.error(f"Generation error: {data['error']}")
        return
    
    system_used = data.get("system_used", "unknown")
    st.success(f"✅ Successfully generated using {system_used} system!")
    
    # Content tabs
    texts = data.get("texts", {})
    tabs = st.tabs(["📝 Personal Statement", "📋 Resume Bullets", "📨 Recommendation"])
    
    with tabs[0]:
        content = texts.get("personal_statement", "")
        st.code(content, language="markdown")
        st.download_button("📥 Download", content.encode("utf-8"), "personal_statement.md", "text/markdown")
    
    with tabs[1]:
        content = texts.get("resume_bullets", "")
        st.code(content, language="markdown")
        st.download_button("📥 Download", content.encode("utf-8"), "resume_bullets.md", "text/markdown")
    
    with tabs[2]:
        content = texts.get("reco_template", "")
        st.code(content, language="markdown")
        st.download_button("📥 Download", content.encode("utf-8"), "recommendation.md", "text/markdown")

# =============================================================================
# PAGE 3: DASHBOARD
# =============================================================================

def render_dashboard():
    """Render the Dashboard page"""
    st.header("📊 Dashboard")
    st.markdown("*View current session status and generated content*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 👤 Current Profile")
        if st.session_state.profile:
            profile = st.session_state.profile
            st.markdown(f"**Name:** {profile.get('name', 'N/A')}")
            st.markdown(f"**Major:** {profile.get('major', 'N/A')}")
            st.markdown(f"**GPA:** {profile.get('gpa', 'N/A')}")
            st.markdown(f"**Skills:** {', '.join(profile.get('skills', [])[:5])}")
            
            with st.expander("View Full Profile"):
                st.json(profile)
        else:
            st.caption("Profile not entered yet")
    
    with col2:
        st.markdown("##### 🎯 Selected Program")
        if st.session_state.selected_program:
            prog = st.session_state.selected_program
            st.markdown(f"**University:** {prog.get('university', 'N/A')}")
            st.markdown(f"**Program:** {prog.get('program_name', 'N/A')}")
            st.metric("Match Score", f"{prog.get('overall_score', 0):.2f}")
        else:
            st.caption("Program not selected yet")
    
    st.markdown("---")
    
    # Matched programs
    st.markdown("##### 🏆 Matched Programs")
    if st.session_state.matched_programs:
        for i, match in enumerate(st.session_state.matched_programs[:5]):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.caption(f"{i+1}. {match.get('university', '')} - {match.get('program_name', '')}")
            with col2:
                st.caption(f"Score: {match.get('overall_score', 0):.2f}")
            with col3:
                if st.button("Select", key=f"dash_select_{i}"):
                    st.session_state.selected_program = match
                    st.session_state.selected_program_details = match.get("program_details")
                    st.session_state.generation_mode = "matched"
                    st.session_state.flow_step = "select"
                    st.rerun()
    else:
        st.caption("No matching performed yet")
    
    st.markdown("---")
    
    # Generated documents
    st.markdown("##### 📝 Generated Documents")
    docs = st.session_state.generated_documents
    
    if docs:
        for doc_type, content in docs.items():
            with st.expander(f"{DOCUMENT_TYPES.get(doc_type, doc_type)}"):
                st.code(content[:500] + "..." if len(content) > 500 else content, language="markdown")
                st.download_button(
                    "📥 Download",
                    content.encode("utf-8"),
                    f"{doc_type}.md",
                    "text/markdown",
                    key=f"dash_download_{doc_type}"
                )
    else:
        st.caption("No documents generated yet")
    
    # Clear session
    st.markdown("---")
    if st.button("🗑️ Clear All Data", type="secondary"):
        for key in ["profile", "resume_text", "matched_programs", "selected_program", 
                    "selected_program_details", "generated_documents", "flow_step"]:
            if key in st.session_state:
                if key == "flow_step":
                    st.session_state[key] = "profile"
                elif key in ["matched_programs", "generated_documents"]:
                    st.session_state[key] = {} if key == "generated_documents" else []
                else:
                    st.session_state[key] = None
        st.success("All data cleared")
        st.rerun()

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Sidebar
    with st.sidebar:
        st.title("🎓 College App Helper")
        st.caption("v3.1 - Smart Matching + AI Writing")
        
        st.markdown("---")
        
        page = st.radio(
            "Select Function",
            options=["smart", "manual", "dashboard"],
            format_func=lambda x: {
                "smart": "🎯 Smart Matching + Generation",
                "manual": "✍️ Quick Generation (Manual)",
                "dashboard": "📊 Dashboard"
            }[x],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # API Status
        st.markdown("##### 🔌 Service Status")
        health = check_api_health()
        
        if health["status"] == "healthy":
            st.success("API Connected ✓")
            
            api_info = get_api_info()
            col1, col2 = st.columns(2)
            with col1:
                if api_info.get("writing_agent_available"):
                    st.caption("✓ Writing Agent")
                else:
                    st.caption("✗ Writing Agent")
            with col2:
                if api_info.get("matching_service_available"):
                    st.caption("✓ Matching")
                else:
                    st.caption("✗ Matching")
        else:
            st.error("API Not Connected")
            st.caption(health.get("message", ""))
            st.code("python -m src.rag_service.api", language="bash")
        
        st.markdown("---")
        st.caption("© 2024 DS301 Project")
    
    # Main content
    if page == "smart":
        render_smart_matching_page()
    elif page == "manual":
        render_manual_generation_page()
    elif page == "dashboard":
        render_dashboard()

if __name__ == "__main__":
    main()
