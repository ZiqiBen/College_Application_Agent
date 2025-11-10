"""
College Application Helper - Multi-Agent Streamlit Frontend

This Streamlit app provides a user interface for the College Application Helper system.
It allows users to input their academic profile, resume, and target program information,
and then generates application artifacts (Personal Statement, Resume Bullets,
Recommendation Letter) using a Retrieval-Augmented Generation (RAG) backend.

Key Features:
- Sidebar configuration for selecting system type (Multi-Agent, Simple, Auto-Select)
  and controlling retrieval + generation parameters.
- Form-based input for applicant profile, resume text, and program description or URL.
- Support for multi-agent Writer–Critic refinement loop with adjustable parameters.
- Integration with a FastAPI backend (via REST calls, not shown in this file).

This script focuses on building the frontend interface and collecting input data
to send to the backend generator.
"""

import streamlit as st
import requests
import json
import time

# Page configuration
st.set_page_config(
    page_title="College Application Helper - Multi-Agent", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 College Application Helper - Multi-Agent System")
st.markdown("*Enhanced with AI Writer and Critic Agents for superior quality*")

# Sidebar for system configuration
with st.sidebar:
    # This block sets up the sidebar for configuring system and retrieval settings.
    st.header("⚙️ System Settings")
    
    # System selection: we offer Multi-Agent, Simple, or Auto-Select
    system_choice = st.radio(
        "Generation System:",
        options=["Multi-Agent (Recommended)", "Simple (Fast)", "Auto-Select"],
        help="Multi-Agent uses iterative improvement for higher quality. Simple is faster but basic."
    )
    
    if system_choice == "Multi-Agent (Recommended)" or system_choice == "Auto-Select":
        st.subheader("Multi-Agent Parameters")
        max_iterations = st.slider(
            "Max Iterations", 
            min_value=1, 
            max_value=5, 
            value=3,
            help="Maximum refinement cycles. More iterations = higher quality but slower."
        )
        
        critique_threshold = st.slider(
            "Quality Threshold", 
            min_value=0.5, 
            max_value=1.0, 
            value=0.8, 
            step=0.05,
            help="Quality score threshold for approval. Higher = stricter quality standards."
        )
        
        fallback_enabled = st.checkbox(
            "Enable Fallback", 
            value=True,
            help="Fall back to simple generator if multi-agent fails"
        )
    else:
        max_iterations = 3
        critique_threshold = 0.8
        fallback_enabled = True
    
    st.subheader("Retrieval Settings")
    topk = st.slider("Evidence Chunks", 1, 10, 3)

# Main form
with st.form("gen_form"):
    # This block creates the main form for user input, including profile, resume, and program info.
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("👤 Profile Information")
        name = st.text_input("Name", value="Alice Zhang")
        major = st.text_input("Major/Field", value="Data Science")
        goals = st.text_area(
            "Career Goals", 
            value="Apply ML to real-world analytics and product data science. Lead data-driven initiatives in tech companies.",
            height=100,
            help="Be specific about your career aspirations"
        )
        
        # Contact info
        with st.expander("Contact Information (Optional)"):
            email = st.text_input("Email", value="alice@example.com")
            gpa = st.text_input("GPA", value="3.78")
        
        # Academic background
        courses = st.text_area(
            "Relevant Courses", 
            value="Machine Learning, Deep Learning, Statistical Inference, Data Management, Algorithms",
            help="Comma-separated list of relevant courses"
        )
        skills = st.text_area(
            "Technical Skills", 
            value="Python, SQL, PyTorch, TensorFlow, R, Tableau, AWS, Git",
            help="Comma-separated list of technical skills"
        )
        
        # Work experiences
        st.subheader("💼 Work Experience")
        num_experiences = st.number_input("Number of Experiences", min_value=0, max_value=5, value=2)
        
        experiences = []
        for i in range(int(num_experiences)):
            with st.expander(f"Experience #{i+1}"):
                exp_title = st.text_input(f"Title {i+1}", value=f"Data Science Intern" if i == 0 else f"Research Assistant")
                exp_org = st.text_input(f"Organization {i+1}", value="Tech Company" if i == 0 else "University Lab")
                exp_impact = st.text_area(
                    f"Impact/Achievement {i+1}", 
                    value="Built ML model improving prediction accuracy by 15%" if i == 0 else "Analyzed large datasets and created visualizations for research publication",
                    height=60
                )
                exp_skills = st.text_input(f"Skills Used {i+1}", value="Python, Scikit-learn, Pandas")
                
                if exp_title and exp_org:
                    experiences.append({
                        "title": exp_title,
                        "org": exp_org, 
                        "impact": exp_impact,
                        "skills": [s.strip() for s in exp_skills.split(",") if s.strip()]
                    })
    
    with col2:
        st.subheader("📄 Resume")
        resume_text = st.text_area(
            "Current Resume (plain text)",
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
• Sentiment Analysis System: Built transformer model achieving F1=0.86 on Twitter data
• Customer Churn Analysis: Used SQL and Tableau to identify key churn factors

SKILLS: Python, R, SQL, PyTorch, TensorFlow, AWS, Git, Tableau""",
            help="Your current resume content - this helps inform the generation"
        )
        
        st.subheader("🎯 Target Program")
        program_url = st.text_input(
            "Program URL (optional)", 
            placeholder="https://example.edu/programs/data-science-ms",
            help="We'll automatically extract program information from the URL"
        )
        
        program_text = st.text_area(
            "Program Description", 
            height=200,
            value="""Master of Science in Data Science

Program Mission: Train the next generation of data scientists to solve complex real-world problems using statistical methods, machine learning, and ethical data practices.

Core Curriculum:
- Statistical Inference and Modeling  
- Machine Learning and Deep Learning
- Data Management and Big Data Systems
- Data Visualization and Communication
- Ethics in Data Science
- Capstone Project

The program emphasizes hands-on experience with real datasets, collaboration with industry partners, and development of both technical and communication skills. Graduates are prepared for roles in data science, machine learning engineering, and analytics across various industries.""",
            help="Paste the program description here if no URL provided"
        )
    
    # Generation button
    submitted = st.form_submit_button("🚀 Generate Application Materials", use_container_width=True)

if submitted:
    # This block handles the generation logic after the form is submitted.
    # It builds the profile and payload, sends the API request, and displays results.
    try:
        # Build profile
        profile = {
            "name": name,
            "major": major, 
            "goals": goals,
            "name": name,
            "major": major, 
            "goals": goals,
            "email": email or None,
            "gpa": float(gpa) if gpa and gpa.strip() else None,
            "gpa": float(gpa) if gpa and gpa.strip() else None,
            "courses": [c.strip() for c in courses.split(",") if c.strip()],
            "skills": [s.strip() for s in skills.split(",") if s.strip()],
            "experiences": experiences
        }
        
        # Build payload based on system choice
        payload = {
            "profile": profile,
            "resume_text": resume_text,
            "program_text": program_text.strip() if program_text.strip() else None,
            "program_url": program_url.strip() if program_url.strip() else None,
            "topk": topk
        }
        
        # Add multi-agent specific parameters
        if system_choice == "Multi-Agent (Recommended)":
            payload.update({
                "use_multi_agent": True,
                "max_iterations": max_iterations,
                "critique_threshold": critique_threshold,
                "fallback_on_error": fallback_enabled
            })
        elif system_choice == "Simple (Fast)":
            payload.update({
                "use_multi_agent": False,
                "fallback_on_error": True
            })
        else:  # Auto-Select
            payload.update({
                "use_multi_agent": True,
                "max_iterations": max_iterations, 
                "critique_threshold": critique_threshold,
                "fallback_on_error": True
            })
        
        # Show progress
        progress_placeholder = st.empty()
        with progress_placeholder:
            with st.spinner(f"Generating with {system_choice.split('(')[0].strip()} system..."):
                start_time = time.time()
                
                # Make API call
                resp = requests.post("http://localhost:8000/generate", json=payload, timeout=120)
                
                generation_time = time.time() - start_time
        
        progress_placeholder.empty()
        
        if resp.status_code == 200:
            data = resp.json()
            
            if "error" in data:
                st.error(f"❌ Generation Error: {data['error']}")
            else:
                # Success header with system info
                system_used = data.get("system_used", "unknown")
                st.success(f"✅ Generated successfully using {system_used} system in {generation_time:.1f}s!")
                
                # Show system information
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("System Used", system_used.replace("_", " ").title())
                with col2:
                    if "generation_metadata" in data.get("report", {}):
                        keywords_count = data["report"]["generation_metadata"].get("keywords_extracted", 0)
                        st.metric("Keywords Extracted", keywords_count)
                with col3:
                    if system_used == "multi_agent":
                        total_iterations = data.get("report", {}).get("overall_quality", {}).get("total_iterations", 0)
                        st.metric("Total Iterations", total_iterations)
                
                # Quality overview for multi-agent
                if system_used == "multi_agent" and "overall_quality" in data.get("report", {}):
                    quality_data = data["report"]["overall_quality"]
                    
                    st.subheader("📊 Quality Overview")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_score = quality_data.get("average_score", 0)
                        st.metric(
                            "Average Quality Score", 
                            f"{avg_score:.2f}",
                            delta=f"{avg_score - 0.5:.2f}" if avg_score > 0.5 else None
                        )
                    
                    with col2:
                        all_approved = quality_data.get("all_approved", False)
                        st.metric("All Content Approved", "✅ Yes" if all_approved else "⚠️ Partial")
                    
                    with col3:
                        total_iterations = quality_data.get("total_iterations", 0)
                        st.metric("Refinement Cycles Used", total_iterations)
                
                # Content tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📝 Personal Statement", "📋 Resume Bullets", "📨 Recommendation", "📊 Detailed Report"])
                
                with tab1:
                    st.subheader("Personal Statement")
                    if system_used == "multi_agent":
                        ps_report = data.get("report", {}).get("content_reports", {}).get("personal_statement", {})
                        if ps_report:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.info(f"Quality Score: {ps_report.get('final_score', 0):.2f} | "
                                       f"Iterations: {ps_report.get('iterations_completed', 0)} | "
                                       f"Status: {'✅ Approved' if ps_report.get('final_approved') else '⚠️ Needs work'}")
                            with col2:
                                if ps_report.get("final_suggestions"):
                                    with st.expander("💡 Suggestions"):
                                        for suggestion in ps_report["final_suggestions"]:
                                            st.write(f"• {suggestion}")
                    
                    st.code(data["texts"]["personal_statement"], language="markdown")
                
                with tab2:
                    st.subheader("Resume Bullets")
                    if system_used == "multi_agent":
                        bullets_report = data.get("report", {}).get("content_reports", {}).get("resume_bullets", {})
                        if bullets_report:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.info(f"Quality Score: {bullets_report.get('final_score', 0):.2f} | "
                                       f"Iterations: {bullets_report.get('iterations_completed', 0)} | "
                                       f"Status: {'✅ Approved' if bullets_report.get('final_approved') else '⚠️ Needs work'}")
                            with col2:
                                if bullets_report.get("final_suggestions"):
                                    with st.expander("💡 Suggestions"):
                                        for suggestion in bullets_report["final_suggestions"]:
                                            st.write(f"• {suggestion}")
                    
                    st.code(data["texts"]["resume_bullets"], language="markdown")
                
                with tab3:
                    st.subheader("Recommendation Letter Template")
                    if system_used == "multi_agent":
                        reco_report = data.get("report", {}).get("content_reports", {}).get("recommendation", {})
                        if reco_report:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.info(f"Quality Score: {reco_report.get('final_score', 0):.2f} | "
                                       f"Iterations: {reco_report.get('iterations_completed', 0)} | "
                                       f"Status: {'✅ Approved' if reco_report.get('final_approved') else '⚠️ Needs work'}")
                            with col2:
                                if reco_report.get("final_suggestions"):
                                    with st.expander("💡 Suggestions"):
                                        for suggestion in reco_report["final_suggestions"]:
                                            st.write(f"• {suggestion}")
                    
                    st.code(data["texts"]["reco_template"], language="markdown")
                
                with tab4:
                    st.subheader("Detailed Generation Report")
                    st.json(data["report"])
                    
                    if data.get("evidence_ids"):
                        st.caption(f"Evidence Sources Used: {', '.join(data['evidence_ids'])}")
                
                # Download section
                st.subheader("💾 Download Generated Content")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        "📝 Personal Statement",
                        data["texts"]["personal_statement"].encode("utf-8"),
                        file_name="personal_statement.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                
                with col2:
                    st.download_button(
                        "📋 Resume Bullets", 
                        data["texts"]["resume_bullets"].encode("utf-8"),
                        file_name="resume_bullets.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                
                with col3:
                    st.download_button(
                        "📨 Recommendation Template",
                        data["texts"]["reco_template"].encode("utf-8"), 
                        file_name="recommendation_template.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                
                # Additional download for report
                st.download_button(
                    "📊 Full Report (JSON)",
                    json.dumps(data["report"], indent=2).encode("utf-8"),
                    file_name="generation_report.json",
                    mime="application/json"
                )
                
                # Warnings and recommendations
                if "generation_metadata" in data.get("report", {}):
                    metadata = data["report"]["generation_metadata"]
                    
                    if metadata.get("profile_warnings"):
                        st.warning("⚠️ Profile Recommendations:")
                        for warning in metadata["profile_warnings"]:
                            st.write(f"• {warning}")
                    
                    if metadata.get("generation_error") and system_used.endswith("_fallback"):
                        st.warning(f"ℹ️ Note: Fell back to simple generator due to: {metadata['generation_error']}")
        
        else:
            st.error(f"❌ API Error {resp.status_code}: {resp.text}")
            
    except ValueError as ve:
        # Handles input errors such as invalid GPA.
        st.error(f"❌ Input Error: {ve}")
    except requests.exceptions.Timeout:
        # Handles request timeout errors.
        st.error("❌ Request timed out. Try reducing max_iterations or using Simple system.")
    except requests.exceptions.ConnectionError:
        # Handles connection errors to the API server.
        st.error("❌ Cannot connect to API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        # Handles any other unexpected errors.
        st.error(f"❌ Unexpected Error: {e}")

# Footer with additional information
# This block displays additional information about the multi-agent system and tips for best results.
st.markdown("---")
with st.expander("ℹ️ About the Multi-Agent System"):
    st.markdown("""
    **Multi-Agent System Features:**
    - **Writer Agent**: Generates and refines content based on your profile
    - **Critic Agent**: Evaluates content quality and provides detailed feedback
    - **Iterative Improvement**: Content goes through multiple refinement cycles
    - **Quality Scoring**: Each piece of content receives a quality score (0-1)
    - **Automatic Fallback**: Falls back to simple generator if needed
    
    **Quality Criteria:**
    - **Personal Statement**: Keyword coverage, structure, specificity, program alignment
    - **Resume Bullets**: Quantification, action words, keyword integration
    - **Recommendation**: Specific examples, enthusiasm level, comprehensiveness
    
    **Tips for Best Results:**
    - Provide detailed work experiences with specific impacts
    - Include relevant technical skills and coursework
    - Be specific in your career goals
    - Use higher quality thresholds for important applications
    """)

st.caption("🔧 Multi-Agent College Application Helper v2.0 | Built with Streamlit")