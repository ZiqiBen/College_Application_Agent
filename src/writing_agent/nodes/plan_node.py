"""
Plan Node - ReWOO style planning for document generation
"""

from typing import Dict, Any
from ..state import WritingState, DocumentType


def plan_node(state: WritingState) -> Dict[str, Any]:
    """
    Plan the document generation strategy based on document type and profile.
    
    This node implements the "Plan" phase of ReWOO (Plan-Tool-Solve).
    It analyzes the task and creates a generation strategy.
    """
    
    document_type = state["document_type"]
    profile = state["profile"]
    program_info = state["program_info"]
    
    # Create generation plan based on document type
    if document_type == DocumentType.PERSONAL_STATEMENT:
        plan = create_ps_plan(profile, program_info)
    elif document_type == DocumentType.RESUME_BULLETS:
        plan = create_resume_plan(profile, program_info)
    else:  # RECOMMENDATION_LETTER
        plan = create_rl_plan(profile, program_info)
    
    return {
        "plan": plan,
        "generation_metadata": {
            **state.get("generation_metadata", {}),
            "plan_created": True,
            "document_type": document_type
        }
    }


def create_ps_plan(profile: Dict[str, Any], program_info: Dict[str, Any]) -> str:
    """Create plan for Personal Statement generation"""
    
    has_experiences = len(profile.get("experiences", [])) > 0
    has_clear_goals = len(profile.get("goals", "")) > 50
    
    plan = """
PERSONAL STATEMENT GENERATION PLAN:

Phase 1 - Information Gathering:
1. Retrieve relevant program details (mission, courses, features) via RAG
2. Extract program keywords and requirements
3. Calculate profile-program match score
4. Identify top 2-3 relevant experiences from profile

Phase 2 - Content Generation:
1. Opening: Create engaging hook based on strongest experience or insight
2. Background: Showcase 2-3 specific experiences with quantifiable outcomes
3. Program Alignment: Connect profile strengths to specific program features
4. Goals: Articulate clear career vision aligned with program mission
5. Conclusion: Confident statement of fit and contribution

Phase 3 - Quality Enhancement:
1. Integrate required keywords naturally throughout
2. Ensure specific program references (courses, faculty, research)
3. Add quantifiable metrics where possible
4. Maintain coherent narrative arc
5. Target length: 500-800 words
"""
    
    if not has_experiences:
        plan += "\nNOTE: Limited work experience - emphasize academic projects and coursework\n"
    
    if not has_clear_goals:
        plan += "\nNOTE: Goals not detailed - infer from major and program focus\n"
    
    return plan


def create_resume_plan(profile: Dict[str, Any], program_info: Dict[str, Any]) -> str:
    """Create plan for Resume Bullets generation"""
    
    exp_count = len(profile.get("experiences", []))
    
    plan = f"""
RESUME BULLETS GENERATION PLAN:

Phase 1 - Information Gathering:
1. Extract program keywords (technical skills, methodologies)
2. Identify relevant experiences ({exp_count} available)
3. Find experiences with quantifiable achievements

Phase 2 - Bullet Creation:
1. For each experience, create 1-2 bullets following formula:
   [Strong Action Verb] + [What] + [How/Tools] + [Measurable Result]
2. Start with most impactful/relevant experiences
3. Integrate program keywords naturally
4. Emphasize technical specificity

Phase 3 - Optimization:
1. Ensure every bullet has quantifiable metrics
2. Verify strong action verbs (Led, Developed, Optimized, etc.)
3. Check keyword coverage
4. Maintain parallel structure
5. Target: 4-6 bullets total
"""
    
    if exp_count < 2:
        plan += "\nNOTE: Limited experiences - may include academic projects as experience\n"
    
    return plan


def create_rl_plan(profile: Dict[str, Any], program_info: Dict[str, Any]) -> str:
    """Create plan for Recommendation Letter generation"""
    
    plan = """
RECOMMENDATION LETTER GENERATION PLAN:

Phase 1 - Context Establishment:
1. Retrieve program details for fit analysis
2. Identify top 2 experiences for concrete examples
3. Extract program keywords for integration

Phase 2 - Letter Structure:
1. Opening: Establish recommender credibility and relationship
2. Technical Excellence: Detail specific technical achievement with metrics
3. Additional Strengths: Highlight communication, leadership, or collaboration
4. Program Fit: Connect candidate strengths to specific program features
5. Closing: Strong, unequivocal recommendation with contact offer

Phase 3 - Enhancement:
1. Ensure 2-3 specific, concrete examples with outcomes
2. Use strong recommendation language ("highest recommendation", "exceptional")
3. Reference specific program elements
4. Maintain credible, professional tone
5. Target length: 400-700 words
"""
    
    return plan
