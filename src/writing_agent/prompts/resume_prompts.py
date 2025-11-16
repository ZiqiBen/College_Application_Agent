"""
Prompt templates for Resume Bullets generation
"""

from typing import Dict, Any, List


def get_resume_generation_prompt(
    profile: Dict[str, Any],
    required_keywords: List[str],
    match_analysis: Dict[str, Any]
) -> str:
    """
    Generate prompt for Resume Bullets initial generation
    """
    
    # Extract experiences
    experiences_detail = ""
    for i, exp in enumerate(profile.get("experiences", []), 1):
        title = exp.get("title", "")
        org = exp.get("org", "")
        impact = exp.get("impact", "")
        skills = ", ".join(exp.get("skills", []))
        
        experiences_detail += f"""
**Experience {i}:**
- Role: {title}
- Organization: {org}
- Impact/Achievement: {impact}
- Skills Used: {skills}
"""
    
    prompt = f"""You are an expert resume writer specializing in graduate school applications and technical roles.

**APPLICANT BACKGROUND:**
- Major: {profile.get('major', '')}
- Skills: {', '.join(profile.get('skills', [])[:8])}
- GPA: {profile.get('gpa', 'N/A')}

**EXPERIENCES TO HIGHLIGHT:**
{experiences_detail}

**TARGET KEYWORDS TO INTEGRATE:**
{', '.join(required_keywords[:10])}

**MATCH ANALYSIS:**
- Strengths: {', '.join(match_analysis.get('strengths', []))}
- Areas to emphasize: {', '.join(match_analysis.get('gaps', []))}

**YOUR TASK:**
Create 4-6 impactful resume bullet points that:

1. **Start with Strong Action Verbs**: Use powerful verbs like Led, Developed, Implemented, Optimized, Architected, Designed
2. **Quantify Everything**: Include specific numbers, percentages, scale (e.g., "processed 1M+ records daily")
3. **Show Impact**: Focus on outcomes and results, not just responsibilities
4. **Integrate Keywords**: Naturally weave in target keywords that align with the program
5. **Use Technical Specificity**: Mention specific tools, technologies, methodologies
6. **Follow Formula**: [Action Verb] + [What] + [How/With What Tools] + [Measurable Result]

**FORMAT REQUIREMENTS:**
- Each bullet starts with "• " or "- "
- Length: 1-2 lines per bullet
- No periods at the end
- Parallel structure across all bullets

**EXAMPLES OF STRONG BULLETS:**
- Developed machine learning pipeline using Python and TensorFlow, improving prediction accuracy by 23% and reducing processing time by 40%
- Led cross-functional team of 5 to design and deploy real-time analytics dashboard, enabling data-driven decisions across 3 departments
- Optimized SQL queries for customer segmentation analysis, reducing query time from 45 minutes to 3 minutes while processing 10M+ records

**CRITICAL REQUIREMENTS:**
- Every bullet MUST include quantifiable metrics
- Emphasize technical skills and tools explicitly
- Show progression and increasing responsibility if multiple experiences
- Align with target program's focus areas

Generate the resume bullets now:"""

    return prompt


def get_resume_revision_prompt(
    current_draft: str,
    reflection_feedback: str,
    improvement_suggestions: List[str],
    profile: Dict[str, Any],
    required_keywords: List[str],
    iteration: int
) -> str:
    """
    Generate prompt for Resume Bullets revision
    """
    
    suggestions_text = "\n".join([f"- {s}" for s in improvement_suggestions])
    
    prompt = f"""You are revising resume bullet points based on expert feedback. This is revision iteration {iteration}.

**CURRENT BULLETS:**
{current_draft}

**FEEDBACK:**
{reflection_feedback}

**SPECIFIC IMPROVEMENTS NEEDED:**
{suggestions_text}

**TARGET KEYWORDS:**
{', '.join(required_keywords[:10])}

**APPLICANT CONTEXT:**
- Skills: {', '.join(profile.get('skills', [])[:6])}
- Major: {profile.get('major', '')}

**YOUR REVISION TASK:**
Improve the resume bullets by addressing ALL feedback points. Specifically:

1. **If quantification is lacking**: Add specific numbers, percentages, scale to every bullet
2. **If action verbs are weak**: Replace weak verbs (worked, helped, did) with strong ones (Led, Developed, Optimized)
3. **If keywords missing**: Naturally integrate missing keywords into appropriate bullets
4. **If impact unclear**: Emphasize measurable outcomes and business/research impact
5. **If too generic**: Add technical specificity (tools, technologies, methodologies)

**ENHANCEMENT STRATEGIES:**
- Convert responsibilities into achievements
- Add comparative metrics (e.g., "reducing time by 60%" instead of just "faster")
- Specify scale (e.g., "managing dataset of 5M rows")
- Name tools explicitly (e.g., "using Python, Pandas, and Scikit-learn")
- Show progression if revising multiple bullets

**FORMAT:**
- Maintain bullet format with "• " or "- "
- Keep 1-2 lines per bullet
- No periods at the end
- Ensure parallel grammatical structure

**OUTPUT:**
Provide ONLY the revised bullet points, with no explanations or commentary.

Revised Resume Bullets:"""

    return prompt
