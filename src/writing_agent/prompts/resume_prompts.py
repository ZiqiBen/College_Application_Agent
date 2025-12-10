"""
Prompt templates for Resume Bullets generation

Includes adaptive revision strategies based on quality score.
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
    iteration: int,
    current_score: float = 0.0,
    dimension_scores: Dict[str, float] = None,
    weakest_dimensions: List[str] = None
) -> str:
    """
    Generate prompt for Resume Bullets revision with adaptive strategy.
    
    Args:
        current_draft: Current bullet points
        reflection_feedback: Feedback from reflection
        improvement_suggestions: Specific suggestions
        profile: Applicant profile
        required_keywords: Keywords to include
        iteration: Current iteration number
        current_score: Current quality score (0.0 to 1.0)
        dimension_scores: Dictionary of dimension scores
        weakest_dimensions: List of weakest dimensions
    """
    
    suggestions_text = "\n".join([f"- {s}" for s in improvement_suggestions])
    
    # Determine revision strategy based on score
    if current_score >= 0.85:
        revision_mode = "POLISH"
        strategy_guidance = """
**POLISH MODE (High Quality Bullets)**
These bullets are already strong. Make precision enhancements only:

1. **Verb Power-Up**: Upgrade action verbs to most impactful alternatives
   - "Managed" -> "Orchestrated", "Spearheaded"
   - "Created" -> "Architected", "Pioneered"
   
2. **Metric Precision**: Make numbers more impressive while accurate
   - Round to impactful figures (e.g., "47%" -> "nearly 50%")
   - Add context to numbers (e.g., "5x industry average")
   
3. **Technical Depth**: Add one more specific tool or technology name

**DO NOT restructure or significantly change proven bullet formats.**
"""
    elif current_score >= 0.70:
        revision_mode = "ENHANCE"
        strategy_guidance = """
**ENHANCE MODE (Good Bullets)**
Focus on strengthening weak areas while preserving strengths:

1. **Missing Metrics**: Add quantifiable metrics to any bullet lacking numbers
2. **Impact Clarity**: Ensure each bullet shows clear outcome/result
3. **Keyword Gaps**: Integrate any missing target keywords
4. **Tool Specificity**: Name specific technologies, tools, frameworks
"""
    else:
        revision_mode = "REBUILD"
        strategy_guidance = """
**REBUILD MODE (Significant Improvement Needed)**
Substantially revise these bullets:

1. **Structure Fix**: Ensure format: [Action Verb] + [What] + [How/Tools] + [Result]
2. **Quantify Everything**: Add numbers to EVERY bullet (%, $, counts, time)
3. **Impact Focus**: Convert responsibilities into achievements
4. **Parallel Structure**: Make all bullets follow consistent grammar
5. **Technical Upgrade**: Add specific tools, technologies, methodologies
"""
    
    # Add dimension scores context if available
    dimension_context = ""
    if dimension_scores:
        dimension_context = "\n**DIMENSION SCORES:**\n"
        for dim, score in dimension_scores.items():
            status = "✓" if score >= 0.85 else ("→" if score >= 0.70 else "✗")
            dimension_context += f"- {dim.replace('_', ' ').title()}: {score:.2f} [{status}]\n"
    
    prompt = f"""You are revising resume bullet points based on expert feedback.
Iteration {iteration}. Current score: {current_score:.2f}/1.0

**REVISION MODE: {revision_mode}**

{strategy_guidance}

**CURRENT BULLETS:**
{current_draft}

**FEEDBACK:**
{reflection_feedback}
{dimension_context}

**SPECIFIC IMPROVEMENTS NEEDED:**
{suggestions_text}

**TARGET KEYWORDS:**
{', '.join(required_keywords[:10])}

**APPLICANT CONTEXT:**
- Skills: {', '.join(profile.get('skills', [])[:6])}
- Major: {profile.get('major', '')}

**FORMAT REQUIREMENTS:**
- Bullet format with "• " or "- "
- 1-2 lines per bullet
- No periods at the end
- Parallel grammatical structure

**OUTPUT:**
Provide ONLY the revised bullet points, with no explanations or commentary.

Revised Resume Bullets:"""

    return prompt
