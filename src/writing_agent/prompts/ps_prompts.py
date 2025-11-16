"""
Prompt templates for Personal Statement generation
"""

from typing import Dict, Any, List


def get_ps_generation_prompt(
    profile: Dict[str, Any],
    program_info: Dict[str, Any],
    retrieved_context: List[str],
    match_analysis: Dict[str, Any],
    required_keywords: List[str]
) -> str:
    """
    Generate prompt for Personal Statement initial generation
    """
    
    # Extract key information
    name = profile.get("name", "the applicant")
    major = profile.get("major", "")
    goals = profile.get("goals", "")
    skills = ", ".join(profile.get("skills", [])[:5])
    
    # Extract program information
    program_name = program_info.get("program_name", "this program")
    program_features = program_info.get("features", "")
    
    # Format experiences
    experiences_text = ""
    for i, exp in enumerate(profile.get("experiences", [])[:3], 1):
        experiences_text += f"{i}. {exp.get('title', '')} at {exp.get('org', '')}: {exp.get('impact', '')}\n"
    
    # Format context
    context_text = "\n\n".join(retrieved_context[:3])
    
    # Format match analysis
    strengths = ", ".join(match_analysis.get("strengths", []))
    gaps = ", ".join(match_analysis.get("gaps", []))
    
    prompt = f"""You are an expert admissions consultant helping write a compelling Personal Statement for graduate school applications.

**APPLICANT PROFILE:**
- Name: {name}
- Background: {major}
- Key Skills: {skills}
- Career Goals: {goals}

**KEY EXPERIENCES:**
{experiences_text}

**TARGET PROGRAM:**
{program_name}

**PROGRAM DETAILS:**
{program_features}

**PROGRAM CONTEXT (from official sources):**
{context_text}

**MATCH ANALYSIS:**
- Strengths to emphasize: {strengths}
- Areas needing attention: {gaps}
- Match Score: {match_analysis.get('overall_score', 0):.2f}/1.0

**REQUIRED KEYWORDS TO NATURALLY INTEGRATE:**
{', '.join(required_keywords[:10])}

**YOUR TASK:**
Write a compelling, personalized Personal Statement (500-800 words) that:

1. **Opens with Impact**: Start with a specific, engaging anecdote or insight that demonstrates genuine passion
2. **Showcases Specific Experiences**: Use concrete examples with measurable outcomes from the applicant's background
3. **Demonstrates Program Fit**: Explicitly connect the applicant's background and goals to specific program features, courses, or faculty
4. **Integrates Keywords Naturally**: Weave in required keywords organically without forcing them
5. **Shows Forward Vision**: Articulate clear, realistic career goals that align with the program's mission
6. **Maintains Authenticity**: Write in a professional yet personal voice that sounds genuine

**STRUCTURE GUIDELINES:**
- Paragraph 1: Compelling opening with specific motivation
- Paragraph 2-3: Concrete experiences with quantifiable achievements
- Paragraph 4: Explicit program alignment with specific references
- Paragraph 5: Future goals and contribution to the program
- Paragraph 6: Strong, confident conclusion

**CRITICAL REQUIREMENTS:**
- Use specific numbers, metrics, and outcomes from experiences
- Reference specific program courses, research areas, or unique features
- Avoid clichés like "ever since I was young" or "passionate about helping people"
- Show don't tell: demonstrate qualities through examples rather than claiming them
- Maintain academic professionalism while showing personality

Write the Personal Statement now:"""

    return prompt


def get_ps_revision_prompt(
    current_draft: str,
    reflection_feedback: str,
    improvement_suggestions: List[str],
    profile: Dict[str, Any],
    program_info: Dict[str, Any],
    iteration: int
) -> str:
    """
    Generate prompt for Personal Statement revision
    """
    
    suggestions_text = "\n".join([f"- {s}" for s in improvement_suggestions])
    
    prompt = f"""You are revising a Personal Statement draft based on expert feedback. This is revision iteration {iteration}.

**CURRENT DRAFT:**
{current_draft}

**FEEDBACK FROM EXPERT REVIEW:**
{reflection_feedback}

**SPECIFIC IMPROVEMENTS NEEDED:**
{suggestions_text}

**APPLICANT CONTEXT (for reference):**
- Background: {profile.get('major', '')}
- Goals: {profile.get('goals', '')}
- Target Program: {program_info.get('program_name', '')}

**YOUR REVISION TASK:**
Improve the Personal Statement by addressing ALL the feedback points above. Specifically:

1. **If keyword coverage is low**: Naturally integrate missing keywords into existing content
2. **If personalization is weak**: Add more specific details, numbers, and concrete examples
3. **If coherence needs work**: Improve transitions and logical flow between paragraphs
4. **If program alignment is weak**: Add explicit references to specific program features, courses, or faculty
5. **If persuasiveness is lacking**: Strengthen the narrative arc and make achievements more compelling

**CRITICAL GUIDELINES:**
- Maintain the core narrative and voice from the original
- Don't just add keywords - integrate them meaningfully
- Use specific examples and quantifiable outcomes
- Ensure every change serves the overall narrative
- Keep the length appropriate (500-800 words)
- Preserve what's already working well

**OUTPUT:**
Provide ONLY the revised Personal Statement text, with no meta-commentary or explanations.

Revised Personal Statement:"""

    return prompt
