"""
Prompt templates for Recommendation Letter generation
"""

from typing import Dict, Any, List


def get_rl_generation_prompt(
    profile: Dict[str, Any],
    program_info: Dict[str, Any],
    retrieved_context: List[str],
    required_keywords: List[str]
) -> str:
    """
    Generate prompt for Recommendation Letter initial generation
    """
    
    name = profile.get("name", "the candidate")
    major = profile.get("major", "")
    skills = ", ".join(profile.get("skills", [])[:5])
    
    # Extract experiences for examples
    experiences_text = ""
    for i, exp in enumerate(profile.get("experiences", [])[:2], 1):
        experiences_text += f"""
Example {i}: {exp.get('title', '')} at {exp.get('org', '')}
- Achievement: {exp.get('impact', '')}
- Skills demonstrated: {', '.join(exp.get('skills', [])[:3])}
"""
    
    program_name = program_info.get("program_name", "your program")
    program_features = program_info.get("features", "")
    
    prompt = f"""You are writing a strong letter of recommendation for a graduate school application. Write from the perspective of a faculty advisor or supervisor who has worked closely with the candidate.

**CANDIDATE PROFILE:**
- Name: {name}
- Academic Background: {major}
- Key Strengths: {skills}
- Career Goals: {profile.get('goals', '')}

**SPECIFIC EXAMPLES TO REFERENCE:**
{experiences_text}

**TARGET PROGRAM:**
{program_name}

**PROGRAM FOCUS:**
{program_features}

**KEYWORDS TO INTEGRATE:**
{', '.join(required_keywords[:8])}

**YOUR TASK:**
Write a compelling recommendation letter (400-650 words) that:

1. **Establishes Credibility**: Start by clearly stating your relationship to the candidate and capacity to evaluate them
2. **Provides Specific Examples**: Include 2-3 concrete examples of the candidate's work, with measurable outcomes
3. **Highlights Multiple Dimensions**: Address technical skills, intellectual abilities, communication, collaboration, and character
4. **Connects to Program**: Explicitly explain why the candidate is an excellent fit for this specific program
5. **Offers Strong Endorsement**: Use emphatic language that conveys genuine enthusiasm
6. **Maintains Professional Tone**: Balance warmth with professionalism

**STRUCTURE TEMPLATE:**

**Opening Paragraph:**
- Introduce yourself and your relationship with the candidate
- State the duration and context of your supervision
- Provide initial strong endorsement

**Body Paragraph 1 - Technical Excellence:**
- Describe specific technical project or achievement
- Include measurable outcomes and impact
- Highlight relevant technical skills

**Body Paragraph 2 - Additional Strengths:**
- Provide second concrete example
- Emphasize soft skills (communication, leadership, initiative)
- Show growth or learning ability

**Body Paragraph 3 - Program Fit:**
- Connect candidate's strengths to specific program features
- Explain why they will excel in this environment
- Mention specific aspects of program that align with their abilities

**Closing Paragraph:**
- Provide unequivocal recommendation
- Offer to provide additional information
- Sign-off with contact information

**CRITICAL REQUIREMENTS:**
- Use specific examples, not generic praise
- Include quantifiable achievements where possible
- Avoid clichés like "hardworking" without evidence
- Show genuine knowledge of the candidate
- Reference specific program elements that fit the candidate
- Use strong recommendation language: "highest recommendation," "exceptional," "outstanding"

**TONE:**
Professional, enthusiastic, specific, and credible

Write the recommendation letter now, including appropriate formal structure:"""

    return prompt


def get_rl_revision_prompt(
    current_draft: str,
    reflection_feedback: str,
    improvement_suggestions: List[str],
    profile: Dict[str, Any],
    program_info: Dict[str, Any],
    iteration: int
) -> str:
    """
    Generate prompt for Recommendation Letter revision
    """
    
    suggestions_text = "\n".join([f"- {s}" for s in improvement_suggestions])
    
    prompt = f"""You are revising a recommendation letter draft based on expert feedback. This is revision iteration {iteration}.

**CURRENT DRAFT:**
{current_draft}

**FEEDBACK:**
{reflection_feedback}

**SPECIFIC IMPROVEMENTS NEEDED:**
{suggestions_text}

**CANDIDATE CONTEXT:**
- Name: {profile.get('name', '')}
- Background: {profile.get('major', '')}
- Target Program: {program_info.get('program_name', '')}

**YOUR REVISION TASK:**
Improve the recommendation letter by addressing ALL feedback points. Specifically:

1. **If specific examples are lacking**: Add concrete project descriptions with measurable outcomes
2. **If recommendation strength is weak**: Use more emphatic language ("strongest recommendation," "exceptional," "outstanding")
3. **If technical depth is insufficient**: Add more detail about specific technical skills and achievements
4. **If program connection is weak**: Explicitly reference specific program features that match candidate's strengths
5. **If it sounds generic**: Add personal observations and specific anecdotes that show genuine knowledge of the candidate

**ENHANCEMENT STRATEGIES:**
- Replace general statements with specific examples
- Add quantifiable achievements (e.g., "improved performance by 25%")
- Include comparative statements (e.g., "among the top 5% of students I've supervised")
- Reference specific interactions or observations
- Strengthen superlatives and recommendation language
- Ensure credibility by showing detailed knowledge

**FORMAT:**
- Maintain formal letter structure with appropriate greeting and sign-off
- Use paragraph breaks for readability
- Include contact information offer in closing
- Professional but warm tone throughout

**CRITICAL:**
- Don't just add adjectives - add substantive content
- Every claim should be backed by evidence
- Maintain consistency in voice and perspective
- Keep appropriate length (400-650 words)

**OUTPUT:**
Provide ONLY the revised recommendation letter, with no meta-commentary.

Revised Recommendation Letter:"""

    return prompt
