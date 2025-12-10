"""
Prompt templates for Recommendation Letter generation

Includes adaptive revision strategies based on quality score.
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
    iteration: int,
    current_score: float = 0.0,
    dimension_scores: Dict[str, float] = None,
    weakest_dimensions: List[str] = None
) -> str:
    """
    Generate prompt for Recommendation Letter revision with adaptive strategy.
    
    Args:
        current_draft: Current letter draft
        reflection_feedback: Feedback from reflection
        improvement_suggestions: Specific suggestions
        profile: Applicant profile
        program_info: Target program information
        iteration: Current iteration number
        current_score: Current quality score (0.0 to 1.0)
        dimension_scores: Dictionary of dimension scores
        weakest_dimensions: List of weakest dimensions
    """
    
    suggestions_text = "\n".join([f"- {s}" for s in improvement_suggestions])
    
    # Determine revision strategy based on score
    if current_score >= 0.85:
        revision_mode = "REFINEMENT"
        strategy_guidance = """
**REFINEMENT MODE (Strong Letter)**
This letter is already compelling. Apply subtle enhancements:

1. **Strengthen Superlatives**: Upgrade recommendation language
   - "highly recommend" -> "offer my strongest possible recommendation"
   - "excellent student" -> "among the top 5% of students I've supervised"
   
2. **Vivid Details**: Add one memorable sensory detail or direct quote
   
3. **Program Specificity**: Ensure specific program features are mentioned by name

4. **Closing Power**: Strengthen the final endorsement statement

**CRITICAL: Maintain letter structure and core examples. Polish, don't rebuild.**
"""
    elif current_score >= 0.70:
        revision_mode = "STRENGTHEN"
        strategy_guidance = """
**STRENGTHEN MODE (Good Letter)**
Improve specific weak areas:

1. **Example Enhancement**: Add more quantifiable outcomes to existing examples
2. **Program Connection**: Explicitly connect candidate strengths to program
3. **Credential Clarity**: Ensure recommender's authority is clearly established
4. **Multi-dimensional**: Ensure technical AND soft skills are addressed
"""
    else:
        revision_mode = "COMPREHENSIVE"
        strategy_guidance = """
**COMPREHENSIVE REVISION MODE (Needs Improvement)**
Substantially revise the letter:

1. **Structure Fix**: 
   - Opening: Relationship + strong initial endorsement
   - Body 1: Technical example with metrics
   - Body 2: Soft skills + character example
   - Program Fit: Specific program alignment
   - Closing: Emphatic, unequivocal recommendation

2. **Evidence Upgrade**: Replace generic praise with specific examples
3. **Quantification**: Add numbers, comparisons, rankings where possible
4. **Credibility**: Clearly establish recommender's position and knowledge
5. **Enthusiasm**: Use stronger recommendation language throughout
"""
    
    # Add dimension scores context if available
    dimension_context = ""
    if dimension_scores:
        dimension_context = "\n**DIMENSION SCORES:**\n"
        for dim, score in dimension_scores.items():
            status = "✓" if score >= 0.85 else ("→" if score >= 0.70 else "✗")
            dimension_context += f"- {dim.replace('_', ' ').title()}: {score:.2f} [{status}]\n"
    
    prompt = f"""You are revising a recommendation letter based on expert feedback.
Iteration {iteration}. Current score: {current_score:.2f}/1.0

**REVISION MODE: {revision_mode}**

{strategy_guidance}

**CURRENT DRAFT:**
{current_draft}

**FEEDBACK:**
{reflection_feedback}
{dimension_context}

**SPECIFIC IMPROVEMENTS NEEDED:**
{suggestions_text}

**CANDIDATE CONTEXT:**
- Name: {profile.get('name', '')}
- Background: {profile.get('major', '')}
- Target Program: {program_info.get('program_name', '')}

**FORMAT:**
- Maintain formal letter structure
- Professional but warm tone
- Include contact information offer in closing
- Keep length appropriate (400-650 words)

**CRITICAL:**
- Don't just add adjectives - add substantive content
- Every claim should be backed by evidence
- Maintain consistency in voice and perspective

**OUTPUT:**
Provide ONLY the revised recommendation letter, with no meta-commentary.

Revised Recommendation Letter:"""

    return prompt
