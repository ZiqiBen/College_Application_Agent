"""
Prompt templates for Personal Statement generation

Includes adaptive revision strategies:
- High score (>=0.85): Fine-tuning mode with micro-level improvements
- Medium score (0.70-0.85): Targeted improvements on weak dimensions
- Low score (<0.70): Comprehensive revision mode
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
    iteration: int,
    current_score: float = 0.0,
    dimension_scores: Dict[str, float] = None,
    weakest_dimensions: List[str] = None,
    previous_draft: str = None
) -> str:
    """
    Generate prompt for Personal Statement revision with adaptive strategy.
    
    The revision strategy adapts based on current_score:
    - High score (>=0.85): Fine-tuning mode - polish language, strengthen specific phrases
    - Medium score (0.70-0.85): Targeted mode - focus on weakest dimensions
    - Low score (<0.70): Comprehensive mode - structural and content overhaul
    
    Args:
        current_draft: Current version of the PS
        reflection_feedback: Feedback from reflection node
        improvement_suggestions: List of specific suggestions
        profile: Applicant profile
        program_info: Target program information
        iteration: Current iteration number
        current_score: Current quality score (0.0 to 1.0)
        dimension_scores: Dictionary of scores for each dimension
        weakest_dimensions: List of dimensions needing most improvement
        previous_draft: Previous version for comparison (optional)
    """
    
    suggestions_text = "\n".join([f"- {s}" for s in improvement_suggestions])
    
    # Determine revision strategy based on score
    if current_score >= 0.85:
        revision_mode = "FINE-TUNING"
        strategy_guidance = """
**FINE-TUNING MODE (High Quality Draft)**
This draft is already excellent. Make ONLY micro-level improvements:

1. **Word Choice Enhancement**: Replace good words with stronger synonyms
   - "helped" -> "facilitated", "spearheaded", "championed"
   - "worked on" -> "engineered", "orchestrated", "pioneered"
   
2. **Precision Boost**: Make numbers and metrics more specific
   - "many users" -> "12,000+ monthly active users"
   - "improved performance" -> "reduced latency by 47%"
   
3. **Emotional Resonance**: Add subtle sensory or emotional details
   - Include one vivid moment that shows genuine passion
   
4. **Flow Polish**: Ensure paragraph transitions are seamless

**CRITICAL: Do NOT restructure or significantly change content.**
**Preserve the narrative arc and core message entirely.**
"""
    elif current_score >= 0.70:
        revision_mode = "TARGETED"
        # Build targeted guidance based on weakest dimensions
        weak_dims_guidance = ""
        if weakest_dimensions:
            for dim in weakest_dimensions:
                if dim == "keyword_coverage":
                    weak_dims_guidance += """
- **Keyword Integration**: Naturally weave in missing keywords into existing sentences
  (Do not add forced keyword lists; integrate organically)
"""
                elif dim == "personalization":
                    weak_dims_guidance += """
- **Personalization**: Add 1-2 more specific details with quantifiable metrics
  (Example: Instead of "led a project", say "led a 4-person team delivering 3 features in 6 weeks")
"""
                elif dim == "coherence":
                    weak_dims_guidance += """
- **Coherence**: Strengthen transitions between paragraphs with connecting phrases
  (Use phrases like "Building on this experience...", "This foundation prepared me...")
"""
                elif dim == "program_alignment":
                    weak_dims_guidance += """
- **Program Alignment**: Add 1-2 specific references to program courses, faculty, or features
  (Research actual program details and mention them by name)
"""
                elif dim == "persuasiveness":
                    weak_dims_guidance += """
- **Persuasiveness**: Strengthen impact statements with "so what" context
  (Show WHY achievements matter, not just WHAT was done)
"""
        
        strategy_guidance = f"""
**TARGETED IMPROVEMENT MODE (Good Draft)**
This draft is solid but needs improvement in specific areas.

**FOCUS YOUR REVISION ON THESE WEAK DIMENSIONS:**
{weak_dims_guidance if weak_dims_guidance else "- Address all feedback points systematically"}

**PRESERVE**: Keep strong elements intact - don't over-edit what works.
"""
    else:
        revision_mode = "COMPREHENSIVE"
        strategy_guidance = """
**COMPREHENSIVE REVISION MODE (Needs Significant Improvement)**
This draft requires substantial improvements:

1. **Structure Overhaul**: Ensure clear paragraph organization
   - Para 1: Compelling hook + motivation
   - Para 2-3: Specific experiences with metrics
   - Para 4: Program fit with specific references
   - Para 5: Future vision + conclusion

2. **Content Enhancement**:
   - Replace ALL generic statements with specific examples
   - Add quantifiable metrics to EVERY achievement
   - Include specific program references (courses, faculty, research)

3. **Keyword Integration**: Ensure all required keywords appear naturally

4. **Voice & Authenticity**: Remove clichés, add genuine personal perspective

**Be bold in making changes - the draft needs significant improvement.**
"""
    
    # Add dimension scores context if available
    dimension_context = ""
    if dimension_scores:
        dimension_context = "\n**CURRENT DIMENSION SCORES:**\n"
        for dim, score in dimension_scores.items():
            status = "✓ Strong" if score >= 0.85 else ("→ Improve" if score >= 0.70 else "✗ Weak")
            dimension_context += f"- {dim.replace('_', ' ').title()}: {score:.2f} [{status}]\n"
    
    # Add draft comparison if available
    comparison_context = ""
    if previous_draft and iteration > 1:
        comparison_context = f"""
**PREVIOUS VERSION (for reference):**
{previous_draft[:500]}...

**IMPORTANT**: Ensure the revision improves upon the previous version. Do not regress on improvements already made.
"""
    
    prompt = f"""You are revising a Personal Statement draft based on expert feedback. 
This is revision iteration {iteration}. Current quality score: {current_score:.2f}/1.0

**REVISION MODE: {revision_mode}**

{strategy_guidance}

**CURRENT DRAFT:**
{current_draft}

**FEEDBACK FROM EXPERT REVIEW:**
{reflection_feedback}
{dimension_context}

**SPECIFIC IMPROVEMENTS NEEDED:**
{suggestions_text}
{comparison_context}

**APPLICANT CONTEXT (for reference):**
- Background: {profile.get('major', '')}
- Goals: {profile.get('goals', '')}
- Target Program: {program_info.get('program_name', '')}

**CRITICAL GUIDELINES:**
- Maintain the core narrative and voice from the original
- Keep the length appropriate (500-800 words)
- Every change should serve the overall narrative

**OUTPUT:**
Provide ONLY the revised Personal Statement text, with no meta-commentary or explanations.

Revised Personal Statement:"""

    return prompt
