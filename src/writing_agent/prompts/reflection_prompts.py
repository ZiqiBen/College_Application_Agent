"""
Prompt templates for Reflection (self-evaluation) node
"""

from typing import Dict, Any, List


def get_reflection_prompt(
    document: str,
    document_type: str,
    profile: Dict[str, Any],
    program_info: Dict[str, Any],
    required_keywords: List[str],
    iteration: int
) -> str:
    """
    Generate prompt for reflection/evaluation of generated document
    """
    
    # Document-specific evaluation criteria
    if document_type == "personal_statement":
        doc_specific_criteria = """
**PERSONAL STATEMENT SPECIFIC CRITERIA:**
1. **Opening Impact**: Does it start with a compelling, specific hook (not generic)?
2. **Narrative Arc**: Is there a clear story connecting past experiences to future goals?
3. **Specificity**: Are there concrete examples with measurable outcomes?
4. **Program Fit**: Are specific program features, courses, or faculty mentioned?
5. **Authenticity**: Does it sound genuine rather than templated?
"""
        min_length, max_length = 500, 800
        
    elif document_type == "resume_bullets":
        doc_specific_criteria = """
**RESUME BULLETS SPECIFIC CRITERIA:**
1. **Action Verbs**: Does each bullet start with a strong action verb?
2. **Quantification**: Does every bullet include specific numbers/metrics?
3. **Impact Focus**: Are results and outcomes emphasized over responsibilities?
4. **Technical Specificity**: Are tools, technologies, and methodologies named?
5. **Parallel Structure**: Do all bullets follow consistent grammatical structure?
"""
        min_length, max_length = 150, 400
        
    else:  # recommendation_letter
        doc_specific_criteria = """
**RECOMMENDATION LETTER SPECIFIC CRITERIA:**
1. **Credibility**: Is the recommender's relationship and capacity clearly established?
2. **Specific Examples**: Are there 2-3 concrete examples with measurable outcomes?
3. **Multiple Dimensions**: Are technical, interpersonal, and character qualities addressed?
4. **Recommendation Strength**: Is the language emphatic and unequivocal?
5. **Program Connection**: Is the fit with the specific program explained?
"""
        min_length, max_length = 400, 700
    
    prompt = f"""You are an expert admissions consultant conducting a detailed quality review of a {document_type.replace('_', ' ')}. This is evaluation iteration {iteration}.

**DOCUMENT TO EVALUATE:**
{document}

**CONTEXT:**
- Applicant Background: {profile.get('major', '')}
- Target Program: {program_info.get('program_name', '')}
- Required Keywords: {', '.join(required_keywords[:10])}

{doc_specific_criteria}

**YOUR EVALUATION TASK:**
Critically evaluate this document across 5 dimensions, assigning scores from 0.0 to 1.0 for each:

**1. KEYWORD COVERAGE (Weight: 20%)**
Score 0.0-1.0: Are required keywords naturally integrated throughout?
- 1.0 = All keywords present and naturally woven in
- 0.7 = Most keywords present but some feel forced
- 0.4 = Missing several keywords or very awkward integration
- 0.0 = Keywords mostly missing or completely forced

**2. PERSONALIZATION (Weight: 25%)**
Score 0.0-1.0: Is the content specific and personalized with concrete examples?
- 1.0 = Highly specific with quantifiable outcomes and unique details
- 0.7 = Some specific examples but room for more detail
- 0.4 = Mostly generic with few specific examples
- 0.0 = Completely generic and templated

**3. COHERENCE (Weight: 20%)**
Score 0.0-1.0: Is the structure logical and well-organized?
- 1.0 = Excellent flow, clear structure, smooth transitions
- 0.7 = Generally coherent with minor flow issues
- 0.4 = Disjointed sections or weak transitions
- 0.0 = Confusing structure or illogical organization

**4. PROGRAM ALIGNMENT (Weight: 20%)**
Score 0.0-1.0: How well does it connect to specific program features?
- 1.0 = Explicitly references specific courses, faculty, or unique features
- 0.7 = General program alignment but lacks specificity
- 0.4 = Weak connection to program
- 0.0 = No clear program connection

**5. PERSUASIVENESS (Weight: 15%)**
Score 0.0-1.0: How compelling and convincing is the overall content?
- 1.0 = Highly compelling with strong narrative and evidence
- 0.7 = Decent but could be more impactful
- 0.4 = Weak impact or unconvincing
- 0.0 = Not persuasive at all

**ADDITIONAL CHECKS:**
- Word count: {len(document.split())} words (target: {min_length}-{max_length})
- Are there any clichés or overused phrases?
- Is the tone appropriate (professional yet personal)?
- Are there any grammatical or structural issues?

**OUTPUT FORMAT (YOU MUST FOLLOW THIS EXACTLY):**

```json
{{
    "scores": {{
        "keyword_coverage": <0.0-1.0>,
        "personalization": <0.0-1.0>,
        "coherence": <0.0-1.0>,
        "program_alignment": <0.0-1.0>,
        "persuasiveness": <0.0-1.0>
    }},
    "overall_score": <calculated weighted average>,
    "feedback": "<2-3 sentences summarizing main strengths and weaknesses>",
    "specific_issues": [
        "<specific issue 1>",
        "<specific issue 2>",
        ...
    ],
    "improvement_suggestions": [
        "<actionable suggestion 1>",
        "<actionable suggestion 2>",
        "<actionable suggestion 3>"
    ],
    "approve": <true if overall_score >= 0.85, false otherwise>
}}
```

**CRITICAL INSTRUCTIONS:**
- Be honest and critical - don't inflate scores
- Provide specific, actionable feedback
- Reference actual content from the document in your feedback
- Each suggestion should be concrete and implementable
- Consider the iteration number - higher iterations should have higher standards

Conduct your evaluation now:"""

    return prompt


def parse_reflection_response(response: str) -> Dict[str, Any]:
    """
    Parse the JSON response from reflection prompt
    
    NOTE: The 'approve' field from the LLM is NOT used for iteration decisions.
    The actual decision is made in reflect_node.py based on quality_threshold.
    """
    import json
    import re
    
    # Try to extract JSON from response
    # Look for content between ```json and ``` or just the JSON object
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON object directly
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Fallback: return default low scores to trigger revision
            return {
                "scores": {
                    "keyword_coverage": 0.5,
                    "personalization": 0.5,
                    "coherence": 0.5,
                    "program_alignment": 0.5,
                    "persuasiveness": 0.5
                },
                "overall_score": 0.5,
                "feedback": "Unable to parse reflection response",
                "specific_issues": ["Parsing error occurred"],
                "improvement_suggestions": ["Review document manually", "Try regenerating content"],
                "approve": False
            }
    
    try:
        result = json.loads(json_str)
        
        # Ensure all required fields exist with proper defaults
        if "scores" not in result:
            result["scores"] = {
                "keyword_coverage": 0.5,
                "personalization": 0.5,
                "coherence": 0.5,
                "program_alignment": 0.5,
                "persuasiveness": 0.5
            }
        
        # Calculate overall score if not provided or if it seems wrong
        weights = {
            "keyword_coverage": 0.20,
            "personalization": 0.25,
            "coherence": 0.20,
            "program_alignment": 0.20,
            "persuasiveness": 0.15
        }
        
        calculated_overall = sum(
            result["scores"].get(dim, 0.5) * weight
            for dim, weight in weights.items()
        )
        
        # Use calculated score if provided score is missing, zero, or seems inconsistent
        if "overall_score" not in result or result["overall_score"] == 0:
            result["overall_score"] = round(calculated_overall, 3)
        elif abs(result["overall_score"] - calculated_overall) > 0.2:
            # If LLM's overall score deviates too much from calculated, use calculated
            print(f"[parse_reflection] LLM overall_score ({result['overall_score']}) differs from calculated ({calculated_overall:.3f}), using calculated")
            result["overall_score"] = round(calculated_overall, 3)
        else:
            result["overall_score"] = round(result["overall_score"], 3)
        
        # Ensure improvement_suggestions is a non-empty list
        if "improvement_suggestions" not in result or not result["improvement_suggestions"]:
            result["improvement_suggestions"] = ["General improvement needed"]
        
        # Ensure feedback exists
        if "feedback" not in result:
            result["feedback"] = "Evaluation completed"
        
        # Ensure specific_issues exists
        if "specific_issues" not in result:
            result["specific_issues"] = []
        
        # NOTE: The approve field is calculated but NOT used for iteration decisions
        # The actual decision is based on quality_threshold in reflect_node.py
        # We still calculate it for logging/reporting purposes
        if "approve" not in result:
            result["approve"] = result["overall_score"] >= 0.85
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"[parse_reflection] JSON decode error: {e}")
        # Fallback if JSON parsing fails - return low scores to trigger revision
        return {
            "scores": {
                "keyword_coverage": 0.5,
                "personalization": 0.5,
                "coherence": 0.5,
                "program_alignment": 0.5,
                "persuasiveness": 0.5
            },
            "overall_score": 0.5,
            "feedback": "JSON parsing failed - content may need revision",
            "specific_issues": ["Could not parse evaluation response"],
            "improvement_suggestions": ["Manual review needed", "Try regenerating"],
            "approve": False
        }
