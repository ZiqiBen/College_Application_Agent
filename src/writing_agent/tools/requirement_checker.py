"""
Requirement Checker Tool

Checks program-specific requirements and extracts must-have elements for documents.
"""

from typing import Dict, Any, List
import re
from langchain.tools import tool


@tool
def check_program_requirements(
    program_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check and extract program-specific requirements.
    
    Returns a dictionary with:
    - word_limit: dict with min/max word counts if specified
    - must_mention: list of topics/elements that must be mentioned
    - special_instructions: list of special requirements
    - format_requirements: dict of formatting requirements
    """
    
    requirements = program_info.get("application_requirements", "")
    features = program_info.get("features", "")
    combined_text = f"{requirements} {features}"
    
    result = {
        "word_limit": {},
        "must_mention": [],
        "special_instructions": [],
        "format_requirements": {}
    }
    
    # Check for word limits
    word_limit_patterns = [
        r'(\d+)\s*[-to]+\s*(\d+)\s*words',
        r'maximum\s*(?:of\s*)?(\d+)\s*words',
        r'minimum\s*(?:of\s*)?(\d+)\s*words',
        r'(\d+)\s*word\s*limit'
    ]
    
    for pattern in word_limit_patterns:
        matches = re.findall(pattern, combined_text.lower())
        if matches:
            if isinstance(matches[0], tuple) and len(matches[0]) == 2:
                result["word_limit"]["min"] = int(matches[0][0])
                result["word_limit"]["max"] = int(matches[0][1])
            else:
                if "maximum" in pattern or "limit" in pattern:
                    result["word_limit"]["max"] = int(matches[0])
                elif "minimum" in pattern:
                    result["word_limit"]["min"] = int(matches[0])
    
    # Check for must-mention topics
    must_mention_keywords = [
        "research interest", "research experience", "career goals",
        "why this program", "why our program", "specific faculty",
        "diversity statement", "leadership experience", "teamwork",
        "contribution", "what you will bring"
    ]
    
    for keyword in must_mention_keywords:
        if keyword in combined_text.lower():
            result["must_mention"].append(keyword)
    
    # Check for special instructions
    special_indicators = [
        "must include", "should address", "required to", "be sure to",
        "make sure", "ensure that", "it is important"
    ]
    
    sentences = re.split(r'[.!?]', combined_text)
    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        for indicator in special_indicators:
            if indicator in sentence_lower and len(sentence_lower) > 20:
                result["special_instructions"].append(sentence.strip())
                break
    
    # Check format requirements
    if "pdf" in combined_text.lower():
        result["format_requirements"]["format"] = "PDF"
    
    if "single space" in combined_text.lower() or "single-space" in combined_text.lower():
        result["format_requirements"]["spacing"] = "single"
    elif "double space" in combined_text.lower() or "double-space" in combined_text.lower():
        result["format_requirements"]["spacing"] = "double"
    
    # Check for font requirements
    font_match = re.search(r'(\d+)\s*pt|point\s+font', combined_text.lower())
    if font_match:
        result["format_requirements"]["font_size"] = font_match.group(1)
    
    return result


def check_program_requirements_direct(
    program_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Direct function call version (non-tool) for internal use
    """
    return check_program_requirements.invoke({
        "program_info": program_info
    })
