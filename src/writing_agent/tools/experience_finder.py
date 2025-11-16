"""
Experience Finder Tool

Finds and ranks relevant experiences from applicant profile based on program keywords.
"""

from typing import Dict, Any, List
from langchain.tools import tool


@tool
def find_relevant_experiences(
    profile: Dict[str, Any],
    program_keywords: List[str],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Find the most relevant experiences from profile based on program keywords.
    
    Args:
        profile: Applicant profile dictionary
        program_keywords: List of program keywords
        top_k: Number of top experiences to return
    
    Returns:
        List of relevant experiences with relevance scores
    """
    
    experiences = profile.get("experiences", [])
    if not experiences:
        return []
    
    program_keywords_set = set([k.lower() for k in program_keywords])
    scored_experiences = []
    
    for exp in experiences:
        score = 0
        matches = []
        
        # Check title for keywords
        title = exp.get("title", "").lower()
        for kw in program_keywords_set:
            if kw in title:
                score += 3
                matches.append(kw)
        
        # Check organization
        org = exp.get("org", "").lower()
        for kw in program_keywords_set:
            if kw in org:
                score += 2
                matches.append(kw)
        
        # Check impact description
        impact = exp.get("impact", "").lower()
        for kw in program_keywords_set:
            if kw in impact:
                score += 2
                matches.append(kw)
        
        # Check skills
        exp_skills = [s.lower() for s in exp.get("skills", [])]
        for skill in exp_skills:
            for kw in program_keywords_set:
                if kw in skill or skill in kw:
                    score += 4
                    matches.append(kw)
        
        scored_experiences.append({
            "experience": exp,
            "relevance_score": score,
            "matched_keywords": list(set(matches))[:5],  # Top 5 unique matches
            "match_count": len(set(matches))
        })
    
    # Sort by relevance score
    scored_experiences.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return scored_experiences[:top_k]


def find_relevant_experiences_direct(
    profile: Dict[str, Any],
    program_keywords: List[str],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Direct function call version (non-tool) for internal use
    """
    return find_relevant_experiences.invoke({
        "profile": profile,
        "program_keywords": program_keywords,
        "top_k": top_k
    })
