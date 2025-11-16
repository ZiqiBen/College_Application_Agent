"""
Match Score Calculator Tool

Calculates the alignment score between applicant profile and program requirements.
"""

from typing import Dict, Any, List
from langchain.tools import tool


@tool
def calculate_match_score(
    profile: Dict[str, Any],
    program_info: Dict[str, Any],
    program_keywords: List[str]
) -> Dict[str, Any]:
    """
    Calculate match score between applicant profile and target program.
    
    Returns a dictionary with:
    - overall_score: float (0-1)
    - dimension_scores: dict of individual dimension scores
    - strengths: list of strong matching areas
    - gaps: list of areas needing emphasis
    """
    
    # Extract profile data
    profile_skills = set([s.lower() for s in profile.get("skills", [])])
    profile_courses = set([c.lower() for c in profile.get("courses", [])])
    profile_major = profile.get("major", "").lower()
    
    # Extract program data
    program_keywords_set = set([k.lower() for k in program_keywords])
    program_courses = set()
    
    # Extract courses from program_info
    if isinstance(program_info.get("courses"), list):
        for course in program_info.get("courses", []):
            if isinstance(course, dict):
                program_courses.add(course.get("name", "").lower())
            else:
                program_courses.add(str(course).lower())
    
    # Calculate dimension scores
    
    # 1. Skills Match (30%)
    skills_overlap = len(profile_skills & program_keywords_set)
    skills_score = min(skills_overlap / max(len(program_keywords_set) * 0.3, 1), 1.0)
    
    # 2. Academic Background Match (25%)
    courses_overlap = len(profile_courses & program_courses) if program_courses else 0
    academic_score = min(courses_overlap / max(len(program_courses) * 0.4, 1), 1.0) if program_courses else 0.7
    
    # 3. Experience Relevance (25%)
    experiences = profile.get("experiences", [])
    relevant_exp_count = 0
    for exp in experiences:
        exp_skills = set([s.lower() for s in exp.get("skills", [])])
        if exp_skills & program_keywords_set:
            relevant_exp_count += 1
    experience_score = min(relevant_exp_count / max(len(experiences), 1), 1.0) if experiences else 0.5
    
    # 4. Goal Alignment (20%)
    goals = profile.get("goals", "").lower()
    goal_keyword_matches = sum(1 for kw in program_keywords_set if kw in goals)
    goal_score = min(goal_keyword_matches / max(len(program_keywords_set) * 0.2, 1), 1.0)
    
    # Calculate weighted overall score
    overall_score = (
        skills_score * 0.30 +
        academic_score * 0.25 +
        experience_score * 0.25 +
        goal_score * 0.20
    )
    
    # Identify strengths and gaps
    strengths = []
    gaps = []
    
    if skills_score >= 0.7:
        strengths.append("Strong technical skills alignment")
    elif skills_score < 0.4:
        gaps.append("Need to emphasize technical skills more")
    
    if academic_score >= 0.7:
        strengths.append("Solid academic preparation")
    elif academic_score < 0.4:
        gaps.append("Should highlight relevant coursework")
    
    if experience_score >= 0.7:
        strengths.append("Relevant work experience")
    elif experience_score < 0.4:
        gaps.append("Need to connect experiences to program focus")
    
    if goal_score >= 0.7:
        strengths.append("Clear goal alignment with program")
    elif goal_score < 0.4:
        gaps.append("Goals should better align with program mission")
    
    return {
        "overall_score": round(overall_score, 3),
        "dimension_scores": {
            "skills": round(skills_score, 3),
            "academic": round(academic_score, 3),
            "experience": round(experience_score, 3),
            "goals": round(goal_score, 3)
        },
        "strengths": strengths,
        "gaps": gaps
    }


def calculate_match_score_direct(
    profile: Dict[str, Any],
    program_info: Dict[str, Any],
    program_keywords: List[str]
) -> Dict[str, Any]:
    """
    Direct function call version (non-tool) for internal use
    """
    return calculate_match_score.invoke({
        "profile": profile,
        "program_info": program_info,
        "program_keywords": program_keywords
    })
