"""
Keyword Extractor Tool

Extracts relevant keywords from program information for document generation.
"""

from typing import Dict, Any, List
import re
from collections import Counter
from langchain.tools import tool


# Common stop words to exclude
STOP_WORDS = {
    "the", "and", "or", "in", "to", "of", "a", "an", "for", "with", "on", 
    "at", "by", "this", "that", "these", "those", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "should", "could", "may", "might", "must", "can", "from", "as",
    "program", "course", "student", "university", "college", "degree"
}

# Technical/domain keywords that should be prioritized
PRIORITY_KEYWORDS = {
    "machine learning", "deep learning", "artificial intelligence", "data science",
    "statistics", "statistical", "analytics", "big data", "nlp", "computer vision",
    "neural networks", "algorithms", "database", "sql", "python", "r", "java",
    "cloud computing", "distributed systems", "software engineering", "research",
    "optimization", "modeling", "visualization", "ethics", "healthcare", "finance"
}


@tool
def extract_keywords(
    program_info: Dict[str, Any],
    top_k: int = 15,
    include_courses: bool = True
) -> List[str]:
    """
    Extract relevant keywords from program information.
    
    Args:
        program_info: Program information dictionary
        top_k: Number of top keywords to return
        include_courses: Whether to include course names as keywords
    
    Returns:
        List of extracted keywords
    """
    
    keywords_counter = Counter()
    
    # Extract from program name
    program_name = program_info.get("program_name", "")
    if program_name:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', program_name.lower())
        for word in words:
            if word not in STOP_WORDS:
                keywords_counter[word] += 3  # Higher weight for program name
    
    # Extract from features/description
    features = program_info.get("features", "")
    if features:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', features.lower())
        for word in words:
            if word not in STOP_WORDS:
                keywords_counter[word] += 2
    
    # Extract from requirements
    requirements = program_info.get("application_requirements", "")
    if requirements:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', requirements.lower())
        for word in words:
            if word not in STOP_WORDS:
                keywords_counter[word] += 1
    
    # Extract from courses
    if include_courses:
        courses = program_info.get("courses", [])
        if isinstance(courses, list):
            for course in courses:
                if isinstance(course, dict):
                    course_name = course.get("name", "")
                    course_desc = course.get("description", "")
                    text = f"{course_name} {course_desc}"
                else:
                    text = str(course)
                
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                for word in words:
                    if word not in STOP_WORDS:
                        keywords_counter[word] += 1.5
    
    # Check for priority multi-word phrases in combined text
    combined_text = " ".join([
        program_name,
        features,
        requirements
    ]).lower()
    
    priority_found = []
    for phrase in PRIORITY_KEYWORDS:
        if phrase in combined_text:
            priority_found.append(phrase)
            # Boost individual words in priority phrases
            for word in phrase.split():
                keywords_counter[word] += 5
    
    # Get top keywords
    top_keywords = [word for word, count in keywords_counter.most_common(top_k * 2)]
    
    # Combine with priority phrases
    result_keywords = list(set(priority_found + top_keywords))[:top_k]
    
    return result_keywords


def extract_keywords_direct(
    program_info: Dict[str, Any],
    top_k: int = 15,
    include_courses: bool = True
) -> List[str]:
    """
    Direct function call version (non-tool) for internal use
    """
    return extract_keywords.invoke({
        "program_info": program_info,
        "top_k": top_k,
        "include_courses": include_courses
    })
