"""
RAG Node - Retrieval Augmented Generation node

Retrieves relevant information from corpus and profile for context.
"""

from typing import Dict, Any, List
import sys
import os

# Add parent directory to path to import from rag_service
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ..state import WritingState
from ..tools.keyword_extractor import extract_keywords_direct
from ..tools.experience_finder import find_relevant_experiences_direct


def rag_node(state: WritingState) -> Dict[str, Any]:
    """
    RAG (Retrieval-Augmented Generation) node.
    
    Retrieves relevant information from:
    1. Program corpus (if available)
    2. Applicant profile (relevant experiences)
    3. Program information
    """
    
    profile = state["profile"]
    program_info = state["program_info"]
    corpus = state.get("corpus", {})
    
    # Extract program keywords
    program_keywords = extract_keywords_direct(
        program_info=program_info,
        top_k=15,
        include_courses=True
    )
    
    # Retrieve from corpus if available
    retrieved_chunks = []
    retrieved_chunk_ids = []
    
    if corpus:
        # Simple retrieval based on keyword matching
        # In production, this would use vector similarity
        retrieved_chunks, retrieved_chunk_ids = retrieve_from_corpus(
            corpus=corpus,
            keywords=program_keywords,
            top_k=state.get("retrieval_top_k", 5)
        )
    else:
        # If no corpus, use program_info directly
        retrieved_chunks = [
            program_info.get("features", ""),
            program_info.get("application_requirements", ""),
            format_courses(program_info.get("courses", []))
        ]
        retrieved_chunks = [c for c in retrieved_chunks if c]  # Filter empty
        retrieved_chunk_ids = ["program_features", "program_requirements", "program_courses"]
    
    # Find relevant experiences from profile
    matched_experiences = find_relevant_experiences_direct(
        profile=profile,
        program_keywords=program_keywords,
        top_k=3
    )
    
    return {
        "retrieved_chunks": retrieved_chunks[:5],  # Top 5 chunks
        "retrieved_chunk_ids": retrieved_chunk_ids[:5],
        "matched_experiences": matched_experiences,
        "program_keywords": program_keywords
    }


def retrieve_from_corpus(
    corpus: Dict[str, str],
    keywords: List[str],
    top_k: int = 5
) -> tuple:
    """
    Simple keyword-based retrieval from corpus.
    
    In production, this should use vector similarity search.
    """
    
    keyword_set = set([kw.lower() for kw in keywords])
    scored_chunks = []
    
    for chunk_id, chunk_text in corpus.items():
        chunk_lower = chunk_text.lower()
        
        # Calculate keyword overlap score
        score = sum(1 for kw in keyword_set if kw in chunk_lower)
        
        if score > 0:
            scored_chunks.append((chunk_id, chunk_text, score))
    
    # Sort by score
    scored_chunks.sort(key=lambda x: x[2], reverse=True)
    
    # Return top_k
    retrieved_ids = [chunk_id for chunk_id, _, _ in scored_chunks[:top_k]]
    retrieved_texts = [text for _, text, _ in scored_chunks[:top_k]]
    
    return retrieved_texts, retrieved_ids


def format_courses(courses: List[Any]) -> str:
    """Format course list into readable text"""
    
    if not courses:
        return ""
    
    formatted = "Program Courses:\n"
    for course in courses[:10]:  # Limit to 10 courses
        if isinstance(course, dict):
            name = course.get("name", "")
            desc = course.get("description", "")
            if desc:
                formatted += f"- {name}: {desc}\n"
            else:
                formatted += f"- {name}\n"
        else:
            formatted += f"- {str(course)}\n"
    
    return formatted
