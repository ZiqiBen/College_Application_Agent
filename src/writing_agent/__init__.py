"""
Writing Agent Module - LangGraph-based Application Document Generator

This module provides intelligent generation of application documents
(Personal Statements, Resumes, Recommendation Letters) using:
- RAG (Retrieval-Augmented Generation)
- ReAct (Reasoning + Acting with tools)
- Reflection (Self-evaluation and improvement)
- Reflexion (Memory-enhanced iterative refinement)
- ReWOO (Plan-Tool-Solve workflow)
"""

from .graph import create_writing_graph
from .state import WritingState, DocumentType

__version__ = "2.0.0"
__all__ = ["create_writing_graph", "WritingState", "DocumentType"]
