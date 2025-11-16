"""
Main LangGraph workflow for Writing Agent
"""

from typing import Dict, Any, Literal, Optional
from langgraph.graph import StateGraph, END
from .state import WritingState, DocumentType, create_initial_state
from .nodes import plan_node, rag_node, react_node, reflect_node, revise_node
from .config import WritingAgentConfig
from .memory import get_memory


def should_continue(state: WritingState) -> str:
    """
    Conditional edge function to determine next node.
    
    Returns:
        - "revise" if should revise
        - "end" if generation is complete
    """
    if state.get("is_complete", False):
        return "end"
    
    if state.get("should_revise", False):
        return "revise"
    
    if state.get("should_continue", True):
        return "reflect"
    
    return "end"


def finalize_output(state: WritingState) -> Dict[str, Any]:
    """
    Finalize the output and update memory.
    
    Args:
        state: Current workflow state
    
    Returns:
        Updated state with finalized document
    """
    current_draft = state.get("current_draft", "")
    overall_score = state.get("overall_quality_score", 0.0)
    iteration_logs = state.get("iteration_logs", [])
    
    # Set final document
    final_document = current_draft
    
    # Create quality report
    quality_report = {
        "final_score": overall_score,
        "total_iterations": len(iteration_logs),
        "iteration_history": iteration_logs,
        "match_score": state.get("match_score"),
        "keyword_coverage": len([
            s for s in state.get("reflection_scores", [])
            if s.get("dimension") == "keyword_coverage"
        ]),
        "approved": overall_score >= state.get("quality_threshold", 0.85)
    }
    
    # Update memory if improvement was significant
    if len(iteration_logs) > 1:
        initial_score = iteration_logs[0]["overall_score"]
        final_score = iteration_logs[-1]["overall_score"]
        
        if final_score > initial_score:
            memory = get_memory()
            memory.record_success(
                document_type=state["document_type"].value,
                initial_score=initial_score,
                final_score=final_score,
                iterations=len(iteration_logs),
                strategies_used=[log.get("actions_taken", []) for log in iteration_logs]
            )
            
            # Save memory to disk
            try:
                memory.save_to_disk()
            except Exception as e:
                print(f"Warning: Failed to save memory: {e}")
    
    # Update generation metadata
    generation_metadata = state.get("generation_metadata", {})
    generation_metadata["finalized"] = True
    generation_metadata["quality_score"] = overall_score
    
    return {
        "final_document": final_document,
        "quality_report": quality_report,
        "is_complete": True,
        "generation_metadata": generation_metadata
    }


def create_writing_graph(
    llm_provider: Literal["openai", "anthropic", "qwen"] = "openai",
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_iterations: int = 3,
    quality_threshold: float = 0.85
) -> StateGraph:
    """
    Create the LangGraph workflow for Writing Agent.
    
    Args:
        llm_provider: LLM provider to use
        model_name: Specific model name (optional)
        temperature: Temperature for generation
        max_iterations: Maximum refinement iterations
        quality_threshold: Quality threshold for approval
    
    Returns:
        Compiled StateGraph
    """
    
    # Validate configuration
    WritingAgentConfig.validate_config()
    
    if not model_name:
        model_name = WritingAgentConfig.get_model_name(llm_provider)
    
    # Create the graph
    workflow = StateGraph(WritingState)
    
    # Add nodes
    workflow.add_node("plan", plan_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("generate", react_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("revise", revise_node)
    workflow.add_node("finalize", finalize_output)
    
    # Define edges
    # Start -> Plan -> RAG -> Generate -> Reflect
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "rag")
    workflow.add_edge("rag", "generate")
    workflow.add_edge("generate", "reflect")
    
    # Conditional edge from Reflect
    # - If should revise: go to Revise
    # - If complete: go to Finalize
    # - Otherwise: loop back to Reflect (shouldn't happen)
    workflow.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "revise": "revise",
            "reflect": "reflect",  # Fallback (shouldn't happen)
            "end": "finalize"
        }
    )
    
    # After Revise, go back to Reflect for evaluation
    workflow.add_edge("revise", "reflect")
    
    # Finalize -> END
    workflow.add_edge("finalize", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


def generate_document(
    profile: Dict[str, Any],
    program_info: Dict[str, Any],
    document_type: DocumentType,
    corpus: Optional[Dict[str, str]] = None,
    llm_provider: Literal["openai", "anthropic", "qwen"] = "openai",
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_iterations: int = 3,
    quality_threshold: float = 0.85
) -> Dict[str, Any]:
    """
    High-level function to generate a document.
    
    Args:
        profile: Applicant profile dictionary
        program_info: Program information dictionary
        document_type: Type of document to generate
        corpus: Optional program corpus
        llm_provider: LLM provider to use
        model_name: Specific model name (optional)
        temperature: Temperature for generation
        max_iterations: Maximum refinement iterations
        quality_threshold: Quality threshold for approval
    
    Returns:
        Dictionary with final_document and quality_report
    """
    
    # Create initial state
    initial_state = create_initial_state(
        profile=profile,
        program_info=program_info,
        document_type=document_type,
        corpus=corpus,
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
        llm_provider=llm_provider,
        model_name=model_name or WritingAgentConfig.get_model_name(llm_provider),
        temperature=temperature
    )
    
    # Create and run graph
    graph = create_writing_graph(
        llm_provider=llm_provider,
        model_name=model_name,
        temperature=temperature,
        max_iterations=max_iterations,
        quality_threshold=quality_threshold
    )
    
    # Execute the workflow
    final_state = graph.invoke(initial_state)
    
    # Extract results
    return {
        "final_document": final_state.get("final_document", ""),
        "quality_report": final_state.get("quality_report", {}),
        "metadata": final_state.get("generation_metadata", {}),
        "iterations": final_state.get("current_iteration", 0),
        "draft_history": final_state.get("draft_history", [])
    }
