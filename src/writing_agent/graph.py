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
    
    Decision logic:
    1. If is_complete=True -> go to finalize (END)
    2. If should_revise=True -> go to revise for improvement
    3. Otherwise -> go to finalize (END)
    
    Returns:
        - "revise" if should revise (quality below threshold, iterations remaining)
        - "end" if generation is complete (quality met or max iterations reached)
    """
    # Log for debugging
    print(f"[should_continue] is_complete={state.get('is_complete')}, "
          f"should_revise={state.get('should_revise')}, "
          f"iteration={state.get('current_iteration')}, "
          f"score={state.get('overall_quality_score')}")
    
    if state.get("is_complete", False):
        print("[should_continue] -> end (is_complete=True)")
        return "end"
    
    if state.get("should_revise", False):
        print("[should_continue] -> revise")
        return "revise"
    
    # Default to end if state is unclear
    print("[should_continue] -> end (default)")
    return "end"


def finalize_output(state: WritingState) -> Dict[str, Any]:
    """
    Finalize the output and update memory with enhanced pattern recording.
    
    Args:
        state: Current workflow state
    
    Returns:
        Updated state with finalized document
    """
    current_draft = state.get("current_draft", "")
    overall_score = state.get("overall_quality_score", 0.0)
    iteration_logs = state.get("iteration_logs", [])
    quality_threshold = state.get("quality_threshold", 0.85)
    
    # Set final document
    final_document = current_draft
    
    # Determine if approved based on threshold (not LLM's approve field)
    approved = overall_score >= quality_threshold
    
    print(f"[Finalize] Final score: {overall_score:.3f}, threshold: {quality_threshold}, approved: {approved}")
    print(f"[Finalize] Total iterations: {len(iteration_logs)}")
    
    # Get dimension scores and analysis
    dimension_scores = state.get("dimension_scores", {})
    keyword_analysis = state.get("keyword_analysis", {})
    weakest_dimensions = state.get("weakest_dimensions", [])
    
    # Create quality report with full iteration history
    quality_report = {
        "final_score": overall_score,
        "quality_threshold": quality_threshold,
        "total_iterations": len(iteration_logs),
        "iteration_history": iteration_logs,
        "match_score": state.get("match_score"),
        "dimension_scores": dimension_scores,
        "keyword_analysis": keyword_analysis,
        "keyword_coverage": keyword_analysis.get("overall_integration_quality", 0),
        "approved": approved,
        "met_threshold": approved,
        "final_reflection_feedback": state.get("reflection_feedback", ""),
        "weakest_dimensions": weakest_dimensions
    }
    
    # Update memory with enhanced pattern recording
    if len(iteration_logs) > 1:
        initial_score = iteration_logs[0]["overall_score"]
        final_score = iteration_logs[-1]["overall_score"]
        
        quality_report["score_improvement"] = final_score - initial_score
        
        # Calculate dimension-level improvements
        initial_dim_scores = iteration_logs[0].get("dimension_scores", {})
        final_dim_scores = iteration_logs[-1].get("dimension_scores", {})
        dimension_improvements = {}
        
        for dim in final_dim_scores:
            if dim in initial_dim_scores:
                dimension_improvements[dim] = final_dim_scores[dim] - initial_dim_scores[dim]
        
        quality_report["dimension_improvements"] = dimension_improvements
        
        if final_score > initial_score:
            memory = get_memory()
            
            # Extract revision focus from iteration logs
            revision_focus = []
            strategies_used = []
            for log in iteration_logs:
                strategies_used.extend(log.get("actions_taken", []))
                if "revision_focus" in log:
                    revision_focus.extend(log["revision_focus"].get("weakest_dimensions", []))
            
            # Record success with enhanced information
            memory.record_success(
                document_type=state["document_type"].value,
                initial_score=initial_score,
                final_score=final_score,
                iterations=len(iteration_logs),
                strategies_used=strategies_used,
                dimension_improvements=dimension_improvements,
                revision_focus=list(set(revision_focus))
            )
            
            # Record iteration-level results for detailed learning
            for i in range(1, len(iteration_logs)):
                prev_log = iteration_logs[i-1]
                curr_log = iteration_logs[i]
                
                prev_dim = prev_log.get("dimension_scores", {})
                curr_dim = curr_log.get("dimension_scores", {})
                dim_changes = {d: curr_dim.get(d, 0) - prev_dim.get(d, 0) for d in curr_dim}
                
                memory.record_iteration_result(
                    document_type=state["document_type"].value,
                    iteration=i,
                    score_before=prev_log["overall_score"],
                    score_after=curr_log["overall_score"],
                    strategies_applied=curr_log.get("actions_taken", []),
                    dimension_changes=dim_changes
                )
            
            # Save memory to disk
            try:
                memory.save_to_disk()
                print(f"[Finalize] Memory saved. Stats: {memory.get_stats()}")
            except Exception as e:
                print(f"Warning: Failed to save memory: {e}")
    
    # Update generation metadata
    generation_metadata = state.get("generation_metadata", {})
    generation_metadata["finalized"] = True
    generation_metadata["quality_score"] = overall_score
    generation_metadata["approved"] = approved
    
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
