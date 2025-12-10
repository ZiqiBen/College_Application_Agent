"""
Memory management for Reflexion-style learning

Enhanced with:
- Dimension-level pattern tracking
- Successful revision strategy recording
- Issue-resolution pairs for learning
- Score-appropriate pattern matching
"""

from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime
from .config import WritingAgentConfig


class ReflexionMemory:
    """
    Manages memory for Reflexion-style iterative improvement.
    
    Stores:
    - Successful patterns that led to improvements
    - Common issues and how they were resolved
    - Dimension-level improvement strategies
    - Iteration history for learning
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize memory manager.
        
        Args:
            cache_dir: Directory to cache memory (optional)
        """
        self.cache_dir = cache_dir or WritingAgentConfig.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.successful_patterns: List[Dict[str, Any]] = []
        self.common_issues: Dict[str, List[str]] = {}
        self.improvement_strategies: Dict[str, List[str]] = {}
        self.dimension_patterns: Dict[str, List[Dict[str, Any]]] = {}  # New: dimension-level tracking
    
    def record_success(
        self,
        document_type: str,
        initial_score: float,
        final_score: float,
        iterations: int,
        strategies_used: List[str],
        dimension_improvements: Dict[str, float] = None,
        revision_focus: List[str] = None
    ):
        """
        Record a successful generation pattern.
        
        Args:
            document_type: Type of document
            initial_score: Starting quality score
            final_score: Final quality score
            iterations: Number of iterations taken
            strategies_used: List of strategies that worked
            dimension_improvements: Per-dimension score changes
            revision_focus: Dimensions that were focused on
        """
        improvement = final_score - initial_score
        
        if improvement > 0.05:  # Lower threshold to capture more patterns
            pattern = {
                "document_type": document_type,
                "initial_score": initial_score,
                "final_score": final_score,
                "improvement": improvement,
                "iterations": iterations,
                "strategies": strategies_used,
                "dimension_improvements": dimension_improvements or {},
                "revision_focus": revision_focus or [],
                "timestamp": datetime.now().isoformat(),
                "score_range": self._get_score_range(initial_score)
            }
            
            self.successful_patterns.append(pattern)
            
            # Also record dimension-level patterns
            if dimension_improvements:
                for dim, dim_improvement in dimension_improvements.items():
                    if dim_improvement > 0.05:
                        self._record_dimension_pattern(
                            document_type=document_type,
                            dimension=dim,
                            improvement=dim_improvement,
                            strategies=strategies_used,
                            initial_score=initial_score
                        )
            
            # Keep only recent successful patterns
            if len(self.successful_patterns) > 100:
                self.successful_patterns = self.successful_patterns[-100:]
    
    def _get_score_range(self, score: float) -> str:
        """Categorize score into ranges for better pattern matching."""
        if score >= 0.85:
            return "high"
        elif score >= 0.70:
            return "medium"
        else:
            return "low"
    
    def _record_dimension_pattern(
        self,
        document_type: str,
        dimension: str,
        improvement: float,
        strategies: List[str],
        initial_score: float
    ):
        """Record a successful pattern for a specific dimension."""
        key = f"{document_type}:{dimension}"
        
        if key not in self.dimension_patterns:
            self.dimension_patterns[key] = []
        
        pattern = {
            "improvement": improvement,
            "strategies": strategies,
            "initial_score": initial_score,
            "score_range": self._get_score_range(initial_score),
            "timestamp": datetime.now().isoformat()
        }
        
        self.dimension_patterns[key].append(pattern)
        
        # Keep only recent patterns per dimension
        if len(self.dimension_patterns[key]) > 20:
            self.dimension_patterns[key] = self.dimension_patterns[key][-20:]
    
    def record_issue(
        self,
        document_type: str,
        issue_type: str,
        resolution: str
    ):
        """
        Record a common issue and its resolution.
        
        Args:
            document_type: Type of document
            issue_type: Category of issue
            resolution: How it was resolved
        """
        key = f"{document_type}:{issue_type}"
        
        if key not in self.common_issues:
            self.common_issues[key] = []
        
        self.common_issues[key].append(resolution)
        
        # Keep only recent resolutions
        if len(self.common_issues[key]) > 20:
            self.common_issues[key] = self.common_issues[key][-20:]
    
    def get_relevant_patterns(
        self,
        document_type: str,
        current_score: float
    ) -> List[Dict[str, Any]]:
        """
        Get relevant successful patterns for current situation.
        
        Prioritizes patterns with similar initial scores for better applicability.
        
        Args:
            document_type: Type of document being generated
            current_score: Current quality score
        
        Returns:
            List of relevant patterns, sorted by relevance
        """
        relevant = []
        current_range = self._get_score_range(current_score)
        
        for pattern in self.successful_patterns:
            if pattern["document_type"] == document_type:
                # Calculate relevance score
                score_diff = abs(pattern["initial_score"] - current_score)
                same_range = pattern.get("score_range") == current_range
                
                relevance = {
                    **pattern,
                    "_relevance_score": (1.0 - score_diff) + (0.3 if same_range else 0)
                }
                relevant.append(relevance)
        
        # Sort by relevance and improvement
        relevant.sort(
            key=lambda x: (x["_relevance_score"], x["improvement"]), 
            reverse=True
        )
        
        return relevant[:5]  # Top 5 relevant patterns
    
    def get_dimension_strategies(
        self,
        document_type: str,
        dimension: str,
        current_score: float
    ) -> List[str]:
        """
        Get successful strategies for improving a specific dimension.
        
        Args:
            document_type: Type of document
            dimension: Dimension to improve (e.g., "keyword_coverage")
            current_score: Current overall score
        
        Returns:
            List of strategies that have worked for this dimension
        """
        key = f"{document_type}:{dimension}"
        patterns = self.dimension_patterns.get(key, [])
        
        if not patterns:
            return []
        
        # Filter patterns by score range
        current_range = self._get_score_range(current_score)
        relevant_patterns = [p for p in patterns if p.get("score_range") == current_range]
        
        # If no patterns in current range, use all patterns
        if not relevant_patterns:
            relevant_patterns = patterns
        
        # Sort by improvement
        relevant_patterns.sort(key=lambda x: x["improvement"], reverse=True)
        
        # Extract unique strategies
        strategies = []
        seen = set()
        for pattern in relevant_patterns[:5]:
            for strategy in pattern.get("strategies", []):
                if strategy not in seen:
                    seen.add(strategy)
                    strategies.append(strategy)
        
        return strategies[:3]  # Top 3 strategies
    
    def get_issue_resolutions(
        self,
        document_type: str,
        issue_type: str
    ) -> List[str]:
        """
        Get known resolutions for a specific issue type.
        
        Args:
            document_type: Type of document
            issue_type: Type of issue
        
        Returns:
            List of resolution strategies
        """
        key = f"{document_type}:{issue_type}"
        return self.common_issues.get(key, [])
    
    def suggest_strategies(
        self,
        document_type: str,
        current_issues: List[str],
        current_score: float = 0.6,
        weak_dimensions: List[str] = None
    ) -> List[str]:
        """
        Suggest improvement strategies based on current issues and weak dimensions.
        
        Enhanced to consider:
        - Current score range for appropriate strategies
        - Specific weak dimensions
        - Past successful patterns
        
        Args:
            document_type: Type of document
            current_issues: List of current issues
            current_score: Current overall score
            weak_dimensions: List of weak dimensions
        
        Returns:
            List of suggested strategies
        """
        suggestions = []
        
        # Get dimension-specific strategies first
        if weak_dimensions:
            for dim in weak_dimensions:
                dim_strategies = self.get_dimension_strategies(
                    document_type, dim, current_score
                )
                suggestions.extend(dim_strategies)
        
        # Add issue-based resolutions
        for issue in current_issues:
            # Extract issue category
            for issue_type in ["keyword", "personalization", "coherence", "alignment", "persuasiveness"]:
                if issue_type in issue.lower():
                    resolutions = self.get_issue_resolutions(document_type, issue_type)
                    suggestions.extend(resolutions[:2])
        
        # Add strategies from successful patterns with similar scores
        patterns = self.get_relevant_patterns(document_type, current_score)
        for pattern in patterns[:3]:
            for strategy in pattern.get("strategies", []):
                if isinstance(strategy, str):
                    suggestions.append(strategy)
                elif isinstance(strategy, list):
                    suggestions.extend(strategy[:2])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if isinstance(s, str) and s not in seen:
                seen.add(s)
                unique_suggestions.append(s)
        
        return unique_suggestions[:5]  # Top 5 suggestions
    
    def record_iteration_result(
        self,
        document_type: str,
        iteration: int,
        score_before: float,
        score_after: float,
        strategies_applied: List[str],
        dimension_changes: Dict[str, float]
    ):
        """
        Record the result of a single iteration for detailed learning.
        
        Args:
            document_type: Type of document
            iteration: Iteration number
            score_before: Score before revision
            score_after: Score after revision
            strategies_applied: What strategies were applied
            dimension_changes: Per-dimension score changes
        """
        key = f"{document_type}:iteration_history"
        
        if key not in self.improvement_strategies:
            self.improvement_strategies[key] = []
        
        result = {
            "iteration": iteration,
            "score_before": score_before,
            "score_after": score_after,
            "improvement": score_after - score_before,
            "strategies": strategies_applied,
            "dimension_changes": dimension_changes,
            "effective": score_after > score_before,
            "timestamp": datetime.now().isoformat()
        }
        
        self.improvement_strategies[key].append(result)
        
        # Keep only recent results
        if len(self.improvement_strategies[key]) > 50:
            self.improvement_strategies[key] = self.improvement_strategies[key][-50:]
    
    def save_to_disk(self, filename: str = "reflexion_memory.json"):
        """Save memory to disk for persistence"""
        filepath = os.path.join(self.cache_dir, filename)
        
        data = {
            "successful_patterns": self.successful_patterns,
            "common_issues": self.common_issues,
            "improvement_strategies": self.improvement_strategies,
            "dimension_patterns": self.dimension_patterns,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_disk(self, filename: str = "reflexion_memory.json"):
        """Load memory from disk"""
        filepath = os.path.join(self.cache_dir, filename)
        
        if not os.path.exists(filepath):
            return
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.successful_patterns = data.get("successful_patterns", [])
            self.common_issues = data.get("common_issues", {})
            self.improvement_strategies = data.get("improvement_strategies", {})
            self.dimension_patterns = data.get("dimension_patterns", {})
        except Exception as e:
            print(f"Warning: Failed to load memory from disk: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memory for debugging/reporting."""
        return {
            "total_patterns": len(self.successful_patterns),
            "issue_types": len(self.common_issues),
            "dimension_patterns": {k: len(v) for k, v in self.dimension_patterns.items()},
            "avg_improvement": (
                sum(p["improvement"] for p in self.successful_patterns) / len(self.successful_patterns)
                if self.successful_patterns else 0
            )
        }


# Global memory instance
_global_memory: Optional[ReflexionMemory] = None


def get_memory() -> ReflexionMemory:
    """Get global memory instance"""
    global _global_memory
    
    if _global_memory is None:
        _global_memory = ReflexionMemory()
        # Try to load existing memory
        _global_memory.load_from_disk()
    
    return _global_memory


def reset_memory():
    """Reset global memory instance (useful for testing)"""
    global _global_memory
    _global_memory = None
