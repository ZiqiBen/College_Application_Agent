"""
Memory management for Reflexion-style learning
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
    
    def record_success(
        self,
        document_type: str,
        initial_score: float,
        final_score: float,
        iterations: int,
        strategies_used: List[str]
    ):
        """
        Record a successful generation pattern.
        
        Args:
            document_type: Type of document
            initial_score: Starting quality score
            final_score: Final quality score
            iterations: Number of iterations taken
            strategies_used: List of strategies that worked
        """
        improvement = final_score - initial_score
        
        if improvement > 0.1:  # Significant improvement
            pattern = {
                "document_type": document_type,
                "initial_score": initial_score,
                "final_score": final_score,
                "improvement": improvement,
                "iterations": iterations,
                "strategies": strategies_used,
                "timestamp": datetime.now().isoformat()
            }
            
            self.successful_patterns.append(pattern)
            
            # Keep only recent successful patterns
            if len(self.successful_patterns) > 100:
                self.successful_patterns = self.successful_patterns[-100:]
    
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
        
        Args:
            document_type: Type of document being generated
            current_score: Current quality score
        
        Returns:
            List of relevant patterns
        """
        relevant = []
        
        for pattern in self.successful_patterns:
            if pattern["document_type"] == document_type:
                # Prefer patterns with similar initial scores
                score_diff = abs(pattern["initial_score"] - current_score)
                if score_diff < 0.2:
                    relevant.append(pattern)
        
        # Sort by improvement magnitude
        relevant.sort(key=lambda x: x["improvement"], reverse=True)
        
        return relevant[:5]  # Top 5 relevant patterns
    
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
        current_issues: List[str]
    ) -> List[str]:
        """
        Suggest improvement strategies based on current issues.
        
        Args:
            document_type: Type of document
            current_issues: List of current issues
        
        Returns:
            List of suggested strategies
        """
        suggestions = []
        
        for issue in current_issues:
            # Extract issue category (e.g., "keyword_coverage", "personalization")
            for issue_type in ["keyword", "personalization", "coherence", "alignment", "persuasiveness"]:
                if issue_type in issue.lower():
                    resolutions = self.get_issue_resolutions(document_type, issue_type)
                    suggestions.extend(resolutions[:2])  # Top 2 resolutions
        
        # Add strategies from successful patterns
        patterns = self.get_relevant_patterns(document_type, 0.6)  # Assume mid-range score
        for pattern in patterns[:3]:
            suggestions.extend(pattern.get("strategies", []))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)
        
        return unique_suggestions[:5]  # Top 5 suggestions
    
    def save_to_disk(self, filename: str = "reflexion_memory.json"):
        """Save memory to disk for persistence"""
        filepath = os.path.join(self.cache_dir, filename)
        
        data = {
            "successful_patterns": self.successful_patterns,
            "common_issues": self.common_issues,
            "improvement_strategies": self.improvement_strategies,
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
        except Exception as e:
            print(f"Warning: Failed to load memory from disk: {e}")


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
