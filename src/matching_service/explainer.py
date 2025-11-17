"""
LLM-enhanced explanation generation for program matches
"""

from typing import Dict, Any, Optional
import os

# Import LLM utilities (adjust path as needed)
try:
    from ..writing_agent.llm_utils import get_llm, call_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: LLM utilities not available. Using rule-based explanations only.")


class MatchExplainer:
    """
    Generates detailed, personalized explanations for program matches
    using LLM when available
    """
    
    def __init__(
        self, 
        llm_provider: str = "openai",
        model_name: str = "gpt-4-turbo-preview",
        use_llm: bool = True
    ):
        """
        Initialize explainer
        
        Args:
            llm_provider: LLM provider (openai, anthropic, qwen)
            model_name: Specific model name
            use_llm: Whether to use LLM for explanations
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.use_llm = use_llm and LLM_AVAILABLE
        
        if self.use_llm:
            try:
                self.llm = get_llm(
                    provider=llm_provider,
                    model_name=model_name,
                    temperature=0.7
                )
            except Exception as e:
                print(f"Warning: Failed to initialize LLM: {e}")
                self.use_llm = False
    
    def generate_detailed_explanation(
        self,
        profile: Dict[str, Any],
        program: Dict[str, Any],
        overall_score: float,
        dimension_scores: Dict[str, Any],
        strengths: list,
        gaps: list
    ) -> str:
        """
        Generate detailed explanation of program match
        
        Args:
            profile: Student profile
            program: Program data
            overall_score: Overall match score
            dimension_scores: Scores for each dimension
            strengths: List of student strengths
            gaps: List of student gaps
        
        Returns:
            Detailed explanation text
        """
        
        if self.use_llm:
            try:
                return self._generate_llm_explanation(
                    profile, program, overall_score, 
                    dimension_scores, strengths, gaps
                )
            except Exception as e:
                print(f"Warning: LLM explanation failed: {e}")
                # Fall back to rule-based
        
        return self._generate_rule_based_explanation(
            profile, program, overall_score,
            dimension_scores, strengths, gaps
        )
    
    def _generate_llm_explanation(
        self,
        profile: Dict[str, Any],
        program: Dict[str, Any],
        overall_score: float,
        dimension_scores: Dict[str, Any],
        strengths: list,
        gaps: list
    ) -> str:
        """Generate explanation using LLM"""
        
        # Format dimension scores
        scores_text = "\n".join([
            f"- {dim.replace('_', ' ').title()}: {score.score:.1%}"
            for dim, score in dimension_scores.items()
        ])
        
        prompt = f"""Analyze this student-program match and provide a concise, insightful explanation (100-150 words).

Student Profile:
- Major: {profile.get('major', 'N/A')}
- GPA: {profile.get('gpa', 'N/A')}
- Skills: {', '.join(profile.get('skills', [])[:5])}
- Experience: {len(profile.get('experiences', []))} positions
- Goals: {profile.get('goals', 'N/A')[:100]}...

Program:
- Name: {program.get('name', 'N/A')}
- University: {program.get('university', 'N/A')}
- Focus: {', '.join(program.get('focus_areas', [])[:3])}

Match Results:
- Overall Score: {overall_score:.1%}
{scores_text}

Strengths: {', '.join(strengths[:3])}
Areas to Improve: {', '.join(gaps[:3])}

Provide a personalized explanation that:
1. Explains why this is a {self._get_match_quality(overall_score)} match
2. Highlights the student's key strengths for this program
3. Notes important areas to address
4. Offers 1-2 specific recommendations

Be encouraging but honest. Use a professional, supportive tone."""

        try:
            explanation = call_llm(self.llm, prompt)
            return explanation.strip()
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}")
    
    def _generate_rule_based_explanation(
        self,
        profile: Dict[str, Any],
        program: Dict[str, Any],
        overall_score: float,
        dimension_scores: Dict[str, Any],
        strengths: list,
        gaps: list
    ) -> str:
        """Generate explanation using rules (fallback)"""
        
        program_name = program.get("name", "this program")
        match_quality = self._get_match_quality(overall_score)
        
        # Opening statement
        if overall_score >= 0.8:
            opening = f"Outstanding fit! Your profile is exceptionally well-aligned with {program_name}."
        elif overall_score >= 0.6:
            opening = f"Strong match. Your background positions you well for {program_name}."
        elif overall_score >= 0.4:
            opening = f"Moderate fit with {program_name}. With some improvements, you could be a strong candidate."
        else:
            opening = f"Limited alignment with {program_name}. Significant preparation would be needed."
        
        # Strengths section
        if strengths:
            strengths_text = f" Your key strengths include: {', '.join(strengths[:3])}."
        else:
            strengths_text = ""
        
        # Best dimension
        best_dim = max(dimension_scores.items(), key=lambda x: x[1].score)
        best_dim_name = best_dim[0].replace('_', ' ')
        best_score = best_dim[1].score
        
        dimension_text = f" Your {best_dim_name} alignment ({best_score:.0%}) is particularly strong."
        
        # Gaps section
        if gaps:
            gaps_text = f" Areas to strengthen: {', '.join(gaps[:2])}."
        else:
            gaps_text = " Your profile is comprehensive across all dimensions."
        
        # Recommendation
        if overall_score >= 0.7:
            recommendation = " Focus on crafting compelling application essays that highlight your unique experiences."
        elif overall_score >= 0.5:
            recommendation = " Consider taking additional coursework or gaining more relevant experience to strengthen your application."
        else:
            recommendation = " Explore foundational programs or certificate courses to build essential skills before applying."
        
        return opening + strengths_text + dimension_text + gaps_text + recommendation
    
    def _get_match_quality(self, score: float) -> str:
        """Get match quality description from score"""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "moderate"
        else:
            return "weak"
    
    def generate_comparison_explanation(
        self,
        profile: Dict[str, Any],
        matches: list
    ) -> str:
        """
        Generate explanation comparing top matches
        
        Args:
            profile: Student profile
            matches: List of ProgramMatch objects
        
        Returns:
            Comparative explanation
        """
        if not matches:
            return "No programs meet the minimum criteria."
        
        if len(matches) == 1:
            return f"Only one program meets your criteria: {matches[0].program_name}."
        
        # Get top 3
        top_matches = matches[:3]
        
        comparison = "Among your top matches:\n\n"
        
        for i, match in enumerate(top_matches, 1):
            comparison += f"{i}. **{match.program_name}** ({match.university})\n"
            comparison += f"   - Match Score: {match.overall_score:.1%}\n"
            comparison += f"   - Best For: {match.strengths[0] if match.strengths else 'Overall fit'}\n"
            if match.gaps:
                comparison += f"   - Consider: {match.gaps[0]}\n"
            comparison += "\n"
        
        return comparison.strip()