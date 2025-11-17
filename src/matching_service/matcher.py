"""
Core program matching engine
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from .models import (
    ProgramMatch, MatchingResult, MatchLevel,
    DimensionScore, ProgramData
)
from .scorer import DimensionScorer


class ProgramMatcher:
    """
    Program matching engine that analyzes student background
    and finds the best-fitting programs
    """
    
    def __init__(self, corpus_dir: str = "data/corpus"):
        """
        Initialize matcher
        
        Args:
            corpus_dir: Directory containing program corpus
        """
        self.corpus_dir = corpus_dir
        self.programs = self._load_programs()
        self.scorer = DimensionScorer()
        
        print(f"Loaded {len(self.programs)} programs from corpus")
    
    def _load_programs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load program data from corpus directory
        
        Returns:
            Dictionary mapping program_id to program data
        """
        programs = {}
        
        if not os.path.exists(self.corpus_dir):
            print(f"Warning: Corpus directory {self.corpus_dir} not found")
            return programs
        
        # Recursively find all JSON files
        corpus_path = Path(self.corpus_dir)
        json_files = list(corpus_path.rglob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract program information
                program_id = data.get("id", json_file.stem)
                
                # Parse program data from various formats
                program_data = self._parse_program_data(data, program_id)
                
                if program_data:
                    programs[program_id] = program_data
                    
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue
        
        return programs
    
    def _parse_program_data(self, data: Dict[str, Any], program_id: str) -> Optional[Dict[str, Any]]:
        """
        Parse program data from JSON into standardized format
        
        Args:
            data: Raw JSON data
            program_id: Program identifier
        
        Returns:
            Standardized program dictionary or None
        """
        try:
            # Extract fields from data (adjust based on your corpus schema)
            extracted_fields = data.get("extracted_fields", {})
            
            program_name = extracted_fields.get("program_name") or data.get("title", "Unknown Program")
            
            # Build description text from various fields
            description_parts = []
            if data.get("raw_text"):
                description_parts.append(data["raw_text"][:1000])
            if extracted_fields.get("features"):
                description_parts.append(extracted_fields["features"])
            
            description_text = " ".join(description_parts)
            
            # Extract university from source URL or ID
            source_url = data.get("source_url", "")
            university = self._extract_university_name(source_url, program_id)
            
            # Parse courses
            courses_data = extracted_fields.get("courses", [])
            core_courses = []
            if isinstance(courses_data, list):
                for course in courses_data:
                    if isinstance(course, dict):
                        core_courses.append(course.get("name", ""))
                    elif isinstance(course, str):
                        core_courses.append(course)
            
            # Build program data
            program_data = {
                "program_id": program_id,
                "name": program_name,
                "university": university,
                "field": "Data Science",  # Infer from corpus or set default
                "min_gpa": 3.0,  # Default, adjust if available in data
                "required_skills": self._extract_skills(description_text),
                "prerequisite_courses": core_courses,
                "focus_areas": self._extract_focus_areas(description_text),
                "core_courses": core_courses,
                "career_outcomes": extracted_fields.get("features", ""),
                "duration": extracted_fields.get("duration", ""),
                "tuition": extracted_fields.get("tuition"),
                "description_text": description_text,
                "language_requirements": {},
                "source_url": source_url
            }
            
            return program_data
            
        except Exception as e:
            print(f"Warning: Error parsing program {program_id}: {e}")
            return None
    
    def _extract_university_name(self, url: str, program_id: str) -> str:
        """Extract university name from URL or program ID"""
        if "stanford" in url.lower() or "stanford" in program_id.lower():
            return "Stanford University"
        elif "mit" in url.lower() or "mit" in program_id.lower():
            return "MIT"
        elif "cmu" in url.lower() or "cmu" in program_id.lower():
            return "Carnegie Mellon University"
        elif "berkeley" in url.lower() or "ucb" in program_id.lower():
            return "UC Berkeley"
        elif "columbia" in url.lower():
            return "Columbia University"
        else:
            # Try to extract domain
            if "://" in url:
                domain = url.split("://")[1].split("/")[0]
                return domain.replace("www.", "").split(".")[0].title()
            return "Unknown University"
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract technical skills from program text"""
        common_skills = [
            "Python", "R", "SQL", "Machine Learning", "Deep Learning",
            "Statistics", "Data Visualization", "Big Data", "Algorithms",
            "TensorFlow", "PyTorch", "NLP", "Computer Vision"
        ]
        
        text_lower = text.lower()
        found_skills = [skill for skill in common_skills if skill.lower() in text_lower]
        
        return found_skills[:10]  # Return top 10
    
    def _extract_focus_areas(self, text: str) -> List[str]:
        """Extract program focus areas from text"""
        focus_keywords = [
            "machine learning", "artificial intelligence", "data analytics",
            "statistics", "computer vision", "natural language processing",
            "big data", "data engineering", "business analytics"
        ]
        
        text_lower = text.lower()
        found_areas = [area for area in focus_keywords if area in text_lower]
        
        return found_areas[:5]
    
    def match_programs(
        self,
        profile: Dict[str, Any],
        top_k: int = 5,
        min_score: float = 0.5,
        custom_weights: Optional[Dict[str, float]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> MatchingResult:
        """
        Match student profile against all programs
        
        Args:
            profile: Student profile dictionary
            top_k: Return top K matches
            min_score: Minimum score threshold
            custom_weights: Custom dimension weights
            filters: Optional filters (e.g., location, tuition)
        
        Returns:
            MatchingResult with ranked programs
        """
        matches = []
        
        for program_id, program_data in self.programs.items():
            # Apply filters if provided
            if filters and not self._passes_filters(program_data, filters):
                continue
            
            # Score all dimensions
            dimension_scores = {
                "academic": self.scorer.score_academic(profile, program_data),
                "skills": self.scorer.score_skills(profile, program_data),
                "experience": self.scorer.score_experience(profile, program_data),
                "goals": self.scorer.score_goals(profile, program_data),
                "requirements": self.scorer.score_requirements(profile, program_data)
            }
            
            # Compute overall score
            overall_score = self.scorer.compute_overall_score(
                dimension_scores,
                custom_weights
            )
            
            # Skip if below threshold
            if overall_score < min_score:
                continue
            
            # Determine match level
            match_level = self._determine_match_level(overall_score)
            
            # Identify strengths and gaps
            strengths = self._identify_strengths(profile, program_data, dimension_scores)
            gaps = self._identify_gaps(profile, program_data, dimension_scores)
            recommendations = self._generate_recommendations(gaps, dimension_scores)
            
            # Generate explanation
            explanation = self._generate_explanation(
                profile, program_data, overall_score, dimension_scores
            )
            
            # Create match object
            match = ProgramMatch(
                program_id=program_id,
                program_name=program_data["name"],
                university=program_data["university"],
                overall_score=overall_score,
                match_level=match_level,
                dimension_scores=dimension_scores,
                strengths=strengths,
                gaps=gaps,
                recommendations=recommendations,
                explanation=explanation,
                metadata={
                    "duration": program_data.get("duration"),
                    "tuition": program_data.get("tuition"),
                    "source_url": program_data.get("source_url")
                }
            )
            
            matches.append(match)
        
        # Sort by overall score
        matches.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Take top K
        top_matches = matches[:top_k]
        
        # Generate overall insights
        insights = self._generate_overall_insights(profile, top_matches, matches)
        
        # Build result
        result = MatchingResult(
            student_profile_summary=self._summarize_profile(profile),
            total_programs_evaluated=len(self.programs),
            matches=top_matches,
            top_k=top_k,
            min_score_threshold=min_score,
            matching_timestamp=datetime.now().isoformat(),
            overall_insights=insights
        )
        
        return result
    
    def _passes_filters(self, program: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if program passes filter criteria"""
        # Implement filter logic (e.g., location, tuition range, etc.)
        # This is a placeholder - customize based on your needs
        
        if "max_tuition" in filters:
            tuition_str = program.get("tuition", "")
            # Simple tuition parsing (improve as needed)
            if tuition_str and "$" in tuition_str:
                try:
                    tuition_value = int(''.join(filter(str.isdigit, tuition_str)))
                    if tuition_value > filters["max_tuition"]:
                        return False
                except:
                    pass
        
        return True
    
    def _determine_match_level(self, score: float) -> MatchLevel:
        """Determine match quality level from score"""
        if score >= 0.8:
            return MatchLevel.EXCELLENT
        elif score >= 0.6:
            return MatchLevel.GOOD
        elif score >= 0.4:
            return MatchLevel.MODERATE
        else:
            return MatchLevel.WEAK
    
    def _identify_strengths(
        self, 
        profile: Dict[str, Any], 
        program: Dict[str, Any],
        scores: Dict[str, DimensionScore]
    ) -> List[str]:
        """Identify student's strengths for this program"""
        strengths = []
        
        # Check high-scoring dimensions
        for dim_name, dim_score in scores.items():
            if dim_score.score >= 0.8:
                strengths.append(f"Strong {dim_name} match ({dim_score.score:.0%})")
        
        # GPA strength
        if profile.get("gpa", 0) >= program.get("min_gpa", 3.0) + 0.3:
            strengths.append(f"Excellent GPA ({profile.get('gpa', 0):.2f})")
        
        # Skills strength
        student_skills = set(s.lower() for s in profile.get("skills", []))
        required_skills = set(s.lower() for s in program.get("required_skills", []))
        if required_skills and len(student_skills & required_skills) / len(required_skills) > 0.8:
            strengths.append("Excellent skill alignment")
        
        # Experience strength
        if len(profile.get("experiences", [])) >= 2:
            strengths.append("Substantial professional experience")
        
        return strengths[:5]  # Top 5
    
    def _identify_gaps(
        self, 
        profile: Dict[str, Any], 
        program: Dict[str, Any],
        scores: Dict[str, DimensionScore]
    ) -> List[str]:
        """Identify areas for improvement"""
        gaps = []
        
        # Check low-scoring dimensions
        for dim_name, dim_score in scores.items():
            if dim_score.score < 0.5:
                gaps.append(f"Improve {dim_name} ({dim_score.score:.0%} match)")
        
        # GPA gap
        if profile.get("gpa", 0) < program.get("min_gpa", 3.0):
            gaps.append(f"GPA below requirement ({profile.get('gpa', 0):.2f} vs {program.get('min_gpa', 3.0):.2f})")
        
        # Skills gap
        student_skills = set(s.lower() for s in profile.get("skills", []))
        required_skills = set(s.lower() for s in program.get("required_skills", []))
        missing_skills = required_skills - student_skills
        if missing_skills:
            gaps.append(f"Missing skills: {', '.join(list(missing_skills)[:3])}")
        
        # Experience gap
        if not profile.get("experiences"):
            gaps.append("Limited professional experience")
        
        return gaps[:5]  # Top 5
    
    def _generate_recommendations(
        self, 
        gaps: List[str],
        scores: Dict[str, DimensionScore]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        for dim_name, dim_score in scores.items():
            if dim_score.score < 0.6:
                if dim_name == "academic":
                    recommendations.append("Consider taking additional coursework in program focus areas")
                elif dim_name == "skills":
                    recommendations.append("Develop missing technical skills through online courses or projects")
                elif dim_name == "experience":
                    recommendations.append("Gain relevant experience through internships or research projects")
                elif dim_name == "goals":
                    recommendations.append("Clarify career goals and articulate alignment with program outcomes")
        
        if not recommendations:
            recommendations.append("Profile is strong - focus on crafting compelling application materials")
        
        return recommendations[:3]  # Top 3
    
    def _generate_explanation(
        self,
        profile: Dict[str, Any],
        program: Dict[str, Any],
        overall_score: float,
        dimension_scores: Dict[str, DimensionScore]
    ) -> str:
        """Generate human-readable match explanation"""
        program_name = program.get("name", "this program")
        
        if overall_score >= 0.8:
            opening = f"Excellent fit! Your background strongly aligns with {program_name}."
        elif overall_score >= 0.6:
            opening = f"Good match. You meet most requirements for {program_name}."
        elif overall_score >= 0.4:
            opening = f"Moderate fit. Consider strengthening some areas for {program_name}."
        else:
            opening = f"Limited fit with {program_name}. Significant improvements needed."
        
        # Highlight strongest dimension
        best_dim = max(dimension_scores.items(), key=lambda x: x[1].score)
        best_dim_name = best_dim[0].replace("_", " ").title()
        
        strength_note = f" Your {best_dim_name} ({best_dim[1].score:.0%} match) is particularly strong."
        
        # Note weakest dimension if score is low
        weak_dim = min(dimension_scores.items(), key=lambda x: x[1].score)
        if weak_dim[1].score < 0.5:
            weak_dim_name = weak_dim[0].replace("_", " ").title()
            weakness_note = f" Consider improving your {weak_dim_name} ({weak_dim[1].score:.0%} match)."
        else:
            weakness_note = ""
        
        return opening + strength_note + weakness_note
    
    def _summarize_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of student profile"""
        return {
            "name": profile.get("name", "Student"),
            "major": profile.get("major", "Unknown"),
            "gpa": profile.get("gpa"),
            "skills_count": len(profile.get("skills", [])),
            "experiences_count": len(profile.get("experiences", [])),
            "has_clear_goals": len(profile.get("goals", "")) > 50
        }
    
    def _generate_overall_insights(
        self,
        profile: Dict[str, Any],
        top_matches: List[ProgramMatch],
        all_matches: List[ProgramMatch]
    ) -> Dict[str, Any]:
        """Generate overall insights about matching results"""
        if not top_matches:
            return {"message": "No programs meet the minimum score threshold"}
        
        avg_score = sum(m.overall_score for m in top_matches) / len(top_matches)
        
        # Count matches by level
        level_counts = {}
        for match in all_matches:
            level = match.match_level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Identify common strengths
        all_strengths = []
        for match in top_matches:
            all_strengths.extend(match.strengths)
        
        common_strengths = list(set(all_strengths))[:3]
        
        # Identify common gaps
        all_gaps = []
        for match in top_matches:
            all_gaps.extend(match.gaps)
        
        common_gaps = list(set(all_gaps))[:3]
        
        return {
            "average_match_score": round(avg_score, 3),
            "matches_by_level": level_counts,
            "common_strengths": common_strengths,
            "common_gaps": common_gaps,
            "total_qualified_programs": len(all_matches)
        }