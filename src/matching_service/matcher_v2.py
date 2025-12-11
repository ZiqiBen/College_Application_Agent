"""
Core program matching engine V2 for new dataset

This matcher is designed for the V2 dataset with:
- Nested directory structure (e.g., graduate_programs/seas.harvard.edu/...)
- Schema v2.0 JSON format with extracted_fields
- Rich course information with descriptions
- Application requirements, program background, training outcomes
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models_v2 import (
    ProgramDataV2, ProgramMatchV2, MatchingResultV2,
    MatchLevelV2, DimensionScoreV2, MatchDimensionV2,
    MatchingRequestV2, MatchingResponseV2, ProgramMatchResponseV2,
    DimensionScoreResponseV2
)
from .scorer_v2 import DimensionScorerV2

# Import LLM utilities for generating fit reasons
try:
    from ..writing_agent.llm_utils import get_llm, call_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: LLM utilities not available for V2 matcher.")


class ProgramMatcherV2:
    """
    Program matching engine V2 for new dataset structure
    
    Features:
    - Handles nested directory structure
    - Schema v2.0 JSON parsing with null handling
    - Rich curriculum analysis
    - Course-level matching
    """
    
    def __init__(self, corpus_dir: str = "data_preparation/dataset/graduate_programs"):
        """
        Initialize V2 matcher
        
        Args:
            corpus_dir: Directory containing V2 program corpus
        """
        self.corpus_dir = corpus_dir
        self.programs: Dict[str, ProgramDataV2] = {}
        self._load_programs()
        
        self.scorer = DimensionScorerV2()
        
        print(f"[V2 Matcher] Loaded {len(self.programs)} programs from new dataset")
    
    def _load_programs(self) -> None:
        """Load all programs from V2 corpus directory"""
        if not os.path.exists(self.corpus_dir):
            print(f"Warning: V2 corpus directory {self.corpus_dir} not found")
            return
        
        corpus_path = Path(self.corpus_dir)
        
        # V2 structure: graduate_programs/domain.edu/timestamp_hash.json
        json_files = list(corpus_path.rglob("*.json"))
        
        loaded = 0
        failed = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Parse V2 format
                program = ProgramDataV2.from_json(data)
                
                if program.program_id:
                    self.programs[program.program_id] = program
                    loaded += 1
                    
            except Exception as e:
                failed += 1
                if failed <= 5:  # Only log first 5 failures
                    print(f"Warning: Failed to load {json_file.name}: {e}")
        
        if failed > 0:
            print(f"Warning: {failed} files failed to load")
    
    def match_programs(
        self,
        profile: Dict[str, Any],
        top_k: int = 10,
        min_score: float = 0.4,
        custom_weights: Optional[Dict[str, float]] = None,
        filters: Optional[Dict[str, Any]] = None,
        include_curriculum_analysis: bool = True
    ) -> MatchingResultV2:
        """
        Match student profile with programs
        
        Args:
            profile: Student profile dictionary
            top_k: Number of top matches to return
            min_score: Minimum overall score threshold
            custom_weights: Custom dimension weights
            filters: Optional filters (university, field, etc.)
            include_curriculum_analysis: Whether to include curriculum scoring
        
        Returns:
            MatchingResultV2 with ranked matches
        """
        timestamp = datetime.now().isoformat()
        matches: List[ProgramMatchV2] = []
        
        # Filter programs if needed
        programs_to_evaluate = self._apply_filters(self.programs, filters)
        
        # Score each program
        for program_id, program in programs_to_evaluate.items():
            try:
                match = self._evaluate_program(
                    profile=profile,
                    program=program,
                    custom_weights=custom_weights,
                    include_curriculum=include_curriculum_analysis
                )
                
                if match.overall_score >= min_score:
                    matches.append(match)
                    
            except Exception as e:
                print(f"Warning: Error evaluating {program_id}: {e}")
                continue
        
        # Sort by overall score
        matches.sort(key=lambda m: m.overall_score, reverse=True)
        
        # Take top K
        top_matches = matches[:top_k]
        
        # Generate fit reasons for top matches
        for match in top_matches:
            match.fit_reasons = self._generate_fit_reasons(profile, match)
        
        # Build profile summary
        profile_summary = {
            "name": profile.get("name", ""),
            "major": profile.get("major", ""),
            "gpa": profile.get("gpa"),
            "skills_count": len(profile.get("skills", [])),
            "experience_count": len(profile.get("experiences", [])),
            "goals_length": len(profile.get("goals", ""))
        }
        
        # Generate overall insights
        insights = self._generate_insights(profile, top_matches)
        
        return MatchingResultV2(
            student_profile_summary=profile_summary,
            total_programs_evaluated=len(programs_to_evaluate),
            matches=top_matches,
            top_k=top_k,
            min_score_threshold=min_score,
            matching_timestamp=timestamp,
            dataset_version="v2",
            overall_insights=insights
        )
    
    def _evaluate_program(
        self,
        profile: Dict[str, Any],
        program: ProgramDataV2,
        custom_weights: Optional[Dict[str, float]] = None,
        include_curriculum: bool = True
    ) -> ProgramMatchV2:
        """
        Evaluate a single program against profile
        
        Args:
            profile: Student profile
            program: Program data
            custom_weights: Custom dimension weights
            include_curriculum: Include curriculum analysis
        
        Returns:
            ProgramMatchV2 result
        """
        # Score all dimensions
        dimension_scores = self.scorer.score_all_dimensions(profile, program)
        
        # Optionally remove curriculum if not needed
        if not include_curriculum and MatchDimensionV2.CURRICULUM.value in dimension_scores:
            del dimension_scores[MatchDimensionV2.CURRICULUM.value]
        
        # Compute overall score
        overall_score = self.scorer.compute_overall_score(dimension_scores, custom_weights)
        
        # Determine match level
        match_level = self._determine_match_level(overall_score)
        
        # Extract strengths and gaps
        strengths, gaps = self._extract_strengths_and_gaps(dimension_scores)
        
        # Get matched and relevant courses
        matched_courses, relevant_courses = self._get_course_matches(profile, program)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(dimension_scores, gaps)
        
        return ProgramMatchV2(
            program_id=program.program_id,
            program_name=program.get_display_name(),
            university=program.get_school_name(),
            overall_score=overall_score,
            match_level=match_level,
            dimension_scores=dimension_scores,
            strengths=strengths,
            gaps=gaps,
            fit_reasons=[],  # Generated later for top matches
            recommendations=recommendations,
            matched_courses=matched_courses,
            relevant_courses=relevant_courses,
            program_data=program,
            metadata={
                "department": program.department,
                "source_url": program.source_url,
                "has_courses": len(program.courses) > 0,
                "has_career_info": bool(program.training_outcomes.career_paths)
            }
        )
    
    def _determine_match_level(self, score: float) -> MatchLevelV2:
        """Determine match level from score"""
        if score >= 0.85:
            return MatchLevelV2.EXCELLENT
        elif score >= 0.70:
            return MatchLevelV2.GOOD
        elif score >= 0.55:
            return MatchLevelV2.MODERATE
        elif score >= 0.40:
            return MatchLevelV2.FAIR
        else:
            return MatchLevelV2.WEAK
    
    def _extract_strengths_and_gaps(
        self,
        dimension_scores: Dict[str, DimensionScoreV2]
    ) -> Tuple[List[str], List[str]]:
        """Extract strengths and gaps from dimension scores"""
        strengths = []
        gaps = []
        
        for dim, score in dimension_scores.items():
            if score.score >= 0.7:
                # This is a strength
                strengths.extend(score.matched_items[:2])
                for factor in score.contributing_factors:
                    if any(word in factor.lower() for word in ["exceed", "strong", "highly", "possess"]):
                        strengths.append(factor)
            elif score.score < 0.5:
                # This is a gap
                gaps.extend(score.missing_items[:2])
                for factor in score.contributing_factors:
                    if any(word in factor.lower() for word in ["missing", "below", "need", "may"]):
                        gaps.append(factor)
        
        # Deduplicate and limit
        strengths = list(dict.fromkeys(strengths))[:5]
        gaps = list(dict.fromkeys(gaps))[:5]
        
        return strengths, gaps
    
    def _get_course_matches(
        self,
        profile: Dict[str, Any],
        program: ProgramDataV2
    ) -> Tuple[List[str], List[str]]:
        """Get matched and relevant courses"""
        matched = []
        relevant = []
        
        student_skills = set(s.lower() for s in profile.get("skills", []))
        student_goals = profile.get("goals", "").lower()
        
        for course in program.courses:
            course_text = f"{course.name} {course.description or ''}".lower()
            
            # Check for skill match
            skill_match = any(skill in course_text for skill in student_skills)
            # Check for goal relevance
            goal_match = any(word in course_text for word in student_goals.split() if len(word) > 4)
            
            if skill_match:
                matched.append(course.name)
            elif goal_match:
                relevant.append(course.name)
        
        return matched[:5], relevant[:5]
    
    def _generate_recommendations(
        self,
        dimension_scores: Dict[str, DimensionScoreV2],
        gaps: List[str]
    ) -> List[str]:
        """Generate recommendations based on gaps"""
        recommendations = []
        
        for dim, score in dimension_scores.items():
            if score.score < 0.6:
                if dim == MatchDimensionV2.ACADEMIC.value:
                    recommendations.append("Consider highlighting relevant coursework in your application")
                elif dim == MatchDimensionV2.SKILLS.value:
                    recommendations.append("Emphasize transferable skills and willingness to learn")
                elif dim == MatchDimensionV2.EXPERIENCE.value:
                    recommendations.append("Connect your experiences to the program's focus areas")
                elif dim == MatchDimensionV2.GOALS.value:
                    recommendations.append("Clarify how this program aligns with your career objectives")
                elif dim == MatchDimensionV2.REQUIREMENTS.value:
                    recommendations.append("Review application requirements and prepare required documents")
                elif dim == MatchDimensionV2.CURRICULUM.value:
                    recommendations.append("Research the curriculum and express interest in specific courses")
        
        return recommendations[:4]
    
    def _generate_fit_reasons(
        self,
        profile: Dict[str, Any],
        match: ProgramMatchV2
    ) -> List[str]:
        """Generate personalized 'Why This Program Fits You' reasons"""
        reasons = []
        program = match.program_data
        
        if not program:
            return reasons
        
        # 1. Academic fit
        if match.dimension_scores.get(MatchDimensionV2.ACADEMIC.value):
            academic_score = match.dimension_scores[MatchDimensionV2.ACADEMIC.value]
            if academic_score.score >= 0.7:
                major = profile.get("major", "your background")
                reasons.append(
                    f"Your {major} background provides strong preparation for this program's curriculum"
                )
        
        # 2. Skills alignment
        if match.dimension_scores.get(MatchDimensionV2.SKILLS.value):
            skills_score = match.dimension_scores[MatchDimensionV2.SKILLS.value]
            matched_skills = skills_score.matched_items[:3]
            if matched_skills:
                skill_names = [s.replace("Skill: ", "") for s in matched_skills]
                reasons.append(
                    f"Your skills in {', '.join(skill_names)} directly apply to the program's focus"
                )
        
        # 3. Career alignment
        outcomes = program.training_outcomes
        if outcomes.career_paths:
            student_goals = profile.get("goals", "")
            if student_goals:
                reasons.append(
                    f"The program's career outcomes align with your goal to {student_goals[:50]}..."
                )
        
        # 4. Course relevance
        if match.matched_courses:
            reasons.append(
                f"Courses like '{match.matched_courses[0]}' directly match your interests"
            )
        
        # 5. Mission fit
        bg = program.program_background
        if bg.mission:
            reasons.append(
                f"The program's mission to {bg.mission[:60]}... resonates with your objectives"
            )
        
        return reasons[:4]
    
    def _generate_insights(
        self,
        profile: Dict[str, Any],
        top_matches: List[ProgramMatchV2]
    ) -> Dict[str, Any]:
        """Generate overall matching insights"""
        if not top_matches:
            return {"summary": "No matching programs found"}
        
        # Calculate average scores by dimension
        dim_averages = {}
        for dim in MatchDimensionV2:
            scores = [
                m.dimension_scores.get(dim.value, DimensionScoreV2(dim.value, 0, 0)).score
                for m in top_matches
            ]
            if scores:
                dim_averages[dim.value] = sum(scores) / len(scores)
        
        # Find strongest and weakest dimensions
        sorted_dims = sorted(dim_averages.items(), key=lambda x: x[1], reverse=True)
        strongest = sorted_dims[0] if sorted_dims else None
        weakest = sorted_dims[-1] if sorted_dims else None
        
        # Count universities
        universities = set(m.university for m in top_matches)
        
        return {
            "average_match_score": sum(m.overall_score for m in top_matches) / len(top_matches),
            "dimension_averages": dim_averages,
            "strongest_dimension": strongest[0] if strongest else None,
            "weakest_dimension": weakest[0] if weakest else None,
            "universities_matched": len(universities),
            "programs_with_courses": sum(1 for m in top_matches if m.matched_courses),
            "summary": f"Found {len(top_matches)} matching programs across {len(universities)} universities"
        }
    
    def _apply_filters(
        self,
        programs: Dict[str, ProgramDataV2],
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, ProgramDataV2]:
        """Apply filters to programs"""
        if not filters:
            return programs
        
        filtered = {}
        
        for pid, program in programs.items():
            include = True
            
            # University filter
            if "university" in filters:
                uni_filter = filters["university"].lower()
                if uni_filter not in program.get_school_name().lower():
                    include = False
            
            # Field filter
            if "field" in filters and include:
                field_filter = filters["field"].lower()
                program_text = program.get_full_text().lower()
                if field_filter not in program_text:
                    include = False
            
            # Has courses filter
            if filters.get("has_courses") and include:
                if not program.courses:
                    include = False
            
            if include:
                filtered[pid] = program
        
        return filtered
    
    def get_program(self, program_id: str) -> Optional[ProgramDataV2]:
        """Get a specific program by ID"""
        return self.programs.get(program_id)
    
    def get_program_details_for_writing(self, program_id: str) -> Optional[Dict[str, Any]]:
        """
        Get program details formatted for Writing Agent
        
        Args:
            program_id: Program identifier
        
        Returns:
            Dictionary with program info for writing agent
        """
        program = self.programs.get(program_id)
        if not program:
            return None
        
        # Format courses
        courses_info = []
        for course in program.courses:
            course_dict = {"name": course.name}
            if course.description:
                course_dict["description"] = course.description
            courses_info.append(course_dict)
        
        return {
            "program_id": program.program_id,
            "program_name": program.get_display_name(),
            "university": program.get_school_name(),
            "department": program.department,
            "duration": program.duration,
            "source_url": program.source_url,
            
            # Text content
            "description_text": program.get_full_text(),
            "raw_text": program.raw_text,
            
            # Structured fields
            "courses": courses_info,
            "course_descriptions": program.get_course_descriptions(),
            
            # Requirements
            "application_requirements": {
                "academic_background": program.application_requirements.academic_background,
                "prerequisites": program.application_requirements.prerequisites,
                "gre": program.application_requirements.gre,
                "english_tests": program.application_requirements.english_tests,
                "documents": program.application_requirements.documents,
                "summary": program.application_requirements.summary
            },
            
            # Program info
            "program_background": {
                "mission": program.program_background.mission,
                "environment": program.program_background.environment,
                "faculty": program.program_background.faculty,
                "summary": program.program_background.summary
            },
            
            # Outcomes
            "training_outcomes": {
                "goals": program.training_outcomes.goals,
                "career_paths": program.training_outcomes.career_paths,
                "research_orientation": program.training_outcomes.research_orientation,
                "professional_orientation": program.training_outcomes.professional_orientation,
                "summary": program.training_outcomes.summary
            },
            
            # Tuition and contact
            "tuition": program.tuition,
            "contact_email": program.contact_email,
            
            # Chunks for RAG
            "chunks": [
                {"chunk_id": c.chunk_id, "text": c.text}
                for c in program.chunks
            ]
        }
    
    def list_programs(self) -> List[Dict[str, Any]]:
        """List all available programs with basic info"""
        programs_list = []
        
        for pid, program in self.programs.items():
            programs_list.append({
                "program_id": pid,
                "program_name": program.get_display_name(),
                "university": program.get_school_name(),
                "department": program.department,
                "has_courses": len(program.courses) > 0,
                "has_career_info": bool(program.training_outcomes.career_paths),
                "source_url": program.source_url
            })
        
        return programs_list


def convert_match_to_response(match: ProgramMatchV2) -> ProgramMatchResponseV2:
    """Convert ProgramMatchV2 to API response model"""
    return ProgramMatchResponseV2(
        program_id=match.program_id,
        program_name=match.program_name,
        university=match.university,
        department=match.program_data.department if match.program_data else None,
        overall_score=round(match.overall_score, 3),
        match_level=match.match_level.value,
        dimension_scores={
            dim: DimensionScoreResponseV2(
                dimension=score.dimension,
                score=round(score.score, 3),
                weight=score.weight,
                details=score.details,
                contributing_factors=score.contributing_factors,
                matched_items=score.matched_items,
                missing_items=score.missing_items
            )
            for dim, score in match.dimension_scores.items()
        },
        strengths=match.strengths,
        gaps=match.gaps,
        fit_reasons=match.fit_reasons,
        recommendations=match.recommendations,
        matched_courses=match.matched_courses,
        relevant_courses=match.relevant_courses,
        explanation=match.explanation,
        program_details=match.program_data.to_dict() if hasattr(match.program_data, 'to_dict') else None,
        metadata=match.metadata
    )
