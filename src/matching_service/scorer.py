"""
Multi-dimensional scoring algorithms for program matching
"""

from typing import Dict, Any, List
import numpy as np
from sentence_transformers import SentenceTransformer
from .models import DimensionScore, MatchDimension


class DimensionScorer:
    """Multi-dimensional scorer"""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize scorer
        
        Args:
            embedding_model_name: Sentence-BERT model name
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Default weight configuration
        self.default_weights = {
            MatchDimension.ACADEMIC: 0.30,
            MatchDimension.SKILLS: 0.25,
            MatchDimension.EXPERIENCE: 0.20,
            MatchDimension.GOALS: 0.15,
            MatchDimension.REQUIREMENTS: 0.10
        }
    
    def score_academic(
        self, 
        profile: Dict[str, Any], 
        program: Dict[str, Any]
    ) -> DimensionScore:
        """
        Evaluate academic background match
        
        Factors:
        1. GPA match (40%)
        2. Major relevance (40%)
        3. Coursework background (20%)
        """
        score = 0.0
        details = {}
        factors = []
        
        # 1. GPA match
        student_gpa = profile.get("gpa", 0.0)
        required_gpa = program.get("min_gpa", 3.0)
        
        if student_gpa >= required_gpa + 0.3:
            gpa_score = 1.0
            factors.append(f"GPA {student_gpa:.2f} significantly exceeds requirement {required_gpa:.2f}")
        elif student_gpa >= required_gpa:
            gpa_score = 0.8
            factors.append(f"GPA {student_gpa:.2f} meets requirement {required_gpa:.2f}")
        elif student_gpa >= required_gpa - 0.2:
            gpa_score = 0.5
            factors.append(f"GPA {student_gpa:.2f} slightly below requirement {required_gpa:.2f}")
        else:
            gpa_score = 0.2
            factors.append(f"GPA {student_gpa:.2f} below requirement {required_gpa:.2f}")
        
        score += gpa_score * 0.4
        details["gpa_score"] = gpa_score
        details["student_gpa"] = student_gpa
        details["required_gpa"] = required_gpa
        
        # 2. Major relevance (using embedding similarity)
        student_major = profile.get("major", "")
        program_field = program.get("field", "")
        
        if student_major and program_field:
            major_similarity = self._compute_text_similarity(
                student_major, 
                program_field
            )
            score += major_similarity * 0.4
            details["major_similarity"] = major_similarity
            
            if major_similarity > 0.7:
                factors.append(f"Major '{student_major}' highly relevant to program field '{program_field}'")
            elif major_similarity > 0.4:
                factors.append(f"Major background somewhat relevant to program field")
        else:
            score += 0.2  # Low score when information missing
        
        # 3. Coursework match
        student_courses = set(c.lower() for c in profile.get("courses", []))
        required_courses = set(c.lower() for c in program.get("prerequisite_courses", []))
        
        if required_courses:
            course_overlap = len(student_courses & required_courses)
            course_score = course_overlap / len(required_courses)
            score += course_score * 0.2
            details["course_match_ratio"] = course_score
            
            if course_score >= 0.8:
                factors.append(f"Completed {course_overlap}/{len(required_courses)} prerequisite courses")
            elif course_score < 0.5:
                missing = required_courses - student_courses
                factors.append(f"Missing prerequisites: {', '.join(list(missing)[:3])}")
        else:
            score += 0.15  # Medium score when no requirements
        
        return DimensionScore(
            dimension=MatchDimension.ACADEMIC.value,
            score=min(score, 1.0),
            weight=self.default_weights[MatchDimension.ACADEMIC],
            details=details,
            contributing_factors=factors
        )
    
    def score_skills(
        self, 
        profile: Dict[str, Any], 
        program: Dict[str, Any]
    ) -> DimensionScore:
        """
        Evaluate skills match
        
        Factors:
        1. Required skills coverage (60%)
        2. Skill depth/breadth (40%)
        """
        score = 0.0
        details = {}
        factors = []
        
        student_skills = set(s.lower().strip() for s in profile.get("skills", []))
        required_skills = set(s.lower().strip() for s in program.get("required_skills", []))
        
        if not required_skills:
            # No explicit skill requirements, assess based on program description
            program_text = program.get("description_text", "")
            if program_text and student_skills:
                # Calculate skill appearance frequency in program description
                skill_relevance = sum(
                    1 for skill in student_skills 
                    if skill in program_text.lower()
                ) / max(len(student_skills), 1)
                score = skill_relevance * 0.7
                factors.append(f"Skills relevance to program: {skill_relevance:.2%}")
            else:
                score = 0.5
                factors.append("Program has no explicit skill requirements")
        else:
            # 1. Required skills coverage
            overlap = student_skills & required_skills
            missing = required_skills - student_skills
            
            coverage_ratio = len(overlap) / len(required_skills)
            score += coverage_ratio * 0.6
            
            details["skill_coverage"] = coverage_ratio
            details["matched_skills"] = list(overlap)
            details["missing_skills"] = list(missing)
            
            if coverage_ratio >= 0.8:
                factors.append(f"Possess {len(overlap)}/{len(required_skills)} required skills")
            elif coverage_ratio >= 0.5:
                factors.append(f"Possess most required skills ({len(overlap)}/{len(required_skills)})")
            else:
                factors.append(f"Missing key skills: {', '.join(list(missing)[:3])}")
            
            # 2. Skill depth/breadth
            extra_skills = student_skills - required_skills
            if extra_skills:
                breadth_score = min(len(extra_skills) / 10, 1.0) * 0.4
                score += breadth_score
                factors.append(f"Additionally possess {len(extra_skills)} related skills")
        
        return DimensionScore(
            dimension=MatchDimension.SKILLS.value,
            score=min(score, 1.0),
            weight=self.default_weights[MatchDimension.SKILLS],
            details=details,
            contributing_factors=factors
        )
    
    def score_experience(
        self, 
        profile: Dict[str, Any], 
        program: Dict[str, Any]
    ) -> DimensionScore:
        """
        Evaluate experience match
        
        Factors:
        1. Experience quantity (30%)
        2. Experience relevance (50%)
        3. Impact/achievements (20%)
        """
        score = 0.0
        details = {}
        factors = []
        
        experiences = profile.get("experiences", [])
        program_focus = program.get("focus_areas", [])
        program_text = program.get("description_text", "")
        
        if not experiences:
            score = 0.1
            factors.append("Lack of relevant work/project experience")
            return DimensionScore(
                dimension=MatchDimension.EXPERIENCE.value,
                score=score,
                weight=self.default_weights[MatchDimension.EXPERIENCE],
                details=details,
                contributing_factors=factors
            )
        
        # 1. Experience quantity score
        exp_count = len(experiences)
        if exp_count >= 3:
            quantity_score = 1.0
            factors.append(f"Rich experience with {exp_count} relevant positions")
        elif exp_count == 2:
            quantity_score = 0.8
            factors.append(f"Good experience with {exp_count} relevant positions")
        else:
            quantity_score = 0.5
            factors.append(f"Limited experience with {exp_count} position(s)")
        
        score += quantity_score * 0.3
        details["experience_count"] = exp_count
        
        # 2. Experience relevance (using embeddings)
        exp_texts = []
        for exp in experiences:
            exp_text = f"{exp.get('title', '')} {exp.get('impact', '')} {' '.join(exp.get('skills', []))}"
            exp_texts.append(exp_text)
        
        # Similarity to program focus areas
        focus_text = " ".join(program_focus) if program_focus else program_text[:500]
        
        if exp_texts and focus_text:
            similarities = []
            for exp_text in exp_texts:
                sim = self._compute_text_similarity(exp_text, focus_text)
                similarities.append(sim)
            
            # Take average of top 2 experiences
            top_similarities = sorted(similarities, reverse=True)[:2]
            relevance_score = np.mean(top_similarities) if top_similarities else 0.3
            
            score += relevance_score * 0.5
            details["relevance_score"] = relevance_score
            details["top_experience_matches"] = top_similarities
            
            if relevance_score > 0.7:
                factors.append("Experience highly relevant to program focus")
            elif relevance_score > 0.4:
                factors.append("Experience somewhat relevant to program focus")
        else:
            score += 0.15
        
        # 3. Impact/achievements (based on quantification)
        quantified_count = sum(
            1 for exp in experiences
            if any(char.isdigit() for char in exp.get("impact", ""))
        )
        
        impact_score = min(quantified_count / len(experiences), 1.0)
        score += impact_score * 0.2
        
        if quantified_count >= len(experiences) * 0.7:
            factors.append("Experience descriptions include quantified results")
        
        details["quantified_achievements"] = quantified_count
        
        return DimensionScore(
            dimension=MatchDimension.EXPERIENCE.value,
            score=min(score, 1.0),
            weight=self.default_weights[MatchDimension.EXPERIENCE],
            details=details,
            contributing_factors=factors
        )
    
    def score_goals(
        self, 
        profile: Dict[str, Any], 
        program: Dict[str, Any]
    ) -> DimensionScore:
        """
        Evaluate career goals match
        
        Factors:
        1. Career goals alignment with program outcomes (70%)
        2. Goal clarity (30%)
        """
        score = 0.0
        details = {}
        factors = []
        
        student_goals = profile.get("goals", "")
        program_outcomes = program.get("career_outcomes", "")
        program_text = program.get("description_text", "")
        
        if not student_goals or len(student_goals) < 20:
            score = 0.3
            factors.append("Career goals not clearly articulated")
            details["goal_clarity"] = "unclear"
            return DimensionScore(
                dimension=MatchDimension.GOALS.value,
                score=score,
                weight=self.default_weights[MatchDimension.GOALS],
                details=details,
                contributing_factors=factors
            )
        
        # 1. Goals alignment with program outcomes
        target_text = program_outcomes if program_outcomes else program_text[:500]
        
        if target_text:
            goal_similarity = self._compute_text_similarity(
                student_goals,
                target_text
            )
            score += goal_similarity * 0.7
            details["goal_alignment"] = goal_similarity
            
            if goal_similarity > 0.7:
                factors.append("Career goals highly aligned with program outcomes")
            elif goal_similarity > 0.5:
                factors.append("Career goals reasonably aligned with program")
            else:
                factors.append("Career goals moderately aligned with program")
        else:
            score += 0.35
        
        # 2. Goal clarity
        goal_length = len(student_goals.split())
        if goal_length >= 30:
            clarity_score = 1.0
            factors.append("Career goals clearly and thoroughly articulated")
        elif goal_length >= 15:
            clarity_score = 0.7
        else:
            clarity_score = 0.4
        
        score += clarity_score * 0.3
        details["goal_clarity_score"] = clarity_score
        
        return DimensionScore(
            dimension=MatchDimension.GOALS.value,
            score=min(score, 1.0),
            weight=self.default_weights[MatchDimension.GOALS],
            details=details,
            contributing_factors=factors
        )
    
    def score_requirements(
        self, 
        profile: Dict[str, Any], 
        program: Dict[str, Any]
    ) -> DimensionScore:
        """
        Evaluate hard requirements compliance
        
        Factors:
        1. GPA requirement (40%)
        2. Language test requirement (30%)
        3. Prerequisite courses (30%)
        """
        score = 1.0  # Start with perfect score and deduct
        details = {}
        factors = []
        
        # 1. GPA requirement
        student_gpa = profile.get("gpa", 0.0)
        required_gpa = program.get("min_gpa", 0.0)
        
        if student_gpa >= required_gpa:
            factors.append(f"Meets GPA requirement ({student_gpa:.2f} >= {required_gpa:.2f})")
            details["gpa_met"] = True
        elif student_gpa >= required_gpa - 0.2:
            score -= 0.2
            factors.append(f"GPA slightly below requirement ({student_gpa:.2f} vs {required_gpa:.2f})")
            details["gpa_met"] = False
        else:
            score -= 0.4
            factors.append(f"GPA significantly below requirement ({student_gpa:.2f} vs {required_gpa:.2f})")
            details["gpa_met"] = False
        
        # 2. Language test requirements (simplified)
        lang_req = program.get("language_requirements", {})
        if lang_req:
            toefl_required = lang_req.get("toefl_min", 0)
            ielts_required = lang_req.get("ielts_min", 0)
            
            student_toefl = profile.get("toefl_score", 0)
            student_ielts = profile.get("ielts_score", 0)
            
            if toefl_required > 0 or ielts_required > 0:
                if (student_toefl >= toefl_required) or (student_ielts >= ielts_required):
                    factors.append("Meets language test requirements")
                    details["language_met"] = True
                elif student_toefl > 0 or student_ielts > 0:
                    score -= 0.15
                    factors.append("Language test score slightly below requirement")
                    details["language_met"] = False
                else:
                    score -= 0.3
                    factors.append("Missing language test scores")
                    details["language_met"] = False
        
        # 3. Prerequisite courses
        required_courses = set(c.lower() for c in program.get("prerequisite_courses", []))
        student_courses = set(c.lower() for c in profile.get("courses", []))
        
        if required_courses:
            missing_courses = required_courses - student_courses
            if not missing_courses:
                factors.append("Completed all prerequisite courses")
                details["courses_met"] = True
            elif len(missing_courses) <= len(required_courses) * 0.3:
                score -= 0.15
                factors.append(f"Missing few prerequisites: {', '.join(list(missing_courses)[:2])}")
                details["courses_met"] = False
            else:
                score -= 0.3
                factors.append(f"Missing multiple prerequisites: {', '.join(list(missing_courses)[:3])}")
                details["courses_met"] = False
        
        return DimensionScore(
            dimension=MatchDimension.REQUIREMENTS.value,
            score=max(score, 0.0),
            weight=self.default_weights[MatchDimension.REQUIREMENTS],
            details=details,
            contributing_factors=factors
        )
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        try:
            embeddings = self.embedding_model.encode(
                [text1, text2],
                show_progress_bar=False,
                normalize_embeddings=True
            )
            similarity = float(np.dot(embeddings[0], embeddings[1]))
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            print(f"Warning: Similarity computation failed: {e}")
            return 0.0
    
    def compute_overall_score(
        self,
        dimension_scores: Dict[str, DimensionScore],
        custom_weights: Dict[str, float] = None
    ) -> float:
        """
        Compute overall match score
        
        Args:
            dimension_scores: Scores for each dimension
            custom_weights: Custom weights (optional)
        
        Returns:
            Weighted total score (0-1)
        """
        weights = custom_weights if custom_weights else self.default_weights
        
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, dim_score in dimension_scores.items():
            weight = weights.get(dimension, dim_score.weight)
            total_score += dim_score.score * weight
            total_weight += weight
        
        # Normalize
        return total_score / total_weight if total_weight > 0 else 0.0