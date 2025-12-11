"""
Multi-dimensional scoring algorithms for program matching V2

This scorer is designed for the new V2 dataset with richer information:
- Courses with descriptions for curriculum alignment
- Nested extracted_fields structure
- Application requirements with detailed fields
- Training outcomes with career paths
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import re
from sentence_transformers import SentenceTransformer

from .models_v2 import (
    DimensionScoreV2, MatchDimensionV2, ProgramDataV2,
    CourseInfo, ApplicationRequirementsV2, TrainingOutcomesV2
)


class DimensionScorerV2:
    """Multi-dimensional scorer for V2 dataset"""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize scorer
        
        Args:
            embedding_model_name: Sentence-BERT model name
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Updated weight configuration for V2
        self.default_weights = {
            MatchDimensionV2.ACADEMIC: 0.25,
            MatchDimensionV2.SKILLS: 0.20,
            MatchDimensionV2.EXPERIENCE: 0.15,
            MatchDimensionV2.GOALS: 0.20,
            MatchDimensionV2.REQUIREMENTS: 0.10,
            MatchDimensionV2.CURRICULUM: 0.10  # New dimension
        }
        
        # Precomputed embeddings cache
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching"""
        if not text:
            return np.zeros(384)  # Default embedding size for all-MiniLM-L6-v2
        
        text_hash = hash(text[:500])  # Hash first 500 chars
        if text_hash not in self._embedding_cache:
            self._embedding_cache[text_hash] = self.embedding_model.encode(text)
        return self._embedding_cache[text_hash]
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def score_academic(
        self,
        profile: Dict[str, Any],
        program: ProgramDataV2
    ) -> DimensionScoreV2:
        """
        Evaluate academic background match for V2
        
        Factors:
        1. GPA match (35%)
        2. Major/Field relevance (35%)
        3. Coursework alignment (30%)
        """
        score = 0.0
        details = {}
        factors = []
        matched_items = []
        missing_items = []
        
        # 1. GPA match
        student_gpa = profile.get("gpa", 0.0)
        if student_gpa:
            # V2 doesn't always have explicit GPA requirements
            # Assume 3.0 as baseline unless we can extract it
            required_gpa = self._extract_gpa_requirement(program)
            
            if student_gpa >= required_gpa + 0.3:
                gpa_score = 1.0
                factors.append(f"GPA {student_gpa:.2f} significantly exceeds typical requirement")
                matched_items.append(f"GPA: {student_gpa:.2f}")
            elif student_gpa >= required_gpa:
                gpa_score = 0.8
                factors.append(f"GPA {student_gpa:.2f} meets typical requirement")
                matched_items.append(f"GPA: {student_gpa:.2f}")
            elif student_gpa >= required_gpa - 0.2:
                gpa_score = 0.5
                factors.append(f"GPA {student_gpa:.2f} slightly below typical requirement")
            else:
                gpa_score = 0.2
                factors.append(f"GPA {student_gpa:.2f} below typical requirement")
                missing_items.append(f"Higher GPA recommended (typically {required_gpa}+)")
            
            score += gpa_score * 0.35
            details["gpa_score"] = gpa_score
            details["student_gpa"] = student_gpa
            details["inferred_required_gpa"] = required_gpa
        else:
            score += 0.25  # Partial score when GPA not provided
        
        # 2. Major/Field relevance
        student_major = profile.get("major", "")
        program_field = self._infer_program_field(program)
        
        if student_major and program_field:
            major_similarity = self._compute_text_similarity(student_major, program_field)
            
            # Also check background requirements
            bg_text = program.application_requirements.academic_background or ""
            if bg_text:
                bg_similarity = self._compute_text_similarity(student_major, bg_text)
                major_similarity = max(major_similarity, bg_similarity * 0.8)
            
            score += major_similarity * 0.35
            details["major_similarity"] = major_similarity
            
            if major_similarity > 0.7:
                factors.append(f"Major '{student_major}' highly relevant to program")
                matched_items.append(f"Major: {student_major}")
            elif major_similarity > 0.4:
                factors.append(f"Major background somewhat relevant")
            else:
                factors.append(f"Major background may require additional preparation")
                missing_items.append("Consider strengthening relevant background")
        else:
            score += 0.2
        
        # 3. Coursework alignment with program courses
        student_courses = set(c.lower() for c in profile.get("courses", []))
        program_courses = [c.name.lower() for c in program.courses if c.name]
        
        if student_courses and program_courses:
            # Semantic matching for courses
            course_matches = []
            for s_course in student_courses:
                for p_course in program_courses:
                    sim = self._compute_text_similarity(s_course, p_course)
                    if sim > 0.6:
                        course_matches.append((s_course, p_course, sim))
            
            course_score = min(len(course_matches) / max(len(program_courses), 5), 1.0)
            score += course_score * 0.30
            details["course_match_count"] = len(course_matches)
            
            if course_matches:
                matched_items.extend([f"Course: {m[0]}" for m in course_matches[:3]])
                factors.append(f"Completed {len(course_matches)} relevant courses")
        else:
            score += 0.15
        
        return DimensionScoreV2(
            dimension=MatchDimensionV2.ACADEMIC.value,
            score=min(score, 1.0),
            weight=self.default_weights[MatchDimensionV2.ACADEMIC],
            details=details,
            contributing_factors=factors,
            matched_items=matched_items,
            missing_items=missing_items
        )
    
    def score_skills(
        self,
        profile: Dict[str, Any],
        program: ProgramDataV2
    ) -> DimensionScoreV2:
        """
        Evaluate skills match for V2
        
        Factors:
        1. Required skills coverage (50%)
        2. Skills from course descriptions (30%)
        3. Skill breadth bonus (20%)
        """
        score = 0.0
        details = {}
        factors = []
        matched_items = []
        missing_items = []
        
        student_skills = set(s.lower().strip() for s in profile.get("skills", []))
        
        # Extract required skills from V2 data
        required_skills = self._extract_required_skills(program)
        
        if not required_skills:
            # Use semantic matching with program text
            program_text = program.get_full_text()
            if program_text and student_skills:
                skill_relevance = sum(
                    1 for skill in student_skills
                    if skill in program_text.lower()
                ) / max(len(student_skills), 1)
                score = skill_relevance * 0.6 + 0.2
                factors.append(f"Skills relevance to program: {skill_relevance:.1%}")
            else:
                score = 0.5
                factors.append("Program has no explicit skill requirements")
        else:
            # 1. Required skills coverage
            overlap = student_skills & required_skills
            missing = required_skills - student_skills
            
            coverage_ratio = len(overlap) / len(required_skills)
            score += coverage_ratio * 0.5
            
            details["skill_coverage"] = coverage_ratio
            details["matched_skills"] = list(overlap)
            details["missing_skills"] = list(missing)
            
            matched_items.extend([f"Skill: {s}" for s in list(overlap)[:5]])
            missing_items.extend([f"Skill: {s}" for s in list(missing)[:3]])
            
            if coverage_ratio >= 0.8:
                factors.append(f"Possess {len(overlap)}/{len(required_skills)} required skills")
            elif coverage_ratio >= 0.5:
                factors.append(f"Possess most required skills ({len(overlap)}/{len(required_skills)})")
            else:
                factors.append(f"Missing key skills: {', '.join(list(missing)[:3])}")
            
            # 2. Skills from course descriptions
            course_skill_match = self._match_skills_to_courses(student_skills, program)
            score += course_skill_match * 0.3
            
            # 3. Skill breadth bonus
            extra_skills = student_skills - required_skills
            if extra_skills:
                breadth_score = min(len(extra_skills) / 10, 1.0) * 0.2
                score += breadth_score
                factors.append(f"Additionally possess {len(extra_skills)} related skills")
        
        return DimensionScoreV2(
            dimension=MatchDimensionV2.SKILLS.value,
            score=min(score, 1.0),
            weight=self.default_weights[MatchDimensionV2.SKILLS],
            details=details,
            contributing_factors=factors,
            matched_items=matched_items,
            missing_items=missing_items
        )
    
    def score_experience(
        self,
        profile: Dict[str, Any],
        program: ProgramDataV2
    ) -> DimensionScoreV2:
        """
        Evaluate experience match for V2
        
        Factors:
        1. Experience quantity (25%)
        2. Experience relevance to program (45%)
        3. Research vs Professional orientation match (30%)
        """
        score = 0.0
        details = {}
        factors = []
        matched_items = []
        missing_items = []
        
        experiences = profile.get("experiences", [])
        
        # 1. Experience quantity
        exp_count = len(experiences)
        quantity_score = min(exp_count / 3, 1.0)  # 3+ experiences = full score
        score += quantity_score * 0.25
        details["experience_count"] = exp_count
        
        if exp_count == 0:
            factors.append("No work experience provided")
            missing_items.append("Relevant work/research experience")
        elif exp_count >= 2:
            factors.append(f"Has {exp_count} relevant experiences")
        
        # 2. Experience relevance
        if experiences:
            program_text = program.get_full_text()[:2000]
            exp_descriptions = []
            for exp in experiences:
                desc = f"{exp.get('title', '')} at {exp.get('org', '')}: {exp.get('impact', '')}"
                exp_descriptions.append(desc)
            
            combined_exp = " ".join(exp_descriptions)
            relevance = self._compute_text_similarity(combined_exp, program_text)
            score += relevance * 0.45
            details["experience_relevance"] = relevance
            
            if relevance > 0.6:
                factors.append("Experience highly relevant to program focus")
                matched_items.append(f"Relevant experience in field")
            elif relevance > 0.3:
                factors.append("Experience somewhat relevant to program")
        
        # 3. Research vs Professional orientation match
        outcomes = program.training_outcomes
        has_research_exp = any(
            "research" in str(exp).lower() or "lab" in str(exp).lower()
            for exp in experiences
        )
        has_professional_exp = any(
            "intern" in str(exp).lower() or "engineer" in str(exp).lower() or
            "analyst" in str(exp).lower()
            for exp in experiences
        )
        
        research_oriented = bool(outcomes.research_orientation)
        professional_oriented = bool(outcomes.professional_orientation)
        
        orientation_match = 0.5  # Default
        if research_oriented and has_research_exp:
            orientation_match = 1.0
            factors.append("Research experience aligns with program's research focus")
            matched_items.append("Research experience")
        elif professional_oriented and has_professional_exp:
            orientation_match = 1.0
            factors.append("Professional experience aligns with program's career focus")
            matched_items.append("Professional experience")
        elif research_oriented and not has_research_exp:
            factors.append("Program emphasizes research; consider gaining research experience")
            missing_items.append("Research experience")
        elif professional_oriented and not has_professional_exp:
            factors.append("Program emphasizes industry preparation; consider internships")
            missing_items.append("Industry experience")
        
        score += orientation_match * 0.30
        details["orientation_match"] = orientation_match
        
        return DimensionScoreV2(
            dimension=MatchDimensionV2.EXPERIENCE.value,
            score=min(score, 1.0),
            weight=self.default_weights[MatchDimensionV2.EXPERIENCE],
            details=details,
            contributing_factors=factors,
            matched_items=matched_items,
            missing_items=missing_items
        )
    
    def score_goals(
        self,
        profile: Dict[str, Any],
        program: ProgramDataV2
    ) -> DimensionScoreV2:
        """
        Evaluate goals alignment for V2
        
        Factors:
        1. Career goals alignment (50%)
        2. Mission/values alignment (30%)
        3. Career path match (20%)
        """
        score = 0.0
        details = {}
        factors = []
        matched_items = []
        missing_items = []
        
        student_goals = profile.get("goals", "")
        
        # 1. Career goals alignment with training outcomes
        outcomes = program.training_outcomes
        program_goals_text = " ".join(filter(None, [
            outcomes.goals,
            outcomes.career_paths,
            outcomes.summary
        ]))
        
        if student_goals and program_goals_text:
            goals_similarity = self._compute_text_similarity(student_goals, program_goals_text)
            score += goals_similarity * 0.5
            details["goals_similarity"] = goals_similarity
            
            if goals_similarity > 0.6:
                factors.append("Career goals strongly align with program outcomes")
                matched_items.append("Career goals alignment")
            elif goals_similarity > 0.3:
                factors.append("Career goals partially align with program outcomes")
            else:
                factors.append("Career goals may benefit from more alignment with program focus")
        else:
            score += 0.3
        
        # 2. Mission alignment
        background = program.program_background
        mission_text = " ".join(filter(None, [
            background.mission,
            background.summary,
            background.environment
        ]))
        
        if student_goals and mission_text:
            mission_sim = self._compute_text_similarity(student_goals, mission_text)
            score += mission_sim * 0.3
            details["mission_similarity"] = mission_sim
            
            if mission_sim > 0.5:
                factors.append("Values align with program mission")
                matched_items.append("Program mission alignment")
        else:
            score += 0.2
        
        # 3. Career path match
        career_paths = outcomes.career_paths or ""
        if student_goals and career_paths:
            career_sim = self._compute_text_similarity(student_goals, career_paths)
            score += career_sim * 0.2
            details["career_path_similarity"] = career_sim
            
            if career_sim > 0.5:
                factors.append("Career path matches program's typical outcomes")
        else:
            score += 0.15
        
        return DimensionScoreV2(
            dimension=MatchDimensionV2.GOALS.value,
            score=min(score, 1.0),
            weight=self.default_weights[MatchDimensionV2.GOALS],
            details=details,
            contributing_factors=factors,
            matched_items=matched_items,
            missing_items=missing_items
        )
    
    def score_requirements(
        self,
        profile: Dict[str, Any],
        program: ProgramDataV2
    ) -> DimensionScoreV2:
        """
        Evaluate application requirements compliance for V2
        
        Factors:
        1. Prerequisites match (40%)
        2. Test scores (GRE/TOEFL) (30%)
        3. Documents readiness (30%)
        """
        score = 0.0
        details = {}
        factors = []
        matched_items = []
        missing_items = []
        
        reqs = program.application_requirements
        
        # 1. Prerequisites
        prerequisites = reqs.prerequisites or ""
        if prerequisites:
            student_background = f"{profile.get('major', '')} {' '.join(profile.get('courses', []))}"
            prereq_match = self._compute_text_similarity(student_background, prerequisites)
            score += prereq_match * 0.4
            details["prerequisites_match"] = prereq_match
            
            if prereq_match > 0.6:
                factors.append("Background meets prerequisite requirements")
                matched_items.append("Prerequisites met")
            elif prereq_match > 0.3:
                factors.append("Background partially meets prerequisites")
            else:
                factors.append("May need additional preparation for prerequisites")
                missing_items.append("Review prerequisite requirements")
        else:
            score += 0.35  # No explicit prerequisites
            factors.append("No specific prerequisites listed")
        
        # 2. Test scores
        gre_req = reqs.gre or ""
        english_req = reqs.english_tests or ""
        
        test_score = 0.5  # Default assumption
        if "not required" in gre_req.lower() or "waived" in gre_req.lower():
            test_score = 0.8
            factors.append("GRE not required")
            matched_items.append("GRE waived")
        elif gre_req:
            factors.append(f"GRE requirement: {gre_req[:50]}...")
            missing_items.append("GRE scores")
        
        if english_req:
            factors.append(f"English test required: {english_req[:30]}...")
        
        score += test_score * 0.3
        details["test_requirements_handled"] = test_score
        
        # 3. Documents readiness (assume partial readiness)
        docs = reqs.documents or ""
        if docs:
            doc_score = 0.6  # Assume moderate readiness
            factors.append(f"Required documents: {docs[:50]}...")
        else:
            doc_score = 0.7
            factors.append("Standard application documents expected")
        
        score += doc_score * 0.3
        details["documents_score"] = doc_score
        
        return DimensionScoreV2(
            dimension=MatchDimensionV2.REQUIREMENTS.value,
            score=min(score, 1.0),
            weight=self.default_weights[MatchDimensionV2.REQUIREMENTS],
            details=details,
            contributing_factors=factors,
            matched_items=matched_items,
            missing_items=missing_items
        )
    
    def score_curriculum(
        self,
        profile: Dict[str, Any],
        program: ProgramDataV2
    ) -> DimensionScoreV2:
        """
        Evaluate curriculum alignment for V2 (new dimension)
        
        Factors:
        1. Course interest alignment (50%)
        2. Course level appropriateness (30%)
        3. Course diversity match (20%)
        """
        score = 0.0
        details = {}
        factors = []
        matched_items = []
        missing_items = []
        
        student_goals = profile.get("goals", "")
        student_skills = profile.get("skills", [])
        student_courses = profile.get("courses", [])
        
        program_courses = program.courses
        
        if not program_courses:
            # No curriculum data available
            return DimensionScoreV2(
                dimension=MatchDimensionV2.CURRICULUM.value,
                score=0.5,  # Neutral score
                weight=self.default_weights[MatchDimensionV2.CURRICULUM],
                details={"note": "No curriculum data available"},
                contributing_factors=["Curriculum details not available for analysis"],
                matched_items=[],
                missing_items=[]
            )
        
        # 1. Course interest alignment
        course_texts = [
            f"{c.name} {c.description or ''}" for c in program_courses
        ]
        combined_courses = " ".join(course_texts)
        
        interest_text = f"{student_goals} {' '.join(student_skills)}"
        if interest_text.strip():
            interest_sim = self._compute_text_similarity(interest_text, combined_courses)
            score += interest_sim * 0.5
            details["interest_alignment"] = interest_sim
            
            if interest_sim > 0.6:
                factors.append("Curriculum strongly aligns with your interests")
                matched_items.append("Curriculum interest match")
            elif interest_sim > 0.3:
                factors.append("Curriculum partially aligns with interests")
        else:
            score += 0.25
        
        # 2. Find specifically relevant courses
        relevant_courses = []
        for course in program_courses:
            course_text = f"{course.name} {course.description or ''}"
            for skill in student_skills:
                if skill.lower() in course_text.lower():
                    relevant_courses.append(course.name)
                    break
        
        if relevant_courses:
            relevance_score = min(len(relevant_courses) / 5, 1.0)
            score += relevance_score * 0.3
            factors.append(f"Found {len(relevant_courses)} courses matching your skills")
            matched_items.extend([f"Course: {c}" for c in relevant_courses[:3]])
            details["relevant_courses"] = relevant_courses
        else:
            score += 0.15
        
        # 3. Course diversity
        course_topics = set()
        topic_keywords = ["machine learning", "data", "statistics", "programming",
                        "research", "ethics", "visualization", "analysis", "systems"]
        for course in program_courses:
            course_lower = f"{course.name} {course.description or ''}".lower()
            for topic in topic_keywords:
                if topic in course_lower:
                    course_topics.add(topic)
        
        diversity_score = min(len(course_topics) / 5, 1.0)
        score += diversity_score * 0.2
        details["curriculum_diversity"] = len(course_topics)
        
        if diversity_score > 0.6:
            factors.append("Program offers diverse curriculum")
        
        return DimensionScoreV2(
            dimension=MatchDimensionV2.CURRICULUM.value,
            score=min(score, 1.0),
            weight=self.default_weights[MatchDimensionV2.CURRICULUM],
            details=details,
            contributing_factors=factors,
            matched_items=matched_items,
            missing_items=missing_items
        )
    
    def compute_overall_score(
        self,
        dimension_scores: Dict[str, DimensionScoreV2],
        custom_weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute weighted overall score from dimension scores
        
        Args:
            dimension_scores: Dictionary of dimension scores
            custom_weights: Optional custom weights
        
        Returns:
            Overall score between 0 and 1
        """
        weights = custom_weights or {}
        total_weight = 0.0
        weighted_sum = 0.0
        
        for dim, score in dimension_scores.items():
            weight = weights.get(dim, score.weight)
            weighted_sum += score.score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def score_all_dimensions(
        self,
        profile: Dict[str, Any],
        program: ProgramDataV2
    ) -> Dict[str, DimensionScoreV2]:
        """
        Score all dimensions for a profile-program pair
        
        Args:
            profile: Student profile
            program: Program data
        
        Returns:
            Dictionary of dimension scores
        """
        return {
            MatchDimensionV2.ACADEMIC.value: self.score_academic(profile, program),
            MatchDimensionV2.SKILLS.value: self.score_skills(profile, program),
            MatchDimensionV2.EXPERIENCE.value: self.score_experience(profile, program),
            MatchDimensionV2.GOALS.value: self.score_goals(profile, program),
            MatchDimensionV2.REQUIREMENTS.value: self.score_requirements(profile, program),
            MatchDimensionV2.CURRICULUM.value: self.score_curriculum(profile, program),
        }
    
    # Helper methods
    
    def _extract_gpa_requirement(self, program: ProgramDataV2) -> float:
        """Extract GPA requirement from program data"""
        # Check application requirements
        reqs = program.application_requirements
        text_to_check = " ".join(filter(None, [
            reqs.academic_background,
            reqs.prerequisites,
            reqs.summary,
            program.raw_text[:1000]
        ]))
        
        # Look for GPA patterns
        gpa_patterns = [
            r"(\d\.\d+)\s*GPA",
            r"GPA\s*(?:of\s+)?(\d\.\d+)",
            r"minimum\s+(?:GPA\s+)?(?:of\s+)?(\d\.\d+)",
        ]
        
        for pattern in gpa_patterns:
            match = re.search(pattern, text_to_check, re.IGNORECASE)
            if match:
                try:
                    gpa = float(match.group(1))
                    if 2.0 <= gpa <= 4.0:
                        return gpa
                except ValueError:
                    continue
        
        return 3.0  # Default assumption
    
    def _infer_program_field(self, program: ProgramDataV2) -> str:
        """Infer program field from available data"""
        name = program.get_display_name().lower()
        department = (program.department or "").lower()
        text = program.raw_text[:500].lower() if program.raw_text else ""
        
        combined = f"{name} {department} {text}"
        
        field_keywords = {
            "Data Science": ["data science", "analytics", "data engineering"],
            "Computer Science": ["computer science", "computing", "software", "cs "],
            "Machine Learning": ["machine learning", "ml", "deep learning", "ai", "artificial intelligence"],
            "Business": ["business", "mba", "management"],
            "Statistics": ["statistics", "biostatistics", "statistical"],
            "Engineering": ["engineering", "electrical", "mechanical"],
            "Public Health": ["public health", "mph", "epidemiology"],
            "Economics": ["economics", "econometrics"],
            "Biomedical": ["biomedical", "biotechnology", "bioinformatics"],
        }
        
        for field, keywords in field_keywords.items():
            if any(kw in combined for kw in keywords):
                return field
        
        return "Graduate Studies"
    
    def _extract_required_skills(self, program: ProgramDataV2) -> set:
        """Extract required skills from program data"""
        skills = set()
        
        # Common technical skills to look for
        skill_patterns = [
            "python", "r programming", "sql", "java", "c++",
            "machine learning", "deep learning", "statistics",
            "data analysis", "data visualization", "programming",
            "linear algebra", "calculus", "probability",
            "tensorflow", "pytorch", "tableau", "excel",
            "research", "communication", "teamwork"
        ]
        
        # Search in various text fields
        text_to_search = " ".join(filter(None, [
            program.application_requirements.prerequisites,
            program.application_requirements.academic_background,
            program.get_full_text()[:2000]
        ])).lower()
        
        for skill in skill_patterns:
            if skill in text_to_search:
                skills.add(skill)
        
        return skills
    
    def _match_skills_to_courses(self, student_skills: set, program: ProgramDataV2) -> float:
        """Calculate skill-course alignment score"""
        if not student_skills or not program.courses:
            return 0.0
        
        matches = 0
        total = len(program.courses)
        
        for course in program.courses:
            course_text = f"{course.name} {course.description or ''}".lower()
            for skill in student_skills:
                if skill in course_text:
                    matches += 1
                    break
        
        return min(matches / max(total, 3), 1.0)
