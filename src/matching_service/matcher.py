"""
Core program matching engine
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import (
    ProgramMatch, MatchingResult, MatchLevel,
    DimensionScore, ProgramData
)
from .scorer import DimensionScorer

# Import LLM utilities for generating fit reasons
try:
    from ..writing_agent.llm_utils import get_llm, call_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: LLM utilities not available. Using rule-based fit reasons.")


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
        
        Supports multiple corpus formats:
        1. New format with "school", "program", "sections"
        2. Legacy format with "extracted_fields", "raw_text"
        
        Args:
            data: Raw JSON data
            program_id: Program identifier
        
        Returns:
            Standardized program dictionary or None
        """
        try:
            # Detect corpus format and extract fields accordingly
            
            # NEW FORMAT: Uses "school", "program", "sections" structure
            if "school" in data and "program" in data:
                return self._parse_new_format(data, program_id)
            
            # LEGACY FORMAT: Uses "extracted_fields", "raw_text" structure
            else:
                return self._parse_legacy_format(data, program_id)
            
        except Exception as e:
            print(f"Warning: Error parsing program {program_id}: {e}")
            return None
    
    def _parse_new_format(self, data: Dict[str, Any], program_id: str) -> Optional[Dict[str, Any]]:
        """
        Parse new corpus format with school, program, sections structure
        
        Example JSON:
        {
            "id": "columbia-mscs-00",
            "school": "Columbia University",
            "program": "Master of Science in Computer Science (MS)",
            "source_url": "...",
            "sections": [
                {"type": "mission", "text": "..."},
                {"type": "curriculum", "items": [...], "text": "..."},
                {"type": "requirements", "items": [...], "text": "..."},
                {"type": "outcomes", "text": "..."}
            ]
        }
        """
        program_name = data.get("program", "Unknown Program")
        university = data.get("school", "Unknown University")
        source_url = data.get("source_url", "")
        
        # Parse sections to build description and extract info
        sections = data.get("sections", [])
        description_parts = []
        core_courses = []
        requirements_text = ""
        outcomes_text = ""
        focus_areas = []
        
        for section in sections:
            section_type = section.get("type", "")
            section_text = section.get("text", "")
            section_items = section.get("items", [])
            
            if section_type == "mission":
                description_parts.insert(0, section_text)  # Mission goes first
                # Extract focus areas from mission
                focus_areas.extend(self._extract_focus_areas(section_text))
                
            elif section_type == "curriculum":
                description_parts.append(f"Curriculum: {section_text}")
                # Extract courses from items
                for item in section_items:
                    if isinstance(item, str):
                        core_courses.append(item)
                        # Also extract focus areas from curriculum items
                        focus_areas.extend(self._extract_focus_areas(item))
                        
            elif section_type == "requirements":
                requirements_text = section_text
                description_parts.append(f"Requirements: {section_text}")
                for item in section_items:
                    if isinstance(item, str):
                        description_parts.append(f"- {item}")
                        
            elif section_type == "outcomes":
                outcomes_text = section_text
                description_parts.append(f"Career Outcomes: {section_text}")
            
            else:
                # Handle any other section types
                if section_text:
                    description_parts.append(section_text)
        
        # Build full description
        description_text = "\n\n".join(description_parts)
        
        # Extract skills from full description
        required_skills = self._extract_skills(description_text)
        
        # Deduplicate focus areas
        focus_areas = list(dict.fromkeys(focus_areas))[:8]
        
        # Infer field from program name and description
        field = self._infer_field(program_name, description_text)
        
        program_data = {
            "program_id": program_id,
            "name": program_name,
            "university": university,
            "field": field,
            "min_gpa": 3.0,  # Default, could be extracted from requirements
            "required_skills": required_skills,
            "prerequisite_courses": core_courses[:10],  # Limit to avoid noise
            "focus_areas": focus_areas,
            "core_courses": core_courses[:10],
            "career_outcomes": outcomes_text,
            "duration": "",  # Extract if present
            "tuition": None,
            "description_text": description_text,
            "language_requirements": {},
            "source_url": source_url,
            "sections": sections  # Include raw sections for fit_reasons generation
        }
        
        return program_data
    
    def _parse_legacy_format(self, data: Dict[str, Any], program_id: str) -> Optional[Dict[str, Any]]:
        """Parse legacy corpus format with extracted_fields structure"""
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
            "field": self._infer_field(program_name, description_text),
            "min_gpa": 3.0,
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
    
    def _infer_field(self, program_name: str, description: str) -> str:
        """Infer the program field from name and description"""
        combined_text = f"{program_name} {description}".lower()
        
        field_keywords = {
            "Data Science": ["data science", "data analytics", "analytics", "data engineering"],
            "Computer Science": ["computer science", "computing", "software", "algorithms", "cs ", "mscs"],
            "Machine Learning": ["machine learning", "ml", "deep learning", "artificial intelligence", "ai"],
            "Business Analytics": ["business analytics", "mba", "business administration"],
            "Statistics": ["statistics", "biostatistics", "statistical"],
            "Engineering": ["engineering", "electrical", "mechanical", "civil", "aerospace"],
            "Public Health": ["public health", "mph", "epidemiology", "health"],
            "Economics": ["economics", "econometrics"],
            "Public Policy": ["public policy", "mpp", "public administration", "mpa"],
            "Biomedical": ["biomedical", "biotechnology", "bioinformatics"],
            "Finance": ["finance", "mfin", "financial"],
        }
        
        for field, keywords in field_keywords.items():
            if any(kw in combined_text for kw in keywords):
                return field
        
        return "Graduate Studies"  # Default fallback
    
    def _extract_university_name(self, url: str, program_id: str) -> str:
        """Extract university name from URL or program ID"""
        # First check program_id for common patterns (most reliable)
        program_id_lower = program_id.lower()
        url_lower = url.lower()
        combined = f"{program_id_lower} {url_lower}"
        
        university_patterns = {
            "stanford": "Stanford University",
            "mit": "MIT",
            "cmu": "Carnegie Mellon University",
            "berkeley": "UC Berkeley",
            "ucb": "UC Berkeley",
            "columbia": "Columbia University",
            "harvard": "Harvard University",
            "yale": "Yale University",
            "princeton": "Princeton University",
            "cornell": "Cornell University",
            "upenn": "University of Pennsylvania",
            "penn": "University of Pennsylvania",
            "caltech": "California Institute of Technology",
            "duke": "Duke University",
            "northwestern": "Northwestern University",
            "brown": "Brown University",
            "dartmouth": "Dartmouth College",
            "johns": "Johns Hopkins University",  # Johns Hopkins often starts with "Johns-"
            "hopkins": "Johns Hopkins University",
            "chicago": "University of Chicago",
            "ucla": "UCLA",
            "usc": "University of Southern California",
            "nyu": "New York University",
            "gatech": "Georgia Institute of Technology",
            "georgia": "Georgia Institute of Technology",
            "umich": "University of Michigan",
            "michigan": "University of Michigan",
            "rice": "Rice University",
            "vanderbilt": "Vanderbilt University",
            "emory": "Emory University",
            "washu": "Washington University in St. Louis",
            "wustl": "Washington University in St. Louis",
        }
        
        for pattern, university in university_patterns.items():
            if pattern in combined:
                return university
        
        # Try to extract domain from URL
        if "://" in url:
            domain = url.split("://")[1].split("/")[0]
            return domain.replace("www.", "").split(".")[0].title()
        
        return "Unknown University"
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract technical skills from program text"""
        common_skills = [
            "Python", "R", "SQL", "Machine Learning", "Deep Learning",
            "Statistics", "Data Visualization", "Big Data", "Algorithms",
            "TensorFlow", "PyTorch", "NLP", "Computer Vision",
            "Java", "C++", "JavaScript", "Hadoop", "Spark", "AWS",
            "Natural Language Processing", "Neural Networks", "Regression",
            "Classification", "Clustering", "Optimization", "Linear Algebra",
            "Calculus", "Probability", "Bayesian", "Time Series",
            "Data Mining", "Feature Engineering", "Model Deployment",
            "Docker", "Kubernetes", "Cloud Computing", "ETL",
            "Tableau", "Power BI", "Excel", "Git"
        ]
        
        text_lower = text.lower()
        found_skills = []
        
        for skill in common_skills:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        return found_skills[:15]  # Return top 15
    
    def _extract_focus_areas(self, text: str) -> List[str]:
        """Extract program focus areas from text"""
        focus_keywords = [
            "machine learning", "artificial intelligence", "data analytics",
            "statistics", "computer vision", "natural language processing",
            "big data", "data engineering", "business analytics",
            "deep learning", "neural networks", "reinforcement learning",
            "robotics", "computer systems", "security", "cryptography",
            "databases", "distributed systems", "cloud computing",
            "software engineering", "algorithms", "theory",
            "biostatistics", "bioinformatics", "computational biology",
            "financial engineering", "quantitative finance",
            "healthcare analytics", "public policy", "economics",
            "operations research", "optimization", "simulation",
            "human-computer interaction", "visualization",
            "nlp", "ai", "ml", "cv"
        ]
        
        text_lower = text.lower()
        found_areas = []
        
        for area in focus_keywords:
            if area in text_lower:
                # Normalize some abbreviations
                if area == "nlp":
                    area = "natural language processing"
                elif area == "ai":
                    area = "artificial intelligence"
                elif area == "ml":
                    area = "machine learning"
                elif area == "cv":
                    area = "computer vision"
                
                # Capitalize for display
                area_display = area.title()
                if area_display not in found_areas:
                    found_areas.append(area_display)
        
        return found_areas[:8]  # Return top 8
    
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
            
            # Fit reasons will be generated later (only for top matches)
            fit_reasons = []
            
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
                fit_reasons=fit_reasons,  # Will be populated for top matches
                recommendations=recommendations,
                explanation=explanation,
                metadata={
                    "duration": program_data.get("duration"),
                    "tuition": program_data.get("tuition"),
                    "source_url": program_data.get("source_url"),
                    "program_data": program_data  # Store for later LLM generation
                }
            )
            
            matches.append(match)
        
        # Sort by overall score
        matches.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Take top K
        top_matches = matches[:top_k]
        
        # Generate LLM fit reasons ONLY for top 5 matches (for speed)
        # Use parallel execution for faster response
        llm_limit = min(5, len(top_matches))
        
        def generate_fit_for_match(match_idx: int) -> Tuple[int, List[str]]:
            """Helper function for parallel LLM calls"""
            match = top_matches[match_idx]
            program_data = match.metadata.get("program_data", {})
            if program_data:
                reasons = self._generate_fit_reasons(profile, program_data, match.dimension_scores)
                return (match_idx, reasons)
            return (match_idx, [])
        
        # Execute LLM calls in parallel using ThreadPoolExecutor
        if llm_limit > 0 and LLM_AVAILABLE:
            with ThreadPoolExecutor(max_workers=llm_limit) as executor:
                futures = {executor.submit(generate_fit_for_match, i): i for i in range(llm_limit)}
                for future in as_completed(futures):
                    try:
                        match_idx, reasons = future.result()
                        top_matches[match_idx].fit_reasons = reasons
                    except Exception as e:
                        print(f"Warning: Parallel LLM call failed: {e}")
                        # Fall back to rule-based for this match
                        match_idx = futures[future]
                        program_data = top_matches[match_idx].metadata.get("program_data", {})
                        if program_data:
                            top_matches[match_idx].fit_reasons = self._generate_fit_reasons_rule_based(
                                profile, program_data, top_matches[match_idx].dimension_scores
                            )
        
        # For remaining matches (if any), use rule-based fit reasons
        for match in top_matches[llm_limit:]:
            program_data = match.metadata.get("program_data", {})
            if program_data:
                match.fit_reasons = self._generate_fit_reasons_rule_based(profile, program_data, match.dimension_scores)
        
        # Clean up metadata (remove program_data to reduce response size)
        for match in top_matches:
            if "program_data" in match.metadata:
                del match.metadata["program_data"]
        
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
    
    def _generate_fit_reasons(
        self,
        profile: Dict[str, Any],
        program: Dict[str, Any],
        scores: Dict[str, DimensionScore]
    ) -> List[str]:
        """
        Generate personalized 'Why This Program Fits You' reasons
        using LLM (GPT) for intelligent, natural explanations.
        Falls back to rule-based if LLM is unavailable.
        """
        # Try LLM-based generation first
        if LLM_AVAILABLE:
            try:
                return self._generate_fit_reasons_llm(profile, program, scores)
            except Exception as e:
                print(f"Warning: LLM fit reason generation failed: {e}")
                # Fall back to rule-based
        
        return self._generate_fit_reasons_rule_based(profile, program, scores)
    
    def _generate_fit_reasons_llm(
        self,
        profile: Dict[str, Any],
        program: Dict[str, Any],
        scores: Dict[str, DimensionScore]
    ) -> List[str]:
        """
        Generate fit reasons using LLM (GPT-3.5-turbo for speed).
        """
        # Get LLM instance - use gpt-3.5-turbo for faster response
        llm = get_llm(provider="openai", model_name="gpt-3.5-turbo", temperature=0.7)
        
        # Extract program info
        program_name = program.get("name", "this program")
        university = program.get("university", "")
        
        # Get program sections
        sections = program.get("sections", [])
        mission_text = ""
        curriculum_text = ""
        outcomes_text = ""
        
        for section in sections:
            section_type = section.get("type", "")
            section_text = section.get("text", "")
            if section_type == "mission":
                mission_text = section_text
            elif section_type == "curriculum":
                curriculum_text = section_text
            elif section_type == "outcomes":
                outcomes_text = section_text
        
        focus_areas = program.get("focus_areas", [])
        
        # Format student experiences
        exp_summary = ""
        if profile.get("experiences"):
            exp_list = []
            for exp in profile.get("experiences", [])[:3]:
                title = exp.get("title", "")
                org = exp.get("org", "")
                impact = exp.get("impact", "")
                if title:
                    exp_list.append(f"- {title} at {org}: {impact[:100]}" if impact else f"- {title} at {org}")
            exp_summary = "\n".join(exp_list)
        
        # Build the prompt
        prompt = f"""You are an expert graduate school advisor. Based on the student's profile and the program information below, generate 3-4 specific, personalized reasons explaining why this program is a great fit for this student.

STUDENT PROFILE:
- Name: {profile.get('name', 'Student')}
- Major: {profile.get('major', 'Not specified')}
- GPA: {profile.get('gpa', 'Not specified')}
- Skills: {', '.join(profile.get('skills', [])[:8])}
- Career Goals: {profile.get('goals', 'Not specified')[:200]}
- Experiences:
{exp_summary if exp_summary else '  No experience listed'}

PROGRAM INFORMATION:
- Program: {program_name}
- University: {university}
- Mission: {mission_text[:300] if mission_text else 'Not available'}
- Curriculum Focus: {curriculum_text[:300] if curriculum_text else 'Not available'}
- Career Outcomes: {outcomes_text[:300] if outcomes_text else 'Not available'}
- Focus Areas: {', '.join(focus_areas[:5]) if focus_areas else 'Not specified'}

MATCH SCORES:
- Academic: {scores.get('academic', {}).score if scores.get('academic') else 'N/A'}
- Skills: {scores.get('skills', {}).score if scores.get('skills') else 'N/A'}
- Experience: {scores.get('experience', {}).score if scores.get('experience') else 'N/A'}
- Goals: {scores.get('goals', {}).score if scores.get('goals') else 'N/A'}

Generate 3-4 compelling, specific reasons explaining why this program fits this student. Each reason should:
1. Connect the student's specific background/skills/goals to specific program features
2. Be concise (1-2 sentences each)
3. Sound natural and encouraging
4. Reference actual program details when possible

Format your response as a numbered list (1. 2. 3. 4.) with each reason on its own line. Do not include any other text."""

        system_message = "You are an expert graduate school advisor who helps students understand why specific programs match their backgrounds and goals. Be specific, encouraging, and reference actual program features."
        
        # Call LLM
        response = call_llm(llm, prompt, system_message)
        
        # Parse response into list
        fit_reasons = []
        for line in response.strip().split('\n'):
            line = line.strip()
            # Remove numbering (1. 2. 3. etc.)
            if line and line[0].isdigit():
                # Remove "1. " or "1) " prefix
                if '. ' in line[:4]:
                    line = line.split('. ', 1)[1]
                elif ') ' in line[:4]:
                    line = line.split(') ', 1)[1]
            if line and len(line) > 10:  # Filter out empty or too short lines
                fit_reasons.append(line)
        
        return fit_reasons[:4]  # Return top 4 reasons
    
    def _generate_fit_reasons_rule_based(
        self,
        profile: Dict[str, Any],
        program: Dict[str, Any],
        scores: Dict[str, DimensionScore]
    ) -> List[str]:
        """
        Fallback rule-based fit reason generation.
        """
        fit_reasons = []
        
        student_major = profile.get("major", "").lower()
        student_skills = [s.lower() for s in profile.get("skills", [])]
        student_goals = profile.get("goals", "").lower()
        student_experiences = profile.get("experiences", [])
        
        program_name = program.get("name", "this program")
        university = program.get("university", "")
        
        # Get program sections for context
        sections = program.get("sections", [])
        mission_text = ""
        curriculum_text = ""
        outcomes_text = ""
        
        for section in sections:
            section_type = section.get("type", "")
            section_text = section.get("text", "")
            if section_type == "mission":
                mission_text = section_text
            elif section_type == "curriculum":
                curriculum_text = section_text
            elif section_type == "outcomes":
                outcomes_text = section_text
        
        focus_areas = program.get("focus_areas", [])
        
        # 1. Match student major/background with program mission
        if mission_text:
            fit_reasons.append(
                f"Your background in {profile.get('major', 'your field')} aligns with the program's mission: "
                f"\"{mission_text[:120]}...\""
            )
        
        # 2. Match student skills with curriculum
        if curriculum_text and student_skills:
            fit_reasons.append(
                f"Your skills in {', '.join(student_skills[:3])} directly apply to the curriculum, which "
                f"emphasizes: \"{curriculum_text[:100]}...\""
            )
        
        # 3. Focus areas
        if focus_areas:
            fit_reasons.append(
                f"This program focuses on {', '.join(focus_areas[:3])} — areas that complement your background."
            )
        
        # 4. Career outcomes
        if outcomes_text:
            fit_reasons.append(
                f"The program's career outcomes align with your goals: \"{outcomes_text[:120]}...\""
            )
        
        # Ensure we have at least one reason
        if not fit_reasons:
            fit_reasons.append(
                f"{program_name} at {university} offers opportunities to develop skills in your areas of interest."
            )
        
        return fit_reasons[:4]
    
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