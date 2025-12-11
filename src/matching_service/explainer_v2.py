"""
LLM-enhanced explanation generation for V2 program matches

This explainer is specifically designed for the V2 dataset with:
- Rich course information and descriptions
- 6-dimension matching (including curriculum)
- Detailed program background and outcomes
"""

from typing import Dict, Any, Optional, List
import os

# Import LLM utilities
try:
    from ..writing_agent.llm_utils import get_llm, call_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: LLM utilities not available for V2 explainer.")


class MatchExplainerV2:
    """
    Generates detailed, personalized explanations for V2 program matches
    using LLM when available, with fallback to enhanced rule-based generation
    """
    
    def __init__(
        self, 
        llm_provider: str = "openai",
        model_name: str = "gpt-4-turbo-preview",
        use_llm: bool = True
    ):
        """
        Initialize V2 explainer
        
        Args:
            llm_provider: LLM provider (openai, anthropic, qwen)
            model_name: Specific model name
            use_llm: Whether to use LLM for explanations
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.use_llm = use_llm and LLM_AVAILABLE
        self.llm = None
        
        if self.use_llm:
            try:
                self.llm = get_llm(
                    provider=llm_provider,
                    model_name=model_name,
                    temperature=0.7
                )
            except Exception as e:
                print(f"Warning: Failed to initialize LLM for V2 explainer: {e}")
                self.use_llm = False
    
    def generate_fit_reasons(
        self,
        profile: Dict[str, Any],
        program_data: Any,  # ProgramDataV2
        overall_score: float,
        dimension_scores: Dict[str, Any],
        matched_courses: List[str],
        strengths: List[str],
        gaps: List[str]
    ) -> List[str]:
        """
        Generate personalized 'Why This Program Fits You' reasons
        
        Args:
            profile: Student profile dictionary
            program_data: ProgramDataV2 object
            overall_score: Overall match score
            dimension_scores: Dictionary of dimension scores
            matched_courses: List of matched course names
            strengths: List of student strengths
            gaps: List of areas to improve
        
        Returns:
            List of personalized fit reasons
        """
        if self.use_llm:
            try:
                return self._generate_llm_fit_reasons(
                    profile, program_data, overall_score,
                    dimension_scores, matched_courses, strengths, gaps
                )
            except Exception as e:
                print(f"Warning: LLM fit reasons generation failed: {e}")
                # Fall back to rule-based
        
        return self._generate_rule_based_fit_reasons(
            profile, program_data, overall_score,
            dimension_scores, matched_courses, strengths, gaps
        )
    
    def _generate_llm_fit_reasons(
        self,
        profile: Dict[str, Any],
        program_data: Any,
        overall_score: float,
        dimension_scores: Dict[str, Any],
        matched_courses: List[str],
        strengths: List[str],
        gaps: List[str]
    ) -> List[str]:
        """Generate fit reasons using LLM"""
        
        # Extract program information safely
        program_name = getattr(program_data, 'program_name', 'Graduate Program')
        university = getattr(program_data, 'university', 'University')
        
        # Get focus areas
        bg = getattr(program_data, 'program_background', None)
        focus_areas = getattr(bg, 'focus_areas', []) if bg else []
        mission = getattr(bg, 'mission', '') if bg else ''
        
        # Get outcomes
        outcomes = getattr(program_data, 'training_outcomes', None)
        career_paths = getattr(outcomes, 'career_paths', []) if outcomes else []
        
        # Format dimension scores
        dim_scores_text = []
        for dim_name, score_obj in dimension_scores.items():
            score_val = getattr(score_obj, 'score', 0) if hasattr(score_obj, 'score') else score_obj
            dim_scores_text.append(f"- {dim_name.replace('_', ' ').title()}: {score_val:.0%}")
        
        # Format courses
        courses_text = ', '.join(matched_courses[:5]) if matched_courses else "N/A"
        
        prompt = f"""Generate 4 personalized, compelling reasons explaining why this graduate program is a great fit for this student. Each reason should be specific, actionable, and highlight unique connections.

STUDENT PROFILE:
- Major/Field: {profile.get('major', 'N/A')}
- GPA: {profile.get('gpa', 'N/A')}
- Skills: {', '.join(profile.get('skills', [])[:6])}
- Courses Taken: {', '.join(profile.get('courses', [])[:5])}
- Career Goals: {profile.get('goals', 'N/A')[:200]}
- Experience Count: {len(profile.get('experiences', []))} positions

PROGRAM INFORMATION:
- Program: {program_name}
- University: {university}
- Focus Areas: {', '.join(focus_areas[:4]) if focus_areas else 'N/A'}
- Mission: {mission[:150] if mission else 'N/A'}
- Career Outcomes: {', '.join(career_paths[:3]) if career_paths else 'N/A'}

MATCH ANALYSIS:
- Overall Match Score: {overall_score:.0%}
{chr(10).join(dim_scores_text)}

MATCHED COURSES: {courses_text}
STRENGTHS: {', '.join(strengths[:3]) if strengths else 'N/A'}
AREAS TO DEVELOP: {', '.join(gaps[:2]) if gaps else 'N/A'}

Generate exactly 4 reasons, each on a new line, starting with a relevant emoji. Each reason should be:
1. Specific to this student and program (reference actual skills, courses, or goals)
2. Encouraging and professional
3. 15-30 words long
4. Actionable where possible

Format: Just output 4 lines, each being one reason. No numbering, no extra text."""

        try:
            response = call_llm(self.llm, prompt)
            
            # Parse response into list of reasons
            reasons = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line and len(line) > 10:  # Filter out empty or too short lines
                    # Clean up any numbering
                    if line[0].isdigit() and line[1] in '.):':
                        line = line[2:].strip()
                    reasons.append(line)
            
            # Return top 4 reasons
            return reasons[:4] if reasons else self._generate_rule_based_fit_reasons(
                profile, program_data, overall_score,
                dimension_scores, matched_courses, strengths, gaps
            )
            
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}")
    
    def _generate_rule_based_fit_reasons(
        self,
        profile: Dict[str, Any],
        program_data: Any,
        overall_score: float,
        dimension_scores: Dict[str, Any],
        matched_courses: List[str],
        strengths: List[str],
        gaps: List[str]
    ) -> List[str]:
        """Generate fit reasons using enhanced rules (fallback)"""
        reasons = []
        
        if not program_data:
            return ["This program aligns with your academic background and career goals."]
        
        major = profile.get('major', 'your field')
        skills = profile.get('skills', [])
        goals = profile.get('goals', '')
        
        # Get program info safely
        bg = getattr(program_data, 'program_background', None)
        outcomes = getattr(program_data, 'training_outcomes', None)
        
        # 1. Academic fit reason
        academic_score = dimension_scores.get('academic')
        if academic_score:
            score_val = getattr(academic_score, 'score', 0) if hasattr(academic_score, 'score') else academic_score
            if score_val >= 0.7:
                reasons.append(
                    f"🎓 Your {major} background provides excellent preparation, "
                    f"with {score_val:.0%} alignment in academic fundamentals."
                )
            elif score_val >= 0.5:
                reasons.append(
                    f"📚 Your academic foundation in {major} offers a solid starting point "
                    f"for this program's curriculum."
                )
        
        # 2. Skills fit reason
        skills_score = dimension_scores.get('skills')
        if skills_score:
            matched_items = getattr(skills_score, 'matched_items', []) if hasattr(skills_score, 'matched_items') else []
            if matched_items:
                skill_names = [s.replace("Skill: ", "").replace("Required: ", "") for s in matched_items[:3]]
                reasons.append(
                    f"💡 Your expertise in {', '.join(skill_names)} directly matches "
                    f"the program's technical requirements."
                )
            elif skills:
                reasons.append(
                    f"🛠️ Your technical skills ({', '.join(skills[:2])}) provide a strong "
                    f"foundation for advanced coursework."
                )
        
        # 3. Course match reason
        if matched_courses:
            reasons.append(
                f"📖 Courses like \"{matched_courses[0]}\" align perfectly with your interests, "
                f"offering hands-on learning in relevant areas."
            )
        
        # 4. Career alignment reason
        if outcomes:
            career_paths = getattr(outcomes, 'career_paths', [])
            if career_paths and goals:
                reasons.append(
                    f"🚀 The program's career paths ({career_paths[0][:30]}...) "
                    f"align with your goal to {goals[:40]}..."
                )
            elif career_paths:
                reasons.append(
                    f"💼 Graduates pursue careers in {', '.join(career_paths[:2])}, "
                    f"opening diverse professional opportunities."
                )
        
        # 5. Mission/focus fit reason
        if bg:
            focus_areas = getattr(bg, 'focus_areas', [])
            mission = getattr(bg, 'mission', '')
            
            if focus_areas:
                reasons.append(
                    f"🎯 The program's focus on {focus_areas[0]} "
                    f"resonates with your academic and professional interests."
                )
            elif mission:
                reasons.append(
                    f"✨ The program's mission to {mission[:50]}... "
                    f"aligns with your objectives."
                )
        
        # 6. Curriculum depth reason
        curriculum_score = dimension_scores.get('curriculum')
        if curriculum_score:
            score_val = getattr(curriculum_score, 'score', 0) if hasattr(curriculum_score, 'score') else curriculum_score
            if score_val >= 0.6:
                reasons.append(
                    f"📋 The curriculum structure ({score_val:.0%} match) indicates courses "
                    f"designed for students with your background."
                )
        
        # Ensure we have at least 2 reasons
        if len(reasons) < 2:
            reasons.append(
                f"🌟 Your profile demonstrates {overall_score:.0%} overall compatibility "
                f"with the program's requirements and objectives."
            )
        
        return reasons[:4]
    
    def generate_detailed_explanation(
        self,
        profile: Dict[str, Any],
        program_data: Any,
        overall_score: float,
        dimension_scores: Dict[str, Any],
        matched_courses: List[str],
        strengths: List[str],
        gaps: List[str]
    ) -> str:
        """
        Generate a detailed paragraph explanation of program match
        
        Args:
            profile: Student profile
            program_data: ProgramDataV2 object
            overall_score: Overall match score
            dimension_scores: Scores for each dimension
            matched_courses: Matched course names
            strengths: List of student strengths
            gaps: List of areas to improve
        
        Returns:
            Detailed explanation text
        """
        
        if self.use_llm:
            try:
                return self._generate_llm_detailed_explanation(
                    profile, program_data, overall_score,
                    dimension_scores, matched_courses, strengths, gaps
                )
            except Exception as e:
                print(f"Warning: LLM detailed explanation failed: {e}")
        
        # Fallback to rule-based
        return self._generate_rule_based_detailed_explanation(
            profile, program_data, overall_score,
            dimension_scores, matched_courses, strengths, gaps
        )
    
    def _generate_llm_detailed_explanation(
        self,
        profile: Dict[str, Any],
        program_data: Any,
        overall_score: float,
        dimension_scores: Dict[str, Any],
        matched_courses: List[str],
        strengths: List[str],
        gaps: List[str]
    ) -> str:
        """Generate detailed explanation using LLM"""
        
        program_name = getattr(program_data, 'program_name', 'Graduate Program')
        university = getattr(program_data, 'university', 'University')
        
        # Format dimension scores
        dim_scores_text = []
        for dim_name, score_obj in dimension_scores.items():
            score_val = getattr(score_obj, 'score', 0) if hasattr(score_obj, 'score') else score_obj
            dim_scores_text.append(f"- {dim_name.replace('_', ' ').title()}: {score_val:.0%}")
        
        prompt = f"""Write a concise, insightful explanation (100-150 words) analyzing this student-program match.

Student: {profile.get('major', 'N/A')} major, GPA {profile.get('gpa', 'N/A')}, Skills: {', '.join(profile.get('skills', [])[:4])}
Goals: {profile.get('goals', 'N/A')[:100]}

Program: {program_name} at {university}

Match Analysis:
- Overall Score: {overall_score:.0%}
{chr(10).join(dim_scores_text)}

Strengths: {', '.join(strengths[:3]) if strengths else 'N/A'}
Areas to Develop: {', '.join(gaps[:2]) if gaps else 'N/A'}
Matched Courses: {', '.join(matched_courses[:3]) if matched_courses else 'N/A'}

Provide a personalized explanation that:
1. Explains why this is a {self._get_match_quality(overall_score)} match
2. Highlights 2 key strengths for this program
3. Notes 1 area to address
4. Offers 1 specific recommendation

Be encouraging, professional, and specific. Use a supportive tone."""

        try:
            explanation = call_llm(self.llm, prompt)
            return explanation.strip()
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}")
    
    def _generate_rule_based_detailed_explanation(
        self,
        profile: Dict[str, Any],
        program_data: Any,
        overall_score: float,
        dimension_scores: Dict[str, Any],
        matched_courses: List[str],
        strengths: List[str],
        gaps: List[str]
    ) -> str:
        """Generate detailed explanation using rules (fallback)"""
        
        program_name = getattr(program_data, 'program_name', 'this program') if program_data else 'this program'
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
        
        # Strengths
        if strengths:
            strengths_text = f" Your key strengths include: {', '.join(strengths[:3])}."
        else:
            strengths_text = ""
        
        # Best dimension
        best_dim = None
        best_score = 0
        for dim_name, score_obj in dimension_scores.items():
            score_val = getattr(score_obj, 'score', 0) if hasattr(score_obj, 'score') else score_obj
            if score_val > best_score:
                best_score = score_val
                best_dim = dim_name
        
        if best_dim:
            dimension_text = f" Your {best_dim.replace('_', ' ')} alignment ({best_score:.0%}) is particularly strong."
        else:
            dimension_text = ""
        
        # Courses mention
        if matched_courses:
            courses_text = f" Courses like '{matched_courses[0]}' directly relate to your interests."
        else:
            courses_text = ""
        
        # Gaps
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
        
        return opening + strengths_text + dimension_text + courses_text + gaps_text + recommendation
    
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
    
    def generate_batch_fit_reasons(
        self,
        profile: Dict[str, Any],
        matches: List[Any],  # List of ProgramMatchV2
        batch_size: int = 3
    ) -> Dict[str, List[str]]:
        """
        Generate fit reasons for multiple programs in a single LLM call
        
        This is more efficient than calling generate_fit_reasons for each program
        separately, reducing API calls and avoiding timeout issues.
        
        Args:
            profile: Student profile dictionary
            matches: List of ProgramMatchV2 objects
            batch_size: Number of programs to process in one LLM call (default 3)
        
        Returns:
            Dictionary mapping program_id to list of fit reasons
        """
        results = {}
        
        # Process in batches
        for i in range(0, len(matches), batch_size):
            batch = matches[i:i + batch_size]
            
            if self.use_llm:
                try:
                    batch_results = self._generate_llm_batch_fit_reasons(profile, batch)
                    results.update(batch_results)
                    continue
                except Exception as e:
                    print(f"Warning: LLM batch generation failed: {e}")
                    # Fall through to rule-based
            
            # Fallback: generate rule-based for each in batch
            for match in batch:
                program_id = getattr(match, 'program_id', str(id(match)))
                
                # Extract strengths and gaps
                strengths = []
                gaps = []
                for dim_name, score in match.dimension_scores.items():
                    if score.score >= 0.7 and score.matched_items:
                        strengths.extend(score.matched_items[:2])
                    elif score.score < 0.5:
                        gaps.append(f"Improve {dim_name.replace('_', ' ')}")
                
                results[program_id] = self._generate_rule_based_fit_reasons(
                    profile,
                    match.program_data,
                    match.overall_score,
                    match.dimension_scores,
                    match.matched_courses,
                    strengths,
                    gaps
                )
        
        return results
    
    def _generate_llm_batch_fit_reasons(
        self,
        profile: Dict[str, Any],
        matches: List[Any]
    ) -> Dict[str, List[str]]:
        """Generate fit reasons for multiple programs in one LLM call"""
        
        # Build program descriptions
        programs_text = ""
        program_ids = []
        
        for idx, match in enumerate(matches, 1):
            program_data = match.program_data
            program_id = getattr(match, 'program_id', str(id(match)))
            program_ids.append(program_id)
            
            program_name = getattr(program_data, 'program_name', 'Graduate Program') if program_data else 'Program'
            university = getattr(program_data, 'university', 'University') if program_data else 'University'
            
            # Get focus areas and career paths
            bg = getattr(program_data, 'program_background', None) if program_data else None
            focus_areas = getattr(bg, 'focus_areas', []) if bg else []
            outcomes = getattr(program_data, 'training_outcomes', None) if program_data else None
            career_paths = getattr(outcomes, 'career_paths', []) if outcomes else []
            
            # Format dimension scores
            dim_summary = []
            for dim_name, score in match.dimension_scores.items():
                dim_summary.append(f"{dim_name}: {score.score:.0%}")
            
            courses_text = ', '.join(match.matched_courses[:3]) if match.matched_courses else "N/A"
            
            programs_text += f"""
[PROGRAM {idx}] ID: {program_id}
- Name: {program_name} at {university}
- Match Score: {match.overall_score:.0%}
- Dimensions: {', '.join(dim_summary)}
- Focus: {', '.join(focus_areas[:3]) if focus_areas else 'N/A'}
- Careers: {', '.join(career_paths[:2]) if career_paths else 'N/A'}
- Matched Courses: {courses_text}
"""
        
        prompt = f"""Generate personalized "Why This Program Fits You" reasons for {len(matches)} graduate programs for one student. Be specific and reference actual student attributes.

STUDENT PROFILE:
- Major: {profile.get('major', 'N/A')}
- GPA: {profile.get('gpa', 'N/A')}
- Skills: {', '.join(profile.get('skills', [])[:6])}
- Courses: {', '.join(profile.get('courses', [])[:4])}
- Goals: {profile.get('goals', 'N/A')[:150]}
- Experience: {len(profile.get('experiences', []))} positions

PROGRAMS TO ANALYZE:
{programs_text}

For EACH program, generate exactly 3 compelling reasons. Format your response EXACTLY as:

[PROGRAM 1]
🎓 [First reason about academic/skills fit - 15-25 words]
💡 [Second reason about career/goals alignment - 15-25 words]
📖 [Third reason about courses/curriculum fit - 15-25 words]

[PROGRAM 2]
...

Important:
- Each reason MUST start with an emoji
- Be specific to THIS student's profile (mention their actual skills, major, goals)
- Keep each reason 15-25 words
- Output ONLY the formatted reasons, no other text"""

        try:
            response = call_llm(self.llm, prompt)
            return self._parse_batch_response(response, program_ids)
        except Exception as e:
            raise RuntimeError(f"LLM batch call failed: {e}")
    
    def _parse_batch_response(
        self, 
        response: str, 
        program_ids: List[str]
    ) -> Dict[str, List[str]]:
        """Parse LLM batch response into per-program reasons"""
        results = {}
        
        # Split by program markers
        parts = response.split('[PROGRAM')
        
        for i, program_id in enumerate(program_ids):
            reasons = []
            
            # Find the corresponding part (index i+1 because split creates empty first element or header)
            part_idx = i + 1
            if part_idx < len(parts):
                part = parts[part_idx]
                
                # Extract lines that look like reasons (start with emoji or have emoji)
                for line in part.split('\n'):
                    line = line.strip()
                    # Skip empty lines and program markers
                    if not line or line.startswith(']') or line[0].isdigit():
                        continue
                    # Check if line contains content (emoji or substantial text)
                    if len(line) > 10:
                        # Clean up any leading markers like "1]" or "2]"
                        if ']' in line[:5]:
                            line = line.split(']', 1)[-1].strip()
                        if line:
                            reasons.append(line)
            
            # Ensure at least some reasons
            results[program_id] = reasons[:3] if reasons else [
                "🎓 Your academic background aligns well with this program's requirements.",
                "💡 Your skills and experience position you for success in this program.",
                "📖 The curriculum offers relevant courses for your career goals."
            ]
        
        return results
