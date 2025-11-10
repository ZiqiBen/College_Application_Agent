import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ContentType(Enum):
    PERSONAL_STATEMENT = "personal_statement"
    RESUME_BULLETS = "resume_bullets"
    RECOMMENDATION = "recommendation"



@dataclass
class CritiqueResult:
    score: float  # 0-1 score
    feedback: str
    suggestions: List[str]
    approved: bool

@dataclass
class WriterResult:
    content: str
    metadata: Dict
    confidence: float

class WriterAgent:
    """Agent responsible for generating content"""
    
    def __init__(self, corpus: Dict, evidence_texts: List[str]):
        """Initialize WriterAgent with corpus and evidence texts."""
        self.corpus = corpus
        self.evidence_texts = evidence_texts
    
    def write_personal_statement(self, profile: Dict, program_text: str, 
                                previous_version: Optional[str] = None,
                                critique_feedback: Optional[str] = None) -> WriterResult:
        """Generate or refine a personal statement based on profile and program info, optionally using previous version and critique feedback."""
        bits = self._extract_program_bits(self.evidence_texts)
        program_name = self._extract_program_name(program_text)
        
        # Base content generation
        if previous_version is None:
            content = self._generate_initial_ps(profile, program_name, bits)
            confidence = 0.6
        else:
            # Refine based on critique
            content = self._refine_ps(previous_version, critique_feedback, profile, program_name, bits)
            confidence = 0.8
            
        return WriterResult(
            content=content,
            metadata={"program_name": program_name, "bits": bits},
            confidence=confidence
        )
    
    def write_resume_bullets(self, profile: Dict, must_keywords: List[str],
                           previous_version: Optional[str] = None,
                           critique_feedback: Optional[str] = None) -> WriterResult:
        """Generate or refine resume bullet points based on profile and required keywords, optionally using previous version and critique feedback."""
        if previous_version is None:
            bullets = self._generate_initial_bullets(profile, must_keywords)
            confidence = 0.7
        else:
            bullets = self._refine_bullets(previous_version, critique_feedback, profile, must_keywords)
            confidence = 0.85
            
        return WriterResult(
            content=bullets,
            metadata={"keywords": must_keywords},
            confidence=confidence
        )
    
    def write_recommendation(self, profile: Dict, program_name: str, 
                           program_highlights: str, skills: List[str],
                           previous_version: Optional[str] = None,
                           critique_feedback: Optional[str] = None) -> WriterResult:
        """Generate or refine a recommendation letter based on profile, program, and skills, optionally using previous version and critique feedback."""
        if previous_version is None:
            content = self._generate_initial_recommendation(profile, program_name, program_highlights, skills)
            confidence = 0.6
        else:
            content = self._refine_recommendation(previous_version, critique_feedback, profile, program_name, program_highlights, skills)
            confidence = 0.8
            
        return WriterResult(
            content=content,
            metadata={"program_name": program_name, "highlights": program_highlights},
            confidence=confidence
        )
    
    def _extract_program_bits(self, evidence_texts: List[str]) -> Dict[str, str]:
        """Extract mission and core courses from evidence texts."""
        mission = ""
        courses = []
        for t in evidence_texts:
            tl = t.lower()
            if ("mission" in tl or "Program Mission" in t) and not mission:
                mission = t.strip().split("\n")[0]
            if "Core" in t and not courses:
                try:
                    seg = t.split("Core:")[-1].split(".")[0]
                    courses = [c.strip() for c in seg.split(",") if c.strip()]
                except Exception:
                    pass
        return {
            "mission": mission,
            "courses_str": ", ".join(courses[:3]) if courses else "Machine Learning, Statistical Inference, Data Management"
        }
    
    def _extract_program_name(self, program_text: str) -> str:
        """Extract the program name from the program text using heuristics and regex patterns."""
        # Simple extraction - could be enhanced with NLP
        lines = program_text.split('\n')[:10]
        for line in lines:
            line = line.strip()
            if line and any(word in line.lower() for word in ['program', 'degree', 'master', 'bachelor', 'ms', 'bs', 'phd']):
                if len(line) < 100:  # Reasonable title length
                    return line
        
        # Fallback: look for common program patterns
        common_patterns = [
            r'(Master.*?Data.*?Science)',
            r'(MS.*?Data.*?Science)', 
            r'(Bachelor.*?Data.*?Science)',
            r'(BS.*?Data.*?Science)',
            r'(Data.*?Science.*?Program)',
            r'(Analytics.*?Program)'
        ]
        
        for pattern in common_patterns:
            match = re.search(pattern, program_text, re.IGNORECASE)
            if match:
                return match.group(1)
                
        return "the Data Science program"
    
    def _generate_initial_ps(self, profile: Dict, program_name: str, bits: Dict) -> str:
        """Generate the initial personal statement using profile and program details."""
        skills = ", ".join((profile.get("skills") or [])[:3]) or profile.get("major","Data Science")
        
        # More sophisticated motivation based on profile
        motivation = self._generate_motivation(profile)
        background = self._generate_background(profile)
        interests = self._generate_interests(profile)
        goals = profile.get("goals", "Advance my expertise as a data scientist and contribute to impactful projects.")
        
        ps = f"""# Personal Statement

**Motivation and Purpose**
{motivation}

**Academic and Professional Background**
{background} Through my coursework and projects, I have gained substantial experience in {skills}, which has prepared me well for advanced study in data science.

**Program Alignment**
I am particularly drawn to {program_name} because of its comprehensive curriculum and mission-driven approach. The core courses including {bits["courses_str"]} align perfectly with my career interests in {interests}. {bits["mission"] if bits["mission"] else "The program's focus on practical applications and ethical considerations resonates with my values and career aspirations."}

**Future Goals and Contribution**
{goals} I am eager to contribute to the program through collaborative research projects and to leverage my background in ethical, mission-aligned data science initiatives. I believe my experience and passion will enable me to make meaningful contributions to the academic community.

**Conclusion**
With my foundation in {skills} and demonstrated passion for data science, I am confident that I will thrive in this program and contribute meaningfully to the academic community while advancing toward my career objectives in data science and analytics.
"""
        return ps
    
    def _generate_motivation(self, profile: Dict) -> str:
        """Generate a personalized motivation section for the personal statement."""
        base_motivations = [
            "I am passionate about applying modern machine learning and statistical methods to solve real-world problems through data-driven insights.",
            "My fascination with data science stems from its unique ability to transform raw information into actionable insights that can drive meaningful change.",
            "I am driven by the potential of data science to address complex challenges across industries and create positive societal impact."
        ]
        
        # Customize based on experiences
        if profile.get("experiences"):
            exp = profile["experiences"][0]
            if "healthcare" in str(exp).lower():
                return "I am passionate about leveraging data science to improve healthcare outcomes and patient experiences through advanced analytics and machine learning."
            elif "finance" in str(exp).lower():
                return "My experience in finance has shown me the transformative power of data science in risk management and decision-making, motivating me to deepen my expertise."
            elif "education" in str(exp).lower():
                return "Working in education has highlighted the potential of data science to personalize learning and improve educational outcomes, driving my passion for advanced study."
        
        return base_motivations[0]
    
    def _generate_background(self, profile: Dict) -> str:
        """Generate the background section for the personal statement based on profile."""
        major = profile.get("major", "Data Science")
        base_background = f"With a background in {major}, I have developed strong analytical skills and hands-on experience in data analysis and machine learning."
        
        if profile.get("experiences"):
            exp_count = len(profile["experiences"])
            if exp_count > 1:
                base_background += f" Through {exp_count} professional experiences, I have applied these skills across different domains and challenges."
            else:
                exp = profile["experiences"][0]
                org = exp.get("org", "my previous organization")
                base_background += f" In my role at {org}, I have applied these skills to real-world challenges and delivered measurable results."
        
        return base_background
    
    def _generate_interests(self, profile: Dict) -> str:
        """Generate interests section for the personal statement based on skills and experiences."""
        skills = profile.get("skills", [])
        interests = []
        
        if any("machine learning" in skill.lower() for skill in skills):
            interests.append("applied machine learning")
        if any("statistics" in skill.lower() or "statistical" in skill.lower() for skill in skills):
            interests.append("statistical modeling")
        if any("visualization" in skill.lower() for skill in skills):
            interests.append("data visualization")
        if any("nlp" in skill.lower() or "natural language" in skill.lower() for skill in skills):
            interests.append("natural language processing")
        
        if not interests:
            interests = ["applied machine learning", "statistical analysis", "data-driven decision making"]
        
        return ", ".join(interests[:3])
    
    def _refine_ps(self, previous: str, feedback: str, profile: Dict, program_name: str, bits: Dict) -> str:
        """Refine the personal statement using critic feedback and profile details."""
        refined = previous
        
        if feedback and "more specific" in feedback.lower():
            # Add more specific details
            if profile.get("experiences"):
                exp = profile["experiences"][0]
                impact = exp.get("impact", "delivered measurable results")
                title = exp.get("title", "a data analyst")
                specific_detail = f"For example, in my role as {title}, I {impact}."
                
                # Insert specific detail into background section
                background_pattern = r"(Through my coursework and projects,)"
                replacement = f"\\1 {specific_detail} Additionally,"
                refined = re.sub(background_pattern, replacement, refined)
        
        if feedback and "stronger connection" in feedback.lower():
            # Enhance program connection
            enhanced_connection = f" This curriculum directly supports my goal of developing expertise in advanced analytics and machine learning applications."
            refined = refined.replace(bits["courses_str"], bits["courses_str"] + enhanced_connection)
        
        if feedback and "quantify" in feedback.lower():
            # Add quantification where possible
            if profile.get("gpa"):
                gpa_mention = f" I maintained a {profile['gpa']:.2f} GPA throughout my studies,"
                refined = refined.replace("Through my coursework", f"Through my coursework,{gpa_mention} and")
        
        return refined
    
    def _generate_initial_bullets(self, profile: Dict, must_keywords: List[str]) -> str:
        """Generate initial resume bullet points using profile and required keywords."""
        kws = ", ".join(must_keywords[:6])
        lines = self._bullets_from_profile(profile)
        if not lines:
            # Generate enhanced default bullets with better action words
            lines = [
                "- Led comprehensive end-to-end machine learning project resulting in 15% improvement in predictive accuracy",
                "- Developed and implemented automated data pipeline, reducing manual analysis time by 40% and improving efficiency",
                "- Designed statistical models for cohort analysis and customer segmentation, contributing to 7% reduction in customer churn",
                "- Created interactive data visualizations and dashboards for stakeholder decision-making across multiple departments"
            ]
        guidance = f"# Resume Bullet Points\n**Target Keywords:** {kws}\n**Guidelines:** Use strong action verbs, quantify achievements, align with program focus\n\n"
        return guidance + "\n".join(lines)
    
    def _refine_bullets(self, previous: str, feedback: str, profile: Dict, must_keywords: List[str]) -> str:
        """Refine resume bullet points using critic feedback and profile details."""
        # Extract existing bullets
        lines = previous.split('\n')
        bullets = [line for line in lines if line.strip().startswith('-')]
        non_bullets = [line for line in lines if not line.strip().startswith('-')]
        
        if feedback and "quantify" in feedback.lower():
            # Add more quantification
            refined_bullets = []
            for bullet in bullets:
                if not any(char.isdigit() for char in bullet):
                    # Add realistic quantification
                    bullet = bullet.replace("improved", "improved by 20%")
                    bullet = bullet.replace("reduced", "reduced by 15%")
                    bullet = bullet.replace("increased", "increased by 25%")
                    bullet = bullet.replace("delivered", "delivered 30% faster")
                    bullet = bullet.replace("created", "created 5+ deliverables")
                refined_bullets.append(bullet)
            bullets = refined_bullets
        
        if feedback and "action words" in feedback.lower():
            # Enhance action words
            action_replacements = {
                "worked on": "Led",
                "helped with": "Collaborated on",
                "did": "Executed",
                "made": "Developed",
                "used": "Leveraged"
            }
            for i, bullet in enumerate(bullets):
                for weak, strong in action_replacements.items():
                    if weak in bullet.lower():
                        bullets[i] = bullet.replace(weak, strong)
                        break
        
        if feedback and "keywords" in feedback.lower():
            # Better integrate keywords
            for i, bullet in enumerate(bullets):
                bullet_lower = bullet.lower()
                for kw in must_keywords:
                    if kw.lower() not in bullet_lower and "project" in bullet_lower:
                        bullets[i] = bullet.replace("project", f"{kw} project", 1)
                        break
        
        return '\n'.join(non_bullets + bullets)
    
    def _generate_initial_recommendation(self, profile: Dict, program_name: str, 
                                       program_highlights: str, skills: List[str]) -> str:
        """Generate an initial recommendation letter template using profile and program details."""
        student = profile.get("name", "The student")
        skills_str = ", ".join(skills[:4]) if skills else "data science and analytics"
        
        # Generate more specific examples based on profile
        example1, example2 = self._generate_recommendation_examples(profile)
        
        return f"""# Recommendation Letter Template

To Whom It May Concern,

I am writing to provide my strongest recommendation for {student}'s admission to {program_name}. I have had the privilege of supervising {student} for [TIME PERIOD], during which they have consistently demonstrated exceptional analytical capabilities, technical proficiency, and professional maturity that distinguish them among their peers.

**Technical Excellence and Project Leadership**
{student} has shown remarkable competence in {skills_str}. {example1} Their approach was methodical, their analysis was thorough, and their conclusions were actionable and well-communicated to stakeholders.

**Communication and Collaboration Skills**
Beyond technical excellence, {student} excels at translating complex analytical findings into clear, actionable insights for diverse audiences. {example2} This ability to bridge the technical-business divide is particularly valuable and demonstrates their potential for leadership roles.

**Academic Potential and Program Fit**
Given {student}'s demonstrated strengths in {skills_str} and their passion for data-driven problem solving, I am confident they will excel in your program's focus on {program_highlights}. Their combination of technical skill, analytical thinking, clear communication, and collaborative approach makes them an ideal candidate for advanced study in data science.

I recommend {student} without any reservations and believe they will be a valuable addition to your program and academic community. Please feel free to contact me if you require any additional information.

Sincerely,
[Recommender Name]
[Title and Organization]
[Email Address]
[Phone Number]
"""
    
    def _generate_recommendation_examples(self, profile: Dict) -> Tuple[str, str]:
        """Generate specific example achievements for the recommendation letter based on profile experiences."""
        if profile.get("experiences") and len(profile["experiences"]) >= 1:
            exp = profile["experiences"][0]
            title = exp.get("title", "data analyst")
            impact = exp.get("impact", "delivered significant improvements to our analytical capabilities")
            
            example1 = f"In one notable instance, while working as {title}, {profile.get('name', 'they')} independently executed a comprehensive machine learning project that {impact}."
            
            if len(profile["experiences"]) >= 2:
                exp2 = profile["experiences"][1] 
                example2 = f"In another project, they {exp2.get('impact', 'created valuable deliverables')} which significantly improved our team's decision-making processes."
            else:
                example2 = f"They also produced high-quality data visualizations and reports that significantly improved stakeholder understanding and decision-making processes."
        else:
            example1 = f"In one notable instance, {profile.get('name', 'they')} independently executed a comprehensive machine learning project that resulted in measurable improvements to our analytical capabilities."
            example2 = f"They also produced exceptional data visualizations and analytical reports that significantly enhanced stakeholder understanding and strategic decision-making."
            
        return example1, example2
    
    def _refine_recommendation(self, previous: str, feedback: str, profile: Dict, 
                             program_name: str, program_highlights: str, skills: List[str]) -> str:
        """Refine the recommendation letter using critic feedback and profile details."""
        refined = previous
        
        if feedback and "more specific" in feedback.lower() and profile.get("experiences"):
            # Add more specific details
            exp = profile["experiences"][0]
            org = exp.get("org", "the organization")
            specific_context = f"During their tenure at {org},"
            
            # Insert more context
            refined = refined.replace(
                "I have had the privilege of supervising",
                f"I have had the privilege of supervising {specific_context} where I directly observed their exceptional performance. Over this period,"
            )
        
        if feedback and "stronger language" in feedback.lower():
            # Enhance recommendation strength
            replacements = {
                "recommend": "strongly recommend",
                "good": "exceptional", 
                "well": "excellently",
                "capable": "highly capable",
                "skilled": "exceptionally skilled"
            }
            for weak, strong in replacements.items():
                refined = refined.replace(weak, strong)
        
        return refined
    
    def _bullets_from_profile(self, profile: Dict) -> List[str]:
        """Generate resume bullet points from profile experiences with strong action verbs and quantification"""
        bullets = []
        strong_verbs = ["Led", "Developed", "Implemented", "Designed", "Created", "Optimized", "Analyzed", "Managed"]
        
        for i, exp in enumerate(profile.get("experiences", [])):
            impact = exp.get("impact", "delivered measurable impact")
            skills = exp.get("skills", [])
            skills_str = ", ".join(skills[:3]) if skills else "various technical skills"
            org = exp.get("org", "Organization")
            title = exp.get("title", "Role")
            
            # Use strong action verb
            verb = strong_verbs[i % len(strong_verbs)]
            
            # Ensure impact is quantified if not already
            if not any(char.isdigit() for char in impact):
                impact = f"{impact}, achieving 20% improvement in efficiency"
            
            bullet = f"- {verb} {title.lower()} initiatives at {org}: {impact} (Skills: {skills_str})"
            bullets.append(bullet)
            
        return bullets[:8]

class CriticAgent:
    """Agent responsible for critiquing and providing feedback on content"""
    
    def __init__(self, must_keywords: List[str]):
        """Initialize CriticAgent with required keywords."""
        self.must_keywords = must_keywords
    
    def critique_personal_statement(self, content: str, profile: Dict, program_text: str) -> CritiqueResult:
        """Critique a personal statement for keyword coverage, structure, specificity, and program connection."""
        score = 0.0
        feedback_items = []
        suggestions = []
        
        # Check keyword coverage (30% weight)
        keyword_score = self._check_keyword_coverage(content)
        score += keyword_score * 0.3
        if keyword_score < 0.7:
            missing_keywords = [kw for kw in self.must_keywords if kw.lower() not in content.lower()]
            feedback_items.append(f"Keyword coverage is low ({keyword_score:.2f}). Missing: {', '.join(missing_keywords[:3])}")
            suggestions.append("Naturally incorporate more program-relevant keywords throughout your statement")
        
        # Check structure and flow (25% weight)
        structure_score = self._check_ps_structure(content)
        score += structure_score * 0.25
        if structure_score < 0.8:
            feedback_items.append("Personal statement structure needs improvement")
            suggestions.append("Ensure clear sections: motivation, background, program fit, goals, and conclusion")
        
        # Check specificity (25% weight)
        specificity_score = self._check_specificity(content, profile)
        score += specificity_score * 0.25
        if specificity_score < 0.7:
            feedback_items.append("Statement lacks specific examples and quantifiable details")
            suggestions.append("Add specific examples from your experience with measurable outcomes")
        
        # Check program connection (20% weight)
        connection_score = self._check_program_connection(content, program_text)
        score += connection_score * 0.2
        if connection_score < 0.7:
            feedback_items.append("Connection to specific program features could be stronger")
            suggestions.append("Reference specific program courses, faculty, or unique features that align with your goals")
        
        feedback = "; ".join(feedback_items) if feedback_items else "Personal statement demonstrates strong quality overall"
        approved = score >= 0.75
        
        return CritiqueResult(
            score=score,
            feedback=feedback,
            suggestions=suggestions,
            approved=approved
        )
    
    def critique_resume_bullets(self, content: str, profile: Dict) -> CritiqueResult:
        """Critique resume bullet points for quantification, action words, and keyword integration."""
        bullets = [line.strip() for line in content.split('\n') if line.strip().startswith('-')]
        score = 0.0
        feedback_items = []
        suggestions = []
        
        if not bullets:
            return CritiqueResult(
                score=0.0,
                feedback="No bullet points found",
                suggestions=["Add bullet points describing your achievements"],
                approved=False
            )
        
        # Check quantification (40% weight)
        quantified_count = sum(1 for bullet in bullets if any(char.isdigit() for char in bullet))
        quantification_score = quantified_count / len(bullets)
        score += quantification_score * 0.4
        if quantification_score < 0.6:
            unquantified = len(bullets) - quantified_count
            feedback_items.append(f"{unquantified} bullets lack quantified achievements")
            suggestions.append("Add specific metrics, percentages, or numbers to demonstrate measurable impact")
        
        # Check action words (30% weight)
        action_score = self._check_action_words(bullets)
        score += action_score * 0.3
        if action_score < 0.7:
            feedback_items.append("Use stronger, more impactful action verbs")
            suggestions.append("Start bullets with powerful action words like 'Led', 'Developed', 'Implemented', 'Optimized'")
        
        # Check keyword integration (30% weight)
        keyword_score = self._check_keyword_coverage(content)
        score += keyword_score * 0.3
        if keyword_score < 0.8:
            feedback_items.append("Better integration of program-relevant keywords needed")
            suggestions.append("Naturally incorporate technical keywords that align with the target program")
        
        feedback = "; ".join(feedback_items) if feedback_items else "Resume bullets demonstrate strong professional achievements"
        approved = score >= 0.75
        
        return CritiqueResult(
            score=score,
            feedback=feedback,
            suggestions=suggestions,
            approved=approved
        )
    
    def critique_recommendation(self, content: str, profile: Dict) -> CritiqueResult:
        """Critique a recommendation letter for specific examples, recommendation strength, and comprehensiveness."""
        score = 0.0
        feedback_items = []
        suggestions = []
        
        # Check for specific examples (40% weight)
        example_keywords = ["instance", "example", "project", "demonstrated", "executed", "delivered"]
        has_examples = sum(1 for keyword in example_keywords if keyword in content.lower())
        example_score = min(has_examples / 3, 1.0)  # Normalize to 0-1
        score += example_score * 0.4
        if example_score < 0.7:
            feedback_items.append("Letter lacks sufficient specific examples of student's work")
            suggestions.append("Include 2-3 concrete examples of the student's projects and achievements")
        
        # Check recommendation strength (30% weight)
        strength_words = ["strongly recommend", "highest recommendation", "exceptional", "outstanding", "excellent", "superior"]
        strength_score = 1.0 if any(word in content.lower() for word in strength_words) else 0.5
        score += strength_score * 0.3
        if strength_score < 0.8:
            feedback_items.append("Recommendation could express stronger enthusiasm")
            suggestions.append("Use more emphatic language to convey strong support for the candidate")
        
        # Check comprehensiveness (30% weight)
        required_elements = ["technical", "communication", "recommend", "contact"]
        element_score = sum(1 for element in required_elements if element in content.lower()) / len(required_elements)
        score += element_score * 0.3
        if element_score < 0.8:
            feedback_items.append("Letter structure could be more comprehensive")
            suggestions.append("Include sections on technical skills, soft skills, explicit recommendation, and contact information")
        
        feedback = "; ".join(feedback_items) if feedback_items else "Recommendation letter is comprehensive and compelling"
        approved = score >= 0.75
        
        return CritiqueResult(
            score=score,
            feedback=feedback,
            suggestions=suggestions,
            approved=approved
        )
    
    def _check_keyword_coverage(self, content: str) -> float:
        """Check the proportion of required keywords present in the content."""
        if not self.must_keywords:
            return 1.0
            
        content_lower = content.lower()
        hits = sum(1 for kw in self.must_keywords if kw.lower() in content_lower)
        return hits / len(self.must_keywords)
    
    def _check_ps_structure(self, content: str) -> float:
        """Evaluate the structure and organization of a personal statement."""
        required_sections = ["motivation", "background", "program", "goals", "conclusion"]
        content_lower = content.lower()
        
        # Check for section indicators
        section_indicators = {
            "motivation": ["motivation", "passionate", "driven", "inspired"],
            "background": ["background", "experience", "studied", "education"],
            "program": ["program", "curriculum", "courses", "align"],
            "goals": ["goals", "future", "career", "aspire"],
            "conclusion": ["conclusion", "confident", "contribute", "summary"]
        }
        
        section_scores = []
        for section, indicators in section_indicators.items():
            section_score = 1.0 if any(indicator in content_lower for indicator in indicators) else 0.0
            section_scores.append(section_score)
        
        return sum(section_scores) / len(section_scores)
    
    def _check_specificity(self, content: str, profile: Dict) -> float:
        """Evaluate the specificity and detail level of the content."""
        specificity_indicators = []
        
        # Has numbers/quantification
        specificity_indicators.append(any(char.isdigit() for char in content))
        
        # Sufficient length (indicates detail)
        specificity_indicators.append(len(content.split()) > 250)
        
        # Personalized (uses profile information)
        if profile.get("name"):
            specificity_indicators.append(profile["name"].lower() in content.lower())
        
        # References specific experiences
        if profile.get("experiences"):
            exp_terms = []
            for exp in profile["experiences"]:
                exp_terms.extend([exp.get("title", ""), exp.get("org", ""), exp.get("impact", "")])
            specificity_indicators.append(any(term.lower() in content.lower() for term in exp_terms if term))
        
        # References specific skills
        if profile.get("skills"):
            specificity_indicators.append(any(skill.lower() in content.lower() for skill in profile["skills"]))
        
        return sum(specificity_indicators) / len(specificity_indicators)
    
    def _check_program_connection(self, content: str, program_text: str) -> float:
        """Evaluate how well the content connects to specific program features."""
        if not program_text:
            return 0.5
        
        # Extract key terms from program text (excluding common words)
        common_words = {"the", "and", "or", "in", "to", "of", "a", "an", "for", "with", "on", "at", "by", "this", "that"}
        program_words = [word.lower() for word in re.findall(r'\b[A-Za-z]{4,}\b', program_text)]
        program_terms = [word for word in set(program_words) if word not in common_words][:15]
        
        content_lower = content.lower()
        hits = sum(1 for term in program_terms if term in content_lower)
        
        return min(hits / max(len(program_terms), 1), 1.0)
    
    def _check_action_words(self, bullets: List[str]) -> float:
        """Evaluate the strength of action words used in resume bullet points."""
        if not bullets:
            return 0.0
            
        strong_actions = [
            "led", "developed", "implemented", "created", "managed", "designed", 
            "optimized", "analyzed", "built", "delivered", "achieved", "improved",
            "established", "coordinated", "executed", "spearheaded", "pioneered"
        ]
        
        weak_actions = ["helped", "worked", "did", "used", "made", "assisted"]
        
        strong_count = 0
        weak_count = 0
        
        for bullet in bullets:
            bullet_lower = bullet.lower()
            if any(action in bullet_lower for action in strong_actions):
                strong_count += 1
            elif any(action in bullet_lower for action in weak_actions):
                weak_count += 1
        
        # Penalize for weak actions
        score = strong_count / len(bullets)
        penalty = (weak_count / len(bullets)) * 0.5
        
        return max(0, score - penalty)

class MultiAgentGenerator:
    """Orchestrates the writer and critic agents for iterative content improvement"""
    
    def __init__(self, corpus: Dict, evidence_texts: List[str], must_keywords: List[str]):
        """Initialize MultiAgentGenerator with corpus, evidence texts, and required keywords."""
        self.writer = WriterAgent(corpus, evidence_texts)
        self.critic = CriticAgent(must_keywords)
        self.must_keywords = must_keywords
    
    def generate_with_feedback(self, content_type: ContentType, profile: Dict, 
                             program_text: str, max_iterations: int = 3,
                             threshold: float = 0.8) -> Tuple[str, Dict]:
        """Generate content with iterative writer-critic feedback loop until quality threshold or approval is met."""
        current_content = None
        iteration_log = []
        
        for iteration in range(max_iterations):
            # Writer phase - generate or refine content
            if content_type == ContentType.PERSONAL_STATEMENT:
                writer_result = self.writer.write_personal_statement(
                    profile, program_text, current_content, 
                    iteration_log[-1]["critique"]["feedback"] if iteration_log else None
                )
            elif content_type == ContentType.RESUME_BULLETS:
                writer_result = self.writer.write_resume_bullets(
                    profile, self.must_keywords, current_content,
                    iteration_log[-1]["critique"]["feedback"] if iteration_log else None
                )
            elif content_type == ContentType.RECOMMENDATION:
                skills = profile.get("skills", []) or [profile.get("major", "Data Science")]
                writer_result = self.writer.write_recommendation(
                    profile, "the Data Science program", "mission, ML and analytics", 
                    skills, current_content,
                    iteration_log[-1]["critique"]["feedback"] if iteration_log else None
                )
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            current_content = writer_result.content
            
            # Critic phase - evaluate and provide feedback
            if content_type == ContentType.PERSONAL_STATEMENT:
                critique = self.critic.critique_personal_statement(current_content, profile, program_text)
            elif content_type == ContentType.RESUME_BULLETS:
                critique = self.critic.critique_resume_bullets(current_content, profile)
            elif content_type == ContentType.RECOMMENDATION:
                critique = self.critic.critique_recommendation(current_content, profile)
            
            # Log iteration results
            iteration_log.append({
                "iteration": iteration + 1,
                "writer_confidence": writer_result.confidence,
                "writer_metadata": writer_result.metadata,
                "critique": {
                    "score": critique.score,
                    "feedback": critique.feedback,
                    "suggestions": critique.suggestions,
                    "approved": critique.approved
                }
            })
            
            # Check if content meets quality threshold or is approved
            if critique.score >= threshold or critique.approved:
                break
        
        # Compile final report
        final_report = {
            "content_type": content_type.value,
            "iterations_completed": len(iteration_log),
            "final_score": iteration_log[-1]["critique"]["score"],
            "final_approved": iteration_log[-1]["critique"]["approved"],
            "quality_threshold": threshold,
            "max_iterations": max_iterations,
            "improvement_trajectory": [log["critique"]["score"] for log in iteration_log],
            "final_suggestions": iteration_log[-1]["critique"]["suggestions"],
            "iteration_log": iteration_log
        }
        
        return current_content, final_report

# Utility functions for backward compatibility and fallback
def generate_all_multi_agent(evidence: Dict[str, str], evidence_ids: List[str], 
                           profile: Dict, resume_text: str, program_text: str, 
                           must_keywords: List[str], max_iterations: int = 3,
                           threshold: float = 0.8) -> Tuple[Dict[str, str], Dict]:
    """
    Multi-agent version of generate_all function for backward compatibility.
    Generates personal statement, resume bullets, and recommendation letter with iterative feedback.
    Returns generated texts and an overall report.
    """
    ev_texts = [evidence[i] for i in evidence_ids if i in evidence]
    if not ev_texts and program_text:
        ev_texts = [program_text]  # Use program text as evidence if no other evidence
    
    # Initialize multi-agent generator
    generator = MultiAgentGenerator(evidence, ev_texts, must_keywords)
    
    # Generate each type of content
    ps_content, ps_report = generator.generate_with_feedback(
        ContentType.PERSONAL_STATEMENT, profile, program_text,
        max_iterations, threshold
    )
    
    bullets_content, bullets_report = generator.generate_with_feedback(
        ContentType.RESUME_BULLETS, profile, program_text,
        max_iterations, threshold
    )
    
    skills = profile.get("skills", []) or [profile.get("major", "Data Science")]
    reco_content, reco_report = generator.generate_with_feedback(
        ContentType.RECOMMENDATION, profile, program_text,
        max_iterations, threshold
    )
    
    # Compile results
    texts = {
        "personal_statement": ps_content,
        "resume_bullets": bullets_content,
        "reco_template": reco_content
    }
    
    # Aggregate reports
    overall_report = {
        "system_type": "multi_agent",
        "evidence_ids": evidence_ids,
        "must_keywords": must_keywords,
        "generation_parameters": {
            "max_iterations": max_iterations,
            "quality_threshold": threshold
        },
        "content_reports": {
            "personal_statement": ps_report,
            "resume_bullets": bullets_report,
            "recommendation": reco_report
        },
        "overall_quality": {
            "average_score": (ps_report["final_score"] + bullets_report["final_score"] + reco_report["final_score"]) / 3,
            "total_iterations": ps_report["iterations_completed"] + bullets_report["iterations_completed"] + reco_report["iterations_completed"],
            "all_approved": ps_report["final_approved"] and bullets_report["final_approved"] and reco_report["final_approved"]
        }
    }
    
    return texts, overall_report