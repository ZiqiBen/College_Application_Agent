import json
from typing import Dict, List


def _bullets_from_profile(profile: Dict) -> List[str]:
    """
    Generate resume-style bullet points from a profile dictionary.

    Args:
        profile (Dict): Applicant profile containing "experiences" with fields:
                        - "impact" (str): Description of contribution/impact.
                        - "skills" (List[str]): Related skills.
                        - "org" (str): Organization name.
                        - "title" (str): Role or position title.

    Returns:
        List[str]: Up to 8 formatted resume bullet strings.
    """
    b = []
    for exp in profile.get("experiences", []):
        impact = exp.get("impact", "impact")
        skills = ", ".join(exp.get("skills", [])[:3])
        org = exp.get("org", "Org")
        title = exp.get("title", "Experience")
        b.append(f"- {title} at {org}: {impact} ({skills})")
    return b[:8]


def _extract_program_bits(evidence_texts: List[str]) -> Dict[str, str]:
    """
    Extract key program details (mission and core courses) from retrieved evidence text.

    Args:
        evidence_texts (List[str]): List of evidence passages (retrieved from corpus).

    Returns:
        Dict[str, str]: A dictionary containing:
                        - "mission" (str): Extracted program mission (or empty if not found).
                        - "courses_str" (str): A comma-separated list of up to 3 core courses,
                                               or a fallback default string.
    """
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


def assemble_ps(profile: Dict, program_text: str, evidence_texts: List[str]) -> str:
    """
    Assemble a draft Personal Statement (PS) using profile and program evidence.

    Args:
        profile (Dict): Applicant profile with fields such as "skills", "major", "goals".
        program_text (str): Raw program description text.
        evidence_texts (List[str]): Retrieved evidence passages from program corpus.

    Returns:
        str: A structured draft of a Personal Statement (Markdown format).
    """
    bits = _extract_program_bits(evidence_texts)
    program_name = "the Data Science program"
    skills = ", ".join((profile.get("skills") or [])[:3]) or profile.get("major", "Data Science")
    motivation = "I hope to apply modern ML and statistical methods to real-world analytics."
    background = "I studied data science and built projects in NLP and analytics."
    interests = "applied ML and analytics"
    goals = profile.get("goals", "Grow as a data scientist.")
    contribution = "ethical, mission-aligned data projects"

    ps = f"""# Personal Statement (Draft)

**Motivation.** {motivation}

**Background.** {background}

**Program Fit.** I am particularly drawn to {program_name} for its mission-driven training and core courses.
The curriculum such as {bits["courses_str"]} and the mission "{bits["mission"]}" align with my interests in {interests}.

**Goals.** {goals}

**Conclusion.** With my training in {skills}, I aim to contribute to {contribution} and grow into a product/data science role.
"""
    return ps


def assemble_resume_bullets(profile: Dict, must_keywords: List[str]) -> str:
    """
    Assemble a set of resume bullets aligned with program keywords.

    Args:
        profile (Dict): Applicant profile containing "experiences" (optional).
        must_keywords (List[str]): List of must-have program keywords.

    Returns:
        str: Markdown-formatted resume section with guidance and bullet points.
    """
    kws = ", ".join(must_keywords[:6])
    lines = _bullets_from_profile(profile)
    if not lines:
        lines = ["- Led an end-to-end ML project delivering measurable impact.",
                 "- Automated cohort analysis to reduce churn by 7%."]
    guidance = f"# Resume Bullet Guidelines\n- Align with program keywords: {kws}\n"
    return guidance + "\n".join(lines)


def assemble_reco_template(profile: Dict, program_name: str, program_highlights: str, skills: List[str]) -> str:
    """
    Assemble a recommendation letter template grounded in applicant profile.

    Args:
        profile (Dict): Applicant profile containing at least "name".
        program_name (str): Target program name (e.g., "the MSDS program").
        program_highlights (str): Key features of the program to emphasize.
        skills (List[str]): List of applicant's notable skills.

    Returns:
        str: A draft recommendation letter template (Markdown format).
    """
    student = profile.get("name", "The student")
    skills_str = ", ".join(skills[:4]) if skills else "data science"
    tmpl = f"""# Recommendation Letter (Template)

To whom it may concern,

I am pleased to recommend {student} for admission to {program_name}. I have supervised {student}, during which they demonstrated initiative, analytical thinking, and clear communication.

In one instance, {student} executed a machine learning project end-to-end, resulting in measurable improvements. Another example is a data visualization deliverable that clarified key stakeholder decisions.

Given {student}'s strengths in {skills_str}, I believe they will thrive in your program's focus on {program_highlights}. I strongly recommend them without reservation.

Sincerely,
[Your Name], [Title]
"""
    return tmpl


def build_report(texts: Dict[str, str], evidence_ids: List[str], must_keywords: List[str], allowed_terms: List[str]) -> Dict[str, float]:
    """
    Build a simple quality report for generated outputs.

    Args:
        texts (Dict[str, str]): Generated texts containing "personal_statement", "resume_bullets", "reco_template".
        evidence_ids (List[str]): List of evidence chunk IDs used in generation.
        must_keywords (List[str]): List of required program keywords.
        allowed_terms (List[str]): Whitelisted terms that are permitted in outputs.

    Returns:
        Dict[str, float]: Report with fields:
                          - "evidence_ids" (List[str])
                          - "keyword_coverage" (float, ratio of must_keywords present in PS)
                          - "conflicts" (List[str], placeholder for disallowed terms)
    """
    ps = texts.get("personal_statement", "").lower()
    hits = sum(1 for kw in must_keywords if kw.lower() in ps)
    coverage = (hits / len(must_keywords)) if must_keywords else 1.0
    conflicts = []  # simple placeholder, future: detect disallowed terms
    return {"evidence_ids": evidence_ids, "keyword_coverage": round(coverage, 3), "conflicts": conflicts}


def generate_all(evidence: Dict[str, str], evidence_ids: List[str], profile: Dict,
                 resume_text: str, program_text: str, must_keywords: List[str]) -> Dict[str, str]:
    """
    Generate all application artifacts: Personal Statement, Resume Bullets, Recommendation Letter,
    and a keyword coverage report.

    Args:
        evidence (Dict[str, str]): Mapping of chunk_id -> text (program corpus).
        evidence_ids (List[str]): IDs of retrieved evidence chunks.
        profile (Dict): Applicant profile including name, skills, experiences, and goals.
        resume_text (str): Optional raw resume text (unused in current version).
        program_text (str): Raw program description text.
        must_keywords (List[str]): List of must-have keywords from program corpus.

    Returns:
        Tuple[Dict[str, str], Dict[str, float]]:
            - texts: A dictionary with generated "personal_statement", "resume_bullets", "reco_template".
            - report: A quality report with evidence usage and keyword coverage.
    """
    ev_texts = [evidence[i] for i in evidence_ids if i in evidence]
    ps = assemble_ps(profile, program_text, ev_texts)
    bullets = assemble_resume_bullets(profile, must_keywords)
    skills = profile.get("skills", []) or [profile.get("major", "Data Science")]
    reco = assemble_reco_template(profile, "the Data Science program", "mission, ML and analytics", skills)

    texts = {"personal_statement": ps, "resume_bullets": bullets, "reco_template": reco}
    report = build_report(
        texts,
        evidence_ids,
        must_keywords,
        allowed_terms=["Machine Learning", "Data Visualization", "Data Management", "Statistical Inference"]
    )
    return texts, report
