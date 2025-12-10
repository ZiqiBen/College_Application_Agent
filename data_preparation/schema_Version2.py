from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime


# -------------------------
# Snippet & Chunk
# -------------------------

class Snippet(BaseModel):
    text: str
    char_start: int
    char_end: int


class Chunk(BaseModel):
    chunk_id: str
    text: str
    char_start: int
    char_end: int
    token_count: int
    vector_id: Optional[str] = None


class LLMCall(BaseModel):
    prompt_hash: str
    model: str
    date: datetime
    response_raw_path: Optional[str] = None
    cached: bool = False


# -------------------------
# Course Object
# -------------------------

class Course(BaseModel):
    name: str
    description: Optional[str] = None


# -------------------------
# NEW Structured Fields for Program
# -------------------------

class ProgramBackground(BaseModel):
    """项目背景：教育理念 / 教学资源 / 设施 / 师资等"""
    mission: Optional[str] = None
    environment: Optional[str] = None
    faculty: Optional[str] = None
    resources: Optional[str] = None
    summary: Optional[str] = None   # 页面里若是一个长段落可以直接塞进来


class TrainingOutcomes(BaseModel):
    """培养目标：希望学生成为怎样的人才、能力、未来路径"""
    goals: Optional[str] = None
    career_paths: Optional[str] = None
    research_orientation: Optional[str] = None  # 是否偏 PhD / research
    professional_orientation: Optional[str] = None  # 是否偏 industry
    summary: Optional[str] = None


class ApplicationRequirements(BaseModel):
    """更结构化的申请要求"""
    academic_background: Optional[str] = None
    prerequisites: Optional[str] = None
    gre: Optional[str] = None
    english_tests: Optional[str] = None
    research_experience: Optional[str] = None
    work_experience: Optional[str] = None
    documents: Optional[str] = None
    summary: Optional[str] = None


# -------------------------
# Extracted Fields (Main)
# -------------------------

class ExtractedFields(BaseModel):
    program_name: Optional[str] = None
    school: Optional[str] = None
    department: Optional[str] = None   # ← 新增 department

    duration: Optional[str] = None

    # 课程
    courses: Optional[List[Course]] = None

    # 申请要求（结构化）
    application_requirements: Optional[ApplicationRequirements] = None

    # 项目背景
    program_background: Optional[ProgramBackground] = None

    # 培养目标 / outcomes
    training_outcomes: Optional[TrainingOutcomes] = None

    tuition: Optional[str] = None
    contact_email: Optional[str] = None
    language: Optional[str] = None

    # 其他补充内容（可选）
    others: Optional[Dict[str, Any]] = None


# -------------------------
# Corpus Document (Simplified)
# -------------------------

class CorpusDoc(BaseModel):
    id: str
    source_url: str
    raw_snapshot_path: str
    content_type: str
    crawl_date: datetime
    checksum: str

    title: Optional[str] = None
    raw_text: Optional[str] = None
    language: Optional[str] = None

    # NEW schema version
    schema_version: str = "v2.0"

    extracted_fields: Optional[ExtractedFields] = None

    # 支持 snippet / chunk
    snippets: Optional[Dict[str, Snippet]] = None
    chunks: Optional[List[Chunk]] = None
    llm_calls: Optional[List[LLMCall]] = None

    provenance_notes: Optional[str] = None
    notes: Optional[str] = None
