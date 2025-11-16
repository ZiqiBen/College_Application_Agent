from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime

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

class ExtractedFields(BaseModel):
    program_name: Optional[str] = None
    duration: Optional[str] = None
    courses: Optional[List[str]] = None
    tuition: Optional[str] = None
    application_requirements: Optional[str] = None
    features: Optional[str] = None
    contact_email: Optional[str] = None
    language: Optional[str] = None
    others: Optional[Dict[str, Any]] = None

class CorpusDoc(BaseModel):
    id: str
    source_url: str
    raw_snapshot_path: str
    content_type: str
    crawl_date: datetime
    checksum: str
    language: Optional[str] = None
    title: Optional[str] = None
    raw_text: Optional[str] = None
    schema_version: str = "v1.0"
    extracted_fields: Optional[ExtractedFields] = None
    snippets: Optional[Dict[str, Snippet]] = None
    chunks: Optional[List[Chunk]] = None
    llm_calls: Optional[List[LLMCall]] = None
    provenance_notes: Optional[str] = None
    notes: Optional[str] = None