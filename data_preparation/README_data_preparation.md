# 🔧 Data Preparation Module — Graduate Program Corpus Builder

## Overview

The **Data Preparation** module is a comprehensive pipeline for building structured program datasets from university websites. It crawls graduate program pages, extracts structured information using GPT-4, and produces high-quality JSON corpus files used by the matching and writing services.

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Page Crawling** | Main page + curriculum subpages + course detail pages |
| **Intelligent Discovery** | Auto-discovers linked curriculum/course pages |
| **GPT-4 Extraction** | Structured information extraction with strict JSON schema |
| **Rich Schema** | 6 structured fields: courses, requirements, background, outcomes, etc. |
| **Caching System** | LLM response caching to avoid redundant API calls |
| **Dual Format Support** | HTML and PDF document extraction |
| **Bounded Crawling** | Rate limiting and max-link controls to avoid overload |

---

## 📁 Module Structure

```
data_preparation/
├── pipeline_Version2.py          # Main pipeline orchestrator
├── schema_Version2.py            # Pydantic data models
├── gpt_adapter.py                # OpenAI GPT-4 API adapter
├── qwen_adapter_Version2.py      # Qwen API adapter (alternative)
├── utils_Version2.py             # Crawling, extraction, chunking utilities
├── seeds_example_Version2.txt    # Example seed URLs (174 programs)
├── requirements_Version2.txt     # Python dependencies
├── README_data_preparation.md    # This file
└── dataset/
    └── graduate_programs/        # Output: 67+ university domains
        ├── www.cs.cmu.edu/
        ├── seas.harvard.edu/
        ├── www.gsb.stanford.edu/
        └── ...
```

---

## 🏗️ Architecture

### **Pipeline Flow**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Data Preparation Pipeline                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1: CRAWL MAIN PAGE                                                │
│  ├── HTTP request with rate limiting                                    │
│  ├── Save raw snapshot to out/cache/                                    │
│  └── Extract HTML/PDF content                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2: DISCOVER SUBPAGES (v2)                                         │
│  ├── find_candidate_course_subpages()                                   │
│  ├── Keywords: curriculum, courses, degree requirements, etc.           │
│  └── Max 5 subpage links (same domain only)                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 3: DISCOVER COURSE DETAILS (v3)                                   │
│  ├── extract_course_names_from_text()                                   │
│  ├── generate_possible_detail_urls()                                    │
│  ├── try_fetch_detail_page()                                            │
│  └── Bounded: max 8 courses × 2 URLs each                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 4: MERGE & PREFILTER                                              │
│  ├── Combine main + subpages + course details                           │
│  ├── simple_prefilter() — check for program-related keywords            │
│  └── chunk_text_by_words() — 400 words, 100 overlap                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 5: GPT-4 EXTRACTION                                               │
│  ├── Check LLM cache first (prompt_hash + checksum)                     │
│  ├── call_gpt_chat() with strict JSON schema                            │
│  ├── try_parse_json() — handle markdown wrappers                        │
│  └── validate_and_enrich() — post-processing                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 6: SAVE CORPUS DOC                                                │
│  ├── Build CorpusDoc with extracted_fields                              │
│  ├── Save to dataset/graduate_programs/{domain}/                        │
│  └── Filename: {timestamp}_{checksum}.json                              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Schema (V2)

### **CorpusDoc** — Main Document Structure

```python
class CorpusDoc(BaseModel):
    id: str                              # {domain}_{checksum[:12]}
    source_url: str                      # Original URL
    raw_snapshot_path: str               # Path to cached HTML/PDF
    content_type: str                    # text/html, application/pdf
    crawl_date: datetime                 # UTC timestamp
    checksum: str                        # SHA-256 of raw content
    
    title: Optional[str]                 # Page title
    raw_text: Optional[str]              # Extracted plain text
    language: Optional[str]              # en, zh, etc.
    
    schema_version: str = "v2.0"         # Schema version marker
    
    extracted_fields: ExtractedFields    # ★ Structured program data
    snippets: Dict[str, Snippet]         # Source text evidence
    chunks: List[Chunk]                  # Text chunks for RAG
    llm_calls: List[LLMCall]             # LLM call audit log
```

### **ExtractedFields** — Structured Program Information

```python
class ExtractedFields(BaseModel):
    # Basic Info
    program_name: Optional[str]          # "M.S. in Computer Science"
    school: Optional[str]                # "Stanford University"
    department: Optional[str]            # "Department of Computer Science"
    duration: Optional[str]              # "2 years"
    tuition: Optional[str]               # "$55,000/year"
    contact_email: Optional[str]         # Validated email
    language: Optional[str]              # Instruction language
    
    # Courses (V2: name + description objects)
    courses: Optional[List[Course]]
    # Course = { name: str, description: Optional[str] }
    
    # ★ Structured Application Requirements
    application_requirements: Optional[ApplicationRequirements]
    
    # ★ Program Background & Environment
    program_background: Optional[ProgramBackground]
    
    # ★ Training Outcomes & Career Paths
    training_outcomes: Optional[TrainingOutcomes]
    
    others: Optional[Dict[str, Any]]     # Catch-all for extra fields
```

### **Sub-Models (V2 Enhanced)**

| Model | Fields |
|-------|--------|
| **ApplicationRequirements** | `academic_background`, `prerequisites`, `gre`, `english_tests`, `research_experience`, `work_experience`, `documents`, `summary` |
| **ProgramBackground** | `mission`, `environment`, `faculty`, `resources`, `summary` |
| **TrainingOutcomes** | `goals`, `career_paths`, `research_orientation`, `professional_orientation`, `summary` |
| **Course** | `name`, `description` |

---

## 🔧 Core Components

### **1. pipeline_Version2.py** — Main Orchestrator

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `process_seed(url)` | Process a single URL through the full pipeline |
| `prompt_for_text(text, url)` | Build GPT messages with schema and instructions |
| `try_parse_json(text)` | Parse JSON from GPT response (handles markdown) |
| `validate_and_enrich(extracted, text)` | Post-process: email validation, school inference, course normalization |
| `save_final_doc(doc, domain, checksum)` | Save CorpusDoc to filesystem |

**Prompt Design:**

```python
SYSTEM_PROMPT = """
你是一个严格的结构化信息抽取器。
1. 严格按照给定 schema 输出合法 JSON
2. 所有内容必须来自提供的网页文本，禁止编造
3. courses 字段：课程名必须是网页中真实出现的名称
4. application_requirements：结构化提取 GPA/GRE/语言/材料等
5. program_background：教育理念、师资、资源
6. training_outcomes：培养目标、职业路径
7. school：必须是大学或学院名称，不要泛泛称呼
8. 找不到的字段显式使用 null
"""
```

### **2. utils_Version2.py** — Crawling & Extraction Engine

**Crawling Functions:**

| Function | Purpose |
|----------|---------|
| `crawl_url(url)` | HTTP GET with rate limiting (0.3s delay) |
| `save_snapshot(url, content, ct)` | Save raw HTML/PDF to `out/cache/` |
| `canonicalize_url(url)` | Remove URL fragments |

**Text Extraction:**

| Function | Purpose |
|----------|---------|
| `extract_text_from_html_bytes(html)` | BeautifulSoup parsing, strip noise tags |
| `extract_text_from_pdf_path(path)` | PyMuPDF (fitz) PDF text extraction |
| `simple_prefilter(text)` | Heuristic check for program-related content |

**V2: Subpage Discovery:**

```python
def find_candidate_course_subpages(html_bytes, base_url, max_links=8):
    """
    Find links containing: curriculum, course, degree requirements,
    plan of study, course list, etc.
    Same-domain only, max 8 links.
    """
```

**V3: Course Detail Discovery:**

```python
def discover_course_detail_pages(text, base_domain, max_courses=8, max_candidates_per_course=2):
    """
    1. extract_course_names_from_text() — regex for course-like names
    2. generate_possible_detail_urls() — /coursecatalog/{slug}, /courses/{slug}
    3. try_fetch_detail_page() — fetch and validate
    
    Bounded: max 8 courses × 2 URLs = 16 requests max
    """
```

**Chunking:**

```python
def chunk_text_by_words(text, max_words=400, overlap=100):
    """
    Word-based overlapping chunks with char offsets.
    Returns: [{ chunk_id, text, word_count, char_start, char_end, token_count }]
    """
```

**Caching:**

```python
def save_llm_cache(prompt_hash, checksum, response_text)  # Save to out/llm_cache/
def load_llm_cache(prompt_hash, checksum) -> str | None   # Load if exists
```

### **3. gpt_adapter.py** — OpenAI GPT-4 Adapter

```python
def call_gpt_chat(messages, model="gpt-4o", temperature=0.0, max_tokens=2000):
    """
    OpenAI Chat Completions wrapper.
    - Force model = "gpt-4o" for stability
    - temperature=0 for deterministic extraction
    - Proper error handling and logging
    """
```

**Environment Variable:** `OPENAI_API_KEY` (required)

### **4. qwen_adapter_Version2.py** — Alternative LLM

```python
def call_qwen_chat(messages, model="qwen-7b-chat", temperature=0.0, max_tokens=2000):
    """Alternative adapter for Qwen API."""
```

**Environment Variables:** `QWEN_API_URL`, `QWEN_API_KEY`

---

## 🚀 Quick Start

### **1. Environment Setup**

```bash
cd data_preparation
pip install -r requirements_Version2.txt

# Set API key
export OPENAI_API_KEY="sk-..."
# Or for Windows:
set OPENAI_API_KEY=sk-...
```

### **2. Prepare Seeds File**

Create a text file with one URL per line:

```plaintext
# seeds.txt
https://www.cs.stanford.edu/academics/masters
https://seas.harvard.edu/masters-data-science
https://www.cs.cmu.edu/masters-programs
```

### **3. Run Pipeline**

```bash
python pipeline_Version2.py seeds.txt
```

### **4. Check Output**

```
dataset/graduate_programs/
├── www.cs.stanford.edu/
│   └── 20241201T120000Z_a1b2c3d4e5f6.json
├── seas.harvard.edu/
│   └── 20241201T120100Z_b2c3d4e5f6g7.json
└── www.cs.cmu.edu/
    └── 20241201T120200Z_c3d4e5f6g7h8.json
```

---

## 📂 Output Example

```json
{
  "id": "www.cs.stanford.edu_a1b2c3d4e5f6",
  "source_url": "https://www.cs.stanford.edu/academics/masters",
  "raw_snapshot_path": "out/cache/20241201T120000Z_a1b2c3d4.html",
  "content_type": "text/html",
  "crawl_date": "2024-12-01T12:00:00Z",
  "checksum": "a1b2c3d4e5f6...",
  "schema_version": "v2.0",
  
  "extracted_fields": {
    "program_name": "Master of Science in Computer Science",
    "school": "Stanford University",
    "department": "Department of Computer Science",
    "duration": "1-2 years",
    
    "courses": [
      { "name": "CS 229 - Machine Learning", "description": "Introduction to machine learning techniques..." },
      { "name": "CS 231N - Computer Vision", "description": "Deep learning for visual recognition..." },
      { "name": "CS 224N - NLP", "description": null }
    ],
    
    "application_requirements": {
      "academic_background": "BS in CS or related field",
      "prerequisites": "Linear algebra, probability, programming",
      "gre": "Not required but recommended",
      "english_tests": "TOEFL 100+ or IELTS 7.0+",
      "research_experience": "Recommended for research track",
      "work_experience": null,
      "documents": "Transcripts, 3 recommendation letters, SOP, CV",
      "summary": "Strong technical background required..."
    },
    
    "program_background": {
      "mission": "Train future CS leaders and researchers",
      "environment": "Small cohorts, close faculty mentorship",
      "faculty": "World-renowned AI/Systems/Theory faculty",
      "resources": "Access to Stanford AI Lab, industry partnerships",
      "summary": null
    },
    
    "training_outcomes": {
      "goals": "Develop advanced CS skills for research or industry",
      "career_paths": "Tech companies, startups, PhD programs, research labs",
      "research_orientation": "Optional thesis track for PhD preparation",
      "professional_orientation": "Coursework track for industry careers",
      "summary": null
    },
    
    "tuition": "$57,861/year",
    "contact_email": "mscs-admissions@cs.stanford.edu",
    "language": "English"
  },
  
  "chunks": [
    { "chunk_id": "chunk_0000", "text": "...", "char_start": 0, "char_end": 2000 },
    { "chunk_id": "chunk_0001", "text": "...", "char_start": 1800, "char_end": 3800 }
  ],
  
  "llm_calls": [
    { "prompt_hash": "abc123...", "model": "gpt-4o", "date": "2024-12-01T12:00:00Z", "cached": false }
  ]
}
```

---

## 📈 Dataset Statistics

| Metric | Value |
|--------|-------|
| **University Domains** | 67+ |
| **Seed URLs** | 174 (in example file) |
| **Fields per Program** | 15+ structured fields |
| **Schema Version** | V2.0 |

**Sample Universities:**
- MIT, Stanford, Harvard, CMU, Berkeley
- Columbia, Cornell, Princeton, Yale, Duke
- Caltech, Northwestern, Johns Hopkins, NYU
- UCLA, UCSD, UT Austin, UW, Purdue

---

## 🔍 Debugging & Troubleshooting

### **Common Issues**

| Issue | Solution |
|-------|----------|
| **Empty extracted text** | Page may be JS-rendered; use Playwright to capture |
| **LLM returns non-JSON** | Check `out/llm_cache/` for raw response; adjust prompt |
| **Invalid email** | Auto-set to null by `validate_and_enrich()` |
| **Generic school name** | Pipeline attempts to infer from text patterns |
| **Too many requests** | Adjust `max_links`, `max_courses` parameters |

### **Cache Directories**

```
out/
├── cache/          # Raw HTML/PDF snapshots
│   └── 20241201T120000Z_a1b2c3d4.html
└── llm_cache/      # GPT response cache
    └── abc123def456.txt
```

### **Re-running with Cache**

The pipeline automatically uses cached LLM responses if the `(prompt_hash, checksum)` matches. To force re-extraction:

```bash
# Clear LLM cache
rm -rf out/llm_cache/*

# Then run pipeline
python pipeline_Version2.py seeds.txt
```

---

## 🛡️ Best Practices

1. **Start Small**: Test with 10-20 URLs first, manually verify output
2. **Use Specific URLs**: Program/curriculum pages work better than main pages
3. **Rate Limiting**: Built-in 0.3s delay; don't disable for production
4. **Secure Keys**: Add `out/`, `vectors/`, API keys to `.gitignore`
5. **Monitor Costs**: GPT-4 API calls; use cache to minimize

---

## 🔗 Integration with Main System

The output JSON files are used by:

| Component | Usage |
|-----------|-------|
| **Matching Service (V2)** | Loads `extracted_fields` for 6-dimension scoring |
| **RAG Service** | Uses `chunks` for retrieval-augmented generation |
| **Writing Agent** | References program details for document generation |

**Data Flow:**

```
data_preparation/dataset/ → data/corpus/ → API → Frontend
```

---

## 📚 Dependencies

```
requests          # HTTP client
beautifulsoup4    # HTML parsing
lxml              # Fast HTML parser
pydantic          # Data validation
sentence-transformers  # (Optional) Embeddings
PyMuPDF           # PDF extraction
tqdm              # Progress bars
```
    model: str
    date: datetime
    response_raw_path: Optional[str] = None
    cached: bool = False

class Course(BaseModel):
    name: str
    description: Optional[str] = None

class ExtractedFields(BaseModel):
    program_name: Optional[str] = None
    duration: Optional[str] = None
    # courses is now a list of Course objects: {"name": "...", "description": "..."}
    courses: Optional[List[Course]] = None
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
    schema_version: str = "v1.1"
    extracted_fields: Optional[ExtractedFields] = None
    snippets: Optional[Dict[str, Snippet]] = None
    chunks: Optional[List[Chunk]] = None
    llm_calls: Optional[List[LLMCall]] = None
    provenance_notes: Optional[str] = None
    notes: Optional[str] = None
```

```python name=pipeline_Version2.py
#!/usr/bin/env python3
"""
Main pipeline (PoC) — updated to use Course objects for courses (name + description).

Usage:
  python pipeline_Version2.py seeds.txt

Process per seed URL:
  - crawl & snapshot
  - normalize (html/pdf -> text)
  - prefilter
  - chunk
  - aggregate chunks into a context and call Qwen once
  - parse JSON response, validate, normalize courses, and save final JSON
"""
import os
import sys
import json
import re
import hashlib
from datetime import datetime
from urllib.parse import urlparse
from schema_Version2 import CorpusDoc, ExtractedFields, Snippet, Chunk, LLMCall
import utils_Version2 as utils
from qwen_adapter_Version2 import call_qwen_chat

SYSTEM_PROMPT = (
    "你是严格的结构化信息抽取器。只返回合法 JSON，字段遵循给定 schema。"
    "对每个字段必须尽量返回支持该字段的原文片段（snippet）及其 char_start/char_end。"
    "若字段不存在返回 null。不要输出任何多余文字。"
)

# Add a clear schema and a small few-shot example to encourage courses as objects
USER_SCHEMA_DESC = """
schema:
{
  "program_name": "string|null",
  "duration": "string|null",
  "courses": [ { "name": "string", "description": "string|null" } ]|null,
  "tuition": "string|null",
  "application_requirements": "string|null",
  "features": "string|null",
  "contact_email": "string|null",
  "language": "string|null",
  "source_url": "string"
}

Example:
{
  "program_name": "M.S. in Computer Science",
  "duration": "2 years",
  "courses": [
    { "name": "Algorithms", "description": "Core algorithms course covering graph algorithms, dynamic programming, complexity analysis." },
    { "name": "Machine Learning", "description": "Intro to ML: supervised and unsupervised methods, basic deep learning." },
    { "name": "Operating Systems", "description": null }
  ],
  "tuition": "$55,000 per year",
  "application_requirements": "GPA, TOEFL/IELTS, transcripts, recommendation letters",
  "features": "Research-focused; industry internships available",
  "contact_email": "cs-grad@example.edu",
  "language": "English",
  "source_url": "https://example.edu/grad/programs/mscs"
}
"""

def prompt_for_text(page_text: str, source_url: str) -> list:
    user = f"{USER_SCHEMA_DESC}\n\nsource_url: {source_url}\n\n<<<文档开始>>>\n{page_text}\n<<<文档结束>>>"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user}
    ]

def prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256((system + "\n" + user).encode("utf-8")).hexdigest()

def try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'(\{.*\})', text, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
        return None

def validate_and_enrich(extracted: dict, page_text: str) -> dict:
    """
    - basic validations (email, duration)
    - snippet sanity check (snippet must appear in page_text)
    - normalize courses: if returned as list of strings, convert to list of {name, description:null}
    """
    notes = extracted.get("notes") or ""
    # email validation
    email = extracted.get("contact_email")
    if email:
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            extracted["contact_email"] = None
            notes += "contact_email_invalid;"
    # duration normalization example
    if extracted.get("duration"):
        m = re.search(r'(\d+\s*(year|years|month|months))', str(extracted["duration"]), re.I)
        if m:
            extracted.setdefault("duration_normalized", m.group(1))
    # snippet sanity
    snippets = extracted.get("snippets") or {}
    for k, v in snippets.items():
        if v and isinstance(v, dict) and "text" in v:
            if v["text"] not in page_text:
                notes += f"snippet_mismatch_{k};"
    # normalize courses: handle several possible formats from LLM
    courses = extracted.get("courses")
    if courses is None:
        extracted["courses"] = None
    else:
        # If courses is a list of strings, convert
        if isinstance(courses, list) and len(courses) > 0 and isinstance(courses[0], str):
            extracted["courses"] = [{"name": c, "description": None} for c in courses]
        elif isinstance(courses, list):
            # ensure each element is dict with name + description
            normalized = []
            for item in courses:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("course") or None
                    desc = item.get("description") if "description" in item else None
                    if name:
                        normalized.append({"name": name, "description": desc})
                elif isinstance(item, str):
                    normalized.append({"name": item, "description": None})
            extracted["courses"] = normalized if normalized else None
        else:
            # unexpected type, drop
            extracted["courses"] = None
            notes += "courses_unexpected_type;"
    extracted["notes"] = notes if notes else None
    if extracted.get("confidence") is None:
        extracted["confidence"] = 0.5
    return extracted

def domain_slug(url: str) -> str:
    netloc = urlparse(url).netloc
    return netloc.replace(":", "_")

def save_final_doc(doc: CorpusDoc, domain: str, checksum: str):
    dirpath = os.path.join("data", "corpus", domain)
    os.makedirs(dirpath, exist_ok=True)
    filename = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{checksum[:12]}.json"
    path = os.path.join(dirpath, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc.json(ensure_ascii=False, indent=2))
    return path

def process_seed(url: str):
    try:
        print("Crawling:", url)
        snapshot_path, content, content_type = utils.crawl_url(url)
        domain = domain_slug(url)
        checksum = utils.sha256_bytes(content)
        title = ""
        text = ""
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            text = utils.extract_text_from_pdf_path(snapshot_path)
        else:
            title, text = utils.extract_text_from_html_bytes(content)
        # prefilter
        if not utils.simple_prefilter(text):
            print("Prefilter: unlikely target page, skipping LLM. Saving minimal doc.")
        # chunking
        chunks_raw = utils.chunk_text_by_words(text or "", max_words=400, overlap=100)
        # aggregate relevant text for single LLM call (cap length)
        combined = ""
        for c in chunks_raw:
            combined += c["text"] + "\n\n"
            if len(combined) > 30000:
                break
        combined = combined[:30000]
        messages = prompt_for_text(combined, url)
        phash = prompt_hash(SYSTEM_PROMPT, messages[1]["content"])
        # cache check
        cached = False
        cached_resp = utils.load_llm_cache(phash, checksum)
        resp_text = None
        if cached_resp:
            resp_text = cached_resp
            cached = True
            print("Using cached LLM response.")
        else:
            try:
                resp_text = call_qwen_chat(messages, model="qwen-7b-chat", temperature=0.0, max_tokens=2000)
            except Exception as e:
                print("LLM call failed:", e)
                resp_text = None
        llm_response_path = None
        parsed = None
        if resp_text:
            llm_response_path = utils.save_llm_cache(phash, checksum, resp_text)
            parsed = try_parse_json(resp_text)
        if parsed is None:
            parsed = {
                "program_name": None, "duration": None, "courses": None,
                "tuition": None, "application_requirements": None, "features": None,
                "contact_email": None, "language": None, "source_url": url,
                "snippets": {}, "confidence": 0.0, "notes": "llm_failed_or_no_json"
            }
        parsed = validate_and_enrich(parsed, text)
        # embeddings (optional) - PoC saves vectors to files
        chunks = []
        texts_for_embed = [c["text"] for c in chunks_raw]
        embeddings = None
        if texts_for_embed:
            try:
                embeddings = utils.get_embedding_for_texts(texts_for_embed)
            except Exception:
                embeddings = None
        for i, c in enumerate(chunks_raw):
            vector_id = None
            if embeddings is not None:
                import numpy as np
                os.makedirs("vectors", exist_ok=True)
                vec_fname = f"vectors/{domain}_{checksum[:8]}_chunk{i:04d}.npy"
                np.save(vec_fname, embeddings[i])
                vector_id = vec_fname
            ch = Chunk(
                chunk_id=f"{domain}_{checksum[:8]}_{i:04d}",
                text=c["text"],
                char_start=c["char_start"],
                char_end=c["char_end"],
                token_count=c["token_count"],
                vector_id=vector_id
            )
            chunks.append(ch)
        doc = CorpusDoc(
            id=f"{domain}_{checksum[:12]}",
            source_url=url,
            raw_snapshot_path=snapshot_path,
            content_type=content_type,
            crawl_date=datetime.utcnow(),
            checksum=checksum,
            language=parsed.get("language"),
            title=title,
            raw_text=text,
            extracted_fields=ExtractedFields(**{k:v for k,v in parsed.items() if k in ExtractedFields.__fields__}),
            snippets={k: Snippet(**v) for k,v in (parsed.get("snippets") or {}).items()},
            chunks=chunks,
            llm_calls=[LLMCall(prompt_hash=phash, model="qwen-7b-chat", date=datetime.utcnow(), response_raw_path=llm_response_path, cached=cached)],
            provenance_notes="PoC pipeline: crawl + Qwen extraction + optional sentence-transformers embedding.",
            notes=None
        )
        out_path = save_final_doc(doc, domain, checksum)
        print("Saved final doc:", out_path)
    except Exception as e:
        print("Error processing seed:", url, e)

def main(seeds_file: str):
    with open(seeds_file, "r", encoding="utf-8") as f:
        seeds = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    for s in seeds:
        process_seed(s)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py seeds.txt")
        sys.exit(1)
    main(sys.argv[1])