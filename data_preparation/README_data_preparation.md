```markdown
# Data Preparation — 高校官网语料库（Courses as name+description）

说明
- 本说明针对已修改的 PoC pipeline（pipeline_Version2.py / schema_Version2.py）。
- 目标：把高校项目页面抓取并抽取为结构化 JSON，其中 `courses` 字段为对象数组，每项包含 course `name` 与可选 `description`。

1. 环境准备
- Python 3.9+
- 安装依赖：
  pip install -r requirements_Version2.txt

- 环境变量（至少）：
  - QWEN_API_URL (例如 https://api.qwen.example/v1/chat/completions)
  - QWEN_API_KEY

2. 准备 seeds.txt
- 每行一个 URL（可以是 program 页面、课程页、招生页或招生手册 PDF）。
- 示例文件： `seeds_example_Version2.txt`

3. 运行命令
- 运行 PoC：
  python pipeline_Version2.py seeds_example_Version2.txt

4. 输入/输出位置
- 输入：seeds_example_Version2.txt（或你自备的 seeds.txt）
- 快照（保存原始 HTML/PDF）： `out/cache/<timestamp>_<sha>.html|pdf`
- LLM 原始响应缓存： `out/llm_cache/<key>.txt`
- Embeddings（PoC 本地保存，可选）： `vectors/*.npy`
- 最终 JSON 输出： `data/corpus/<domain>/<timestamp>_<checksum12>.json`

5. JSON 结构 (重点)
- 最小必备字段：
  - id, source_url, raw_snapshot_path, content_type, crawl_date, checksum, title, raw_text
  - extracted_fields: 包含：
    - program_name (string|null)
    - duration (string|null)
    - courses (array of objects | null)
      - each course: { name: string, description: string|null }
    - tuition, application_requirements, features, contact_email, language
  - snippets: 每个字段对应支持片段 { text, char_start, char_end }
  - chunks: 切片列表，含 char offsets 与 vector_id（可选）
  - llm_calls: prompt_hash, model, date, response_raw_path, cached
  - provenance_notes, notes

6. LLM Prompt 与行为要点
- Pipeline 在调用 Qwen 时使用 temperature=0，并给出严格 schema（含 courses 对象示例）与 few-shot 示例（已内置在 pipeline_Version2.py），要求返回合法 JSON 且为每字段返回支持片段（snippet+char offsets）。
- 若 LLM 返回旧格式（courses 为字符串列表），pipeline 会自动将其转换为对象数组（name=原字符串, description=null）。

7. 常见操作/调试
- 若 Qwen 返回非 JSON，请先查看 `out/llm_cache` 中对应 raw response 文件，检查实际返回文本并调整 system prompt 或 few-shot 示例。
- 若抓取到的是动态渲染页面（空白正文），使用 Playwright 手动抓取该页面并将 snapshot 路径替入 seeds。
- 若出现大量重复或页面更新，可通过 `checksum` 对比决定是否重新抽取。

8. 示例 courses 输出片段（示例 JSON 节选）
```json
"extracted_fields": {
  "program_name": "M.S. in Computer Science",
  "duration": "2 years",
  "courses": [
    { "name": "Algorithms", "description": "Core algorithms covering graph, DP, complexity" },
    { "name": "Machine Learning", "description": "Intro to ML: supervised, unsupervised, deep learning basics" },
    { "name": "Operating Systems", "description": null }
  ],
  "tuition": "$55,000 per year",
  "application_requirements": "GPA, TOEFL/IELTS, transcripts, recommendations",
  "features": "Research opportunities; internship required",
  "contact_email": "cs-grad@example.edu",
  "language": "English"
}
```

9. 最佳实践建议（简短）
- 优先在 seeds 中放项目/课程的具体页面（比学校主页更高命中率）。
- 先运行小批量（10–20 个 URL）做人工抽样验证，调整 few-shot 和规则。
- 把敏感信息（API keys、out/cache、vectors）列入 `.gitignore` 并不要上传到远程仓库。
```
```


```python name=schema_Version2.py
from pydantic import BaseModel
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