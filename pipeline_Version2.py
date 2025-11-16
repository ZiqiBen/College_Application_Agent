#!/usr/bin/env python3
"""
Main pipeline (PoC):
  python pipeline.py seeds.txt

Process per seed URL:
  - crawl & snapshot
  - normalize (html/pdf -> text)
  - prefilter
  - chunk
  - aggregate chunks into a context and call Qwen once
  - parse JSON response, validate, and save final JSON
"""
import os
import sys
import json
import re
import hashlib
from datetime import datetime
from urllib.parse import urlparse
from schema import CorpusDoc, ExtractedFields, Snippet, Chunk, LLMCall
import utils
from qwen_adapter import call_qwen_chat

SYSTEM_PROMPT = (
    "你是严格的结构化信息抽取器。只返回合法 JSON，字段遵循给定 schema。"
    "对每个字段必须尽量返回支持该字段的原文片段（snippet）及其 char_start/char_end。"
    "若字段不存在返回 null。不要输出任何多余文字。"
)

USER_SCHEMA_DESC = """
schema:
{
  "program_name": "string|null",
  "duration": "string|null",
  "courses": ["string"]|null,
  "tuition": "string|null",
  "application_requirements": "string|null",
  "features": "string|null",
  "contact_email": "string|null",
  "language": "string|null",
  "source_url": "string"
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
    import json, re
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
    notes = extracted.get("notes") or ""
    # email validation
    email = extracted.get("contact_email")
    if email:
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            extracted["contact_email"] = None
            notes += "contact_email_invalid;"
    # credits or duration - basic normalization example
    if extracted.get("duration"):
        m = re.search(r'(\d+\s*(year|years|month|months))', extracted["duration"], re.I)
        if m:
            extracted.setdefault("duration_normalized", m.group(1))
    # snippet sanity
    snippets = extracted.get("snippets") or {}
    for k, v in snippets.items():
        if v and "text" in v:
            if v["text"] not in page_text:
                notes += f"snippet_mismatch_{k};"
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