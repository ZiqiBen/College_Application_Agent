#!/usr/bin/env python3
"""
Main pipeline (PoC) using OpenAI GPT:
  python pipeline_Version2.py seeds_example_Version2.txt

Process per seed URL:
  - crawl & snapshot
  - normalize (html/pdf -> text)
  - prefilter
  - chunk
  - aggregate chunks into a context and call GPT once
  - parse JSON response, normalize, validate, and save final JSON
"""

import os
import sys
import json
import re
import hashlib
from datetime import datetime
from urllib.parse import urlparse

from schema_Version2 import CorpusDoc, ExtractedFields, Snippet, Chunk, LLMCall
import utils_Version2 as utils  # 给 utils_Version2 起别名为 utils，方便后面统一调用
from gpt_adapter import call_gpt_chat, DEFAULT_MODEL as GPT_MODEL_NAME

# ==========================
# Prompts / 提示词
# ==========================

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
  "courses": [ { "name": "string", "description": "string|null" } ]|null,
  "tuition": "string|null",
  "application_requirements": "string|null",
  "features": "string|null",
  "contact_email": "string|null",
  "language": "string|null",
  "source_url": "string",
  "snippets": {
      "program_name": { "snippet": "string", "char_start": int, "char_end": int },
      ...
  }
}
"""


def prompt_for_text(page_text: str, source_url: str) -> list:
    """
    Build chat messages for GPT 调用 GPT 的 messages 列表
    """
    user = (
        f"{USER_SCHEMA_DESC}\n\n"
        f"source_url: {source_url}\n\n"
        f"<<<文档开始>>>\n{page_text}\n<<<文档结束>>>"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def prompt_hash(system: str, user: str) -> str:
    """
    Hash of (system + user prompt), for caching purposes.
    用于 LLM 缓存的 prompt 哈希
    """
    return hashlib.sha256((system + "\n" + user).encode("utf-8")).hexdigest()


def try_parse_json(text: str):
    """
    Try to parse the model output as JSON.
    尝试从模型输出中解析 JSON（如果有多余文字，尽量截取 {...} 部分）
    """
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{.*\})", text, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
        return None


# ==========================
# JSON 规范化 / 校验
# ==========================

def normalize_parsed_fields(parsed: dict) -> dict:
    """
    Normalize LLM JSON so that:
      - string fields (program_name, features, language, etc.) are plain strings or None
      - any dict-valued field like
          {"snippet": "...", "char_start": ..., "char_end": ...}
        or  {"value": "...", "snippet": "...", ...}
        is moved to parsed["snippets"][field_name]

    将 LLM 返回的 dict 做一层规范化：
      - 避免 program_name / features / language 等字段是 dict
      - 如果字段是 {"snippet": ...} 或 {"value": ...}，
        则移动到 snippets[field_name]，字段本身变成一个简单字符串（或 None）
    """
    if not isinstance(parsed, dict):
        return {}

    # 这些字段在 ExtractedFields 里是纯字符串
    text_fields = [
        "program_name",
        "duration",
        "tuition",
        "application_requirements",
        "features",
        "contact_email",
        "language",
    ]

    snippets = parsed.get("snippets") or {}

    for field in text_fields:
        if field not in parsed:
            continue
        val = parsed[field]

        # 如果本来就是字符串或 None，直接跳过
        if isinstance(val, (str, type(None))):
            continue

        # 如果是 dict，我们尝试从中抽取 snippet / value
        if isinstance(val, dict):
            # snippet 文本的候选
            snippet_text = (
                val.get("snippet")
                or val.get("value")
                or ""
            )
            char_start = val.get("char_start")
            char_end = val.get("char_end")

            # 规范化成 Snippet 需要的结构：text, char_start, char_end
            snippets[field] = {
                "text": snippet_text,
                "char_start": -1 if char_start is None else char_start,
                "char_end": -1 if char_end is None else char_end,
            }

            # 字段本身设为一个“干净”的文本（或者你也可以设为 snippet_text）
            parsed[field] = snippet_text or None

        else:
            # 其他类型（list, int, etc.）不符合预期，先强制转成字符串
            parsed[field] = str(val)

    parsed["snippets"] = snippets
    return parsed


def validate_and_enrich(extracted: dict, page_text: str) -> dict:
    """
    对 LLM 抽取结果做一些简单校验和后处理：
      - email 格式校验
      - duration 简单标准化
      - snippet 文本是否真的出现在原文中
      - courses 字段从 list[str] 转为 list[{"name": ..., "description": ...}]
      - 填补 confidence / notes
    """
    notes = extracted.get("notes") or ""

    # email validation / 邮箱格式校验
    email = extracted.get("contact_email")
    if email:
        if not isinstance(email, str) or not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            extracted["contact_email"] = None
            notes += "contact_email_invalid;"

    # duration normalization / 学制正则抽取
    duration = extracted.get("duration")
    if isinstance(duration, str):
        m = re.search(r"(\d+\s*(year|years|month|months))", duration, re.I)
        if m:
            extracted.setdefault("duration_normalized", m.group(1))

    # snippet sanity / 片段是否真的在原文中
    snippets = extracted.get("snippets") or {}
    for k, v in snippets.items():
        if isinstance(v, dict) and "text" in v:
            snippet_text = v["text"]
            if isinstance(snippet_text, str) and snippet_text:
                if snippet_text not in page_text:
                    notes += f"snippet_mismatch_{k};"

    # normalize courses: 如果 LLM 返回 list[str]，转成 list[dict]
    courses = extracted.get("courses")
    if courses and isinstance(courses, list):
        if len(courses) > 0 and isinstance(courses[0], str):
            extracted["courses"] = [
                {"name": c, "description": None} for c in courses
            ]

    # 缺省 confidence
    if extracted.get("confidence") is None:
        extracted["confidence"] = 0.5

    extracted["notes"] = notes if notes else None
    return extracted


# ==========================
# Helpers / 其他辅助函数
# ==========================

def domain_slug(url: str) -> str:
    """
    Turn a URL into a filesystem-friendly domain slug.
    """
    netloc = urlparse(url).netloc
    return netloc.replace(":", "_")


def save_final_doc(doc: CorpusDoc, domain: str, checksum: str):
    """
    Serialize a CorpusDoc into a JSON file.

    将最终结构化结果 CorpusDoc 序列化为 JSON 文件，保存路径格式如下：
        dataset/graduate_programs/{domain}/
            {UTC时间}_{前12位checksum}.json

    This keeps the dataset clean and consistent:
    - Raw snapshots (HTML/PDF) remain under out/cache/
    - LLM raw outputs stay in out/llm_cache/
    - Final structured corpus goes to dataset/graduate_programs/
    """
    BASE_CORPUS_DIR = "dataset/graduate_programs"

    # 1) 目录：dataset/graduate_programs/{domain}
    dirpath = os.path.join(BASE_CORPUS_DIR, domain)
    os.makedirs(dirpath, exist_ok=True)

    # 2) 文件名：{UTC时间}_{前12位checksum}.json
    filename = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{checksum[:12]}.json"
    path = os.path.join(dirpath, filename)

    # 3) 用 Pydantic v2 -> Python dict
    data = doc.model_dump(mode="python")

    # 4) 用标准库 json 序列化，default=str 解决 datetime 无法序列化问题
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,  # 保留中文，不转义
            indent=2,            # pretty print
            default=str,         # <- 关键：datetime 等特殊类型转成字符串
        )

    return path


# ==========================
# 核心处理逻辑：每个 seed URL
# ==========================

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

        # Prefilter / 预筛选（现在只是打印 warning，不真正跳过）
        if not utils.simple_prefilter(text):
            print("Prefilter: unlikely target page, but still proceeding with GPT for PoC.")

        # Chunking / 文本切块
        chunks_raw = utils.chunk_text_by_words(text or "", max_words=400, overlap=100)

        # Aggregate relevant text for single LLM call (cap length).
        combined = ""
        for c in chunks_raw:
            combined += c["text"] + "\n\n"
            if len(combined) > 30000:
                break
        combined = combined[:30000]

        # Build LLM messages
        messages = prompt_for_text(combined, url)
        phash = prompt_hash(SYSTEM_PROMPT, messages[1]["content"])

        # Cache check / LLM 调用缓存
        cached = False
        cached_resp = utils.load_llm_cache(phash, checksum)
        resp_text = None

        if cached_resp:
            resp_text = cached_resp
            cached = True
            print("Using cached LLM response.")
        else:
            try:
                resp_text = call_gpt_chat(
                    messages,
                    model=GPT_MODEL_NAME,
                    temperature=0.0,
                    max_tokens=2000,
                )
            except Exception as e:
                print("LLM call failed:", e)
                resp_text = None

        llm_response_path = None
        parsed = None

        if resp_text:
            llm_response_path = utils.save_llm_cache(phash, checksum, resp_text)
            parsed = try_parse_json(resp_text)

        # 如果完全解析失败，用一个兜底空结构
        if not isinstance(parsed, dict):
            parsed = {
                "program_name": None,
                "duration": None,
                "courses": None,
                "tuition": None,
                "application_requirements": None,
                "features": None,
                "contact_email": None,
                "language": None,
                "source_url": url,
                "snippets": {},
                "confidence": 0.0,
                "notes": "llm_failed_or_no_json",
            }

        # 先做字段规范化，再做校验 / 富化
        parsed = normalize_parsed_fields(parsed)
        parsed = validate_and_enrich(parsed, text or "")

        # Embeddings for chunks (optional)
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
                vector_id=vector_id,
            )
            chunks.append(ch)

        # 构造 snippets 对象（注意做一次防守性处理）
        raw_snippets = parsed.get("snippets") or {}
        snippet_objs = {}
        for k, v in raw_snippets.items():
            if not isinstance(v, dict):
                continue
            text_val = v.get("text", "") if isinstance(v.get("text"), str) else ""
            cs = v.get("char_start", -1)
            ce = v.get("char_end", -1)
            try:
                cs_int = int(cs) if cs is not None else -1
            except Exception:
                cs_int = -1
            try:
                ce_int = int(ce) if ce is not None else -1
            except Exception:
                ce_int = -1
            snippet_objs[k] = Snippet(
                text=text_val,
                char_start=cs_int,
                char_end=ce_int,
            )

        # 构造 ExtractedFields（使用 model_fields 替代 __fields__ 以兼容 Pydantic v2）
        extracted_fields = ExtractedFields(
            **{k: v for k, v in parsed.items() if k in ExtractedFields.model_fields}
        )

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
            extracted_fields=extracted_fields,
            snippets=snippet_objs,
            chunks=chunks,
            llm_calls=[
                LLMCall(
                    prompt_hash=phash,
                    model=GPT_MODEL_NAME,
                    date=datetime.utcnow(),
                    response_raw_path=llm_response_path,
                    cached=cached,
                )
            ],
            provenance_notes=(
                "Pipeline v2: crawl + GPT extraction + optional sentence-transformers embedding."
            ),
            notes=None,
        )

        out_path = save_final_doc(doc, domain, checksum)
        print("Saved final doc:", out_path)

    except Exception as e:
        print("Error processing seed:", url, e)


# ==========================
# main
# ==========================

def main(seeds_file: str):
    with open(seeds_file, "r", encoding="utf-8") as f:
        seeds = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    for s in seeds:
        process_seed(s)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline_Version2.py seeds_example_Version2.txt")
        sys.exit(1)
    main(sys.argv[1])
