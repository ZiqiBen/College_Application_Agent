#!/usr/bin/env python3
"""
Main pipeline (v2 + v3 integrated)
----------------------------------
Features:
- Crawl main page
- Discover curriculum/course subpages
- Discover course-detail pages (bounded: <= 8 courses × 2 URLs)
- Merge all text
- Feed into GPT (gpt-4o) for structured extraction
- Save final CorpusDoc JSON
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
from gpt_adapter import call_gpt_chat


# --------------------------------------------
# Prompt Template
# --------------------------------------------

SYSTEM_PROMPT = (
    "你是严格的结构化信息抽取器（information extraction system）。"
    "你的任务是从给定的英文网页内容（包括主页面、课程页面、课程详情页面）中，"
    "抽取研究生项目的结构化信息，并以 JSON 格式输出。"
    "要求：\n"
    "1. 严格按照给定的 schema 输出合法 JSON（不能多字段、不能少字段）。\n"
    "2. 尽量从文本中抽取真实字段，不得编造网页中不存在的信息。\n"
    "3. courses 字段：\n"
    "   - 课程名必须来自页面真实文本。\n"
    "   - 若未找到描述，description 用 null。\n"
    "4. snippets 若可能请提供。\n"
    "5. 未找到信息即为 null。\n"
    "6. 禁止使用外部知识。\n"
    "7. 输出中禁止任何解释性文字，只能返回 JSON。\n"
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
  "school": "string|null",
  "source_url": "string"
}
"""


def prompt_for_text(page_text: str, source_url: str) -> list:
    user = (
        f"{USER_SCHEMA_DESC}\n\n"
        f"source_url: {source_url}\n\n"
        "下面是该项目的主页面、课程页面、课程详情页等合并文本，请按 schema 抽取结构化信息。\n\n"
        "<<<文档开始>>>\n"
        f"{page_text}\n"
        "<<<文档结束>>>"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256((system + "\n" + user).encode("utf-8")).hexdigest()


def try_parse_json(text: str):
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


def validate_and_enrich(extracted: dict, combined_text: str) -> dict:
    """
    对 LLM 抽取结果做一些基础校验与轻微规范化。

    combined_text:
        我们传给 LLM 的整体文本（主 + 子页面 + 详情），用于
        - snippet 校验
        - 从文本反推出更合理的 school 名称
    """
    notes = extracted.get("notes") or ""

    # --- email validation -----------------------------------------------------
    email = extracted.get("contact_email")
    if email:
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            extracted["contact_email"] = None
            notes += "contact_email_invalid;"

    # --- duration normalization ----------------------------------------------
    if extracted.get("duration"):
        m = re.search(r"(\d+\s*(year|years|month|months))", extracted["duration"], re.I)
        if m:
            extracted.setdefault("duration_normalized", m.group(1))

    # --- snippet sanity check -------------------------------------------------
    snippets = extracted.get("snippets") or {}
    for k, v in snippets.items():
        if v and isinstance(v, dict) and "text" in v:
            if v["text"] not in combined_text:
                notes += f"snippet_mismatch_{k};"

    # --- normalize courses ----------------------------------------------------
    courses = extracted.get("courses")
    if courses and isinstance(courses, list):
        # ["A", "B", ...] -> [{"name": "A", "description": null}, ...]
        if len(courses) > 0 and isinstance(courses[0], str):
            extracted["courses"] = [{"name": c, "description": None} for c in courses]

    # --- infer / correct school from page text -------------------------------
    school = extracted.get("school")
    school_lower = (school or "").lower()

    GENERIC_SCHOOL_STRINGS = [
        "graduate school",
        "school",
        "division",
        "department",
        "faculty",
    ]

    def looks_generic(s: str) -> bool:
        s = s.strip().lower()
        if not s:
            return True
        return any(g in s for g in GENERIC_SCHOOL_STRINGS)

    def guess_institution_from_text(text: str) -> str | None:
        """
        从文本中找形如 "XXX University/College/Institute/School" 的实体。
        """
        pattern = re.compile(
            r"([A-Z][A-Za-z&,\- ]{1,80}\s("
            r"University|College|Institute|School"
            r"))"
        )
        matches = list(pattern.finditer(text))
        if not matches:
            return None

        def score(m):
            val = m.group(0)
            if "University" in val:
                base = 3
            elif "College" in val or "Institute" in val:
                base = 2
            else:
                base = 1
            length_penalty = len(val) / 100.0
            return base - length_penalty

        best = max(matches, key=score)
        return best.group(1).strip()

    needs_override = (school is None) or looks_generic(school_lower)
    if needs_override:
        inferred_school = guess_institution_from_text(combined_text)
        if inferred_school:
            prev = school
            extracted["school"] = inferred_school
            if prev:
                notes += f"school_overridden_from_{prev}_to_{inferred_school};"
            else:
                notes += f"school_inferred_as_{inferred_school};"

    extracted["notes"] = notes if notes else None

    if extracted.get("confidence") is None:
        extracted["confidence"] = 0.5

    return extracted


def domain_slug(url: str) -> str:
    return urlparse(url).netloc.replace(":", "_")


def save_final_doc(doc: CorpusDoc, domain: str, checksum: str) -> str:
    base_dir = os.path.join("dataset", "graduate_programs")
    dirpath = os.path.join(base_dir, domain)
    os.makedirs(dirpath, exist_ok=True)

    filename = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{checksum[:12]}.json"
    path = os.path.join(dirpath, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    return path


# --------------------------------------------
# Core Logic
# --------------------------------------------

def process_seed(url: str):
    try:
        print("Crawling main page:", url)
        snapshot_path, content, content_type = utils.crawl_url(url)
        domain = domain_slug(url)
        checksum = utils.sha256_bytes(content)

        title = ""
        main_text = ""

        # Step 1: main page text extraction
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            main_text = utils.extract_text_from_pdf_path(snapshot_path)
        else:
            title, main_text = utils.extract_text_from_html_bytes(content)

        # Step 2: discover course/curriculum subpages
        subpage_texts: list[str] = []
        if "html" in content_type:
            try:
                candidate_subpages = utils.find_candidate_course_subpages(
                    html_bytes=content,
                    base_url=url,
                    max_links=5,   # 控制在 5 个以内，防止太慢
                )
            except Exception as e:
                print("  Error while finding candidate subpages:", e)
                candidate_subpages = []

            if candidate_subpages:
                print("  Found candidate course/curriculum subpages:")
                for su in candidate_subpages:
                    print("   -", su)

            for su in candidate_subpages:
                try:
                    sub_snap, sub_content, sub_ct = utils.crawl_url(su)
                    if "pdf" in sub_ct or su.lower().endswith(".pdf"):
                        sub_text = utils.extract_text_from_pdf_path(sub_snap)
                    else:
                        _, sub_text = utils.extract_text_from_html_bytes(sub_content)
                    if sub_text and sub_text.strip():
                        subpage_texts.append(sub_text)
                except Exception as e:
                    print("  Error fetching subpage:", su, e)

        # Step 3: merge main + subpages
        combined_text_for_llm = main_text or ""
        for txt in subpage_texts:
            combined_text_for_llm += "\n\n" + txt

        # v3: discover course detail pages (bounded)
        parsed_base = urlparse(url)
        base_domain = parsed_base.netloc

        detail_descs = utils.discover_course_detail_pages(
            combined_text_for_llm,
            base_domain,
            max_courses=8,
            max_candidates_per_course=2,
        )

        for cname, desc in detail_descs.items():
            combined_text_for_llm += (
                f"\n\n==== COURSE DETAIL START ({cname}) ====\n"
                f"{desc}\n"
                f"==== COURSE DETAIL END ({cname}) ====\n"
            )

        if not combined_text_for_llm.strip():
            print("  Empty extracted text; skipping LLM.")
            doc = CorpusDoc(
                id=f"{domain}_{checksum[:12]}",
                source_url=url,
                raw_snapshot_path=snapshot_path,
                content_type=content_type,
                crawl_date=datetime.utcnow(),
                checksum=checksum,
                language=None,
                title=title,
                raw_text=main_text,
                extracted_fields=ExtractedFields(),
                snippets={},
                chunks=[],
                llm_calls=[],
                provenance_notes="Empty text; no LLM call.",
                notes=None,
            )
            path = save_final_doc(doc, domain, checksum)
            print("Saved minimal doc:", path)
            return

        # Step 4: prefilter
        if not utils.simple_prefilter(combined_text_for_llm):
            print("Prefilter: unlikely target page, but still using GPT for PoC.")

        # Step 5: chunking
        chunks_raw = utils.chunk_text_by_words(
            combined_text_for_llm,
            max_words=400,
            overlap=100,
        )

        # Step 6: build prompt text (cap length)
        combined_for_prompt = ""
        for c in chunks_raw:
            combined_for_prompt += c["text"] + "\n\n"
            if len(combined_for_prompt) > 30000:
                break

        combined_for_prompt = combined_for_prompt[:30000]

        messages = prompt_for_text(combined_for_prompt, url)
        phash = prompt_hash(SYSTEM_PROMPT, messages[1]["content"])

        # Step 7: LLM cache + call
        cached = False
        cached_resp = utils.load_llm_cache(phash, checksum)

        if cached_resp:
            print("Using cached LLM response.")
            resp_text = cached_resp
            cached = True
        else:
            resp_text = call_gpt_chat(
                messages,
                model="gpt-4o",
                temperature=0.0,
                max_tokens=2000,
            )
            if resp_text:
                utils.save_llm_cache(phash, checksum, resp_text)

        parsed = try_parse_json(resp_text) if resp_text else None

        if parsed is None:
            parsed = {
                "program_name": None,
                "duration": None,
                "courses": None,
                "tuition": None,
                "application_requirements": None,
                "features": None,
                "contact_email": None,
                "language": None,
                "school": None,
                "source_url": url,
                "snippets": {},
                "confidence": 0.0,
                "notes": "llm_failed_or_no_json",
            }

        parsed = validate_and_enrich(parsed, combined_text_for_llm)

        # Step 8: chunks only (embeddings disabled for speed)
        chunks: list[Chunk] = []
        for i, c in enumerate(chunks_raw):
            chunks.append(
                Chunk(
                    chunk_id=f"{domain}_{checksum[:8]}_{i:04d}",
                    text=c["text"],
                    char_start=c["char_start"],
                    char_end=c["char_end"],
                    token_count=c["token_count"],
                    vector_id=None,  # 不生成 embedding
                )
            )

        # only keep keys that belong to ExtractedFields (pydantic v2: model_fields)
        extracted_fields = ExtractedFields(
            **{k: v for k, v in parsed.items() if k in ExtractedFields.model_fields}
        )

        snippet_objs = {
            k: Snippet(**v)
            for k, v in (parsed.get("snippets") or {}).items()
            if isinstance(v, dict)
        }

        llm_calls = [
            LLMCall(
                prompt_hash=phash,
                model="gpt-4o",
                date=datetime.utcnow(),
                response_raw_path=None,
                cached=cached,
            )
        ]

        doc = CorpusDoc(
            id=f"{domain}_{checksum[:12]}",
            source_url=url,
            raw_snapshot_path=snapshot_path,
            content_type=content_type,
            crawl_date=datetime.utcnow(),
            checksum=checksum,
            language=parsed.get("language"),
            title=title,
            raw_text=main_text,
            extracted_fields=extracted_fields,
            snippets=snippet_objs,
            chunks=chunks,
            llm_calls=llm_calls,
            provenance_notes="Pipeline v3: main + subpages + bounded course-detail discovery.",
            notes=None,
        )

        out_path = save_final_doc(doc, domain, checksum)
        print("Saved final doc:", out_path)

    except Exception as e:
        print("Error processing seed:", url, e)


# --------------------------------------------
# CLI Entry
# --------------------------------------------

def main(seeds_file: str):
    with open(seeds_file, "r", encoding="utf-8") as f:
        seeds = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    for url in seeds:
        process_seed(url)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline_Version2.py seeds_example_Version2.txt")
        sys.exit(1)

    main(sys.argv[1])
