"""
Ultimate Course Extraction Engine (v3 + v2 merged)
--------------------------------------------------
Upgrades:
✔ original v2 functions retained (find_candidate_course_subpages)
✔ v3 functions added (course detail discovery)
✔ course name extraction from plain text
✔ heuristic scanning for detail pages
✔ HTML/PDF extraction
✔ (embedding helper kept but not required by pipeline)
"""

import os
import re
import time
import json
import hashlib
from urllib.parse import urlparse, urljoin, urldefrag
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Tuple, List, Dict
import requests
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# ---------------------------------
# Global Settings
# ---------------------------------

USER_AGENT = "UniCorpusBot/0.1 (+zk2160@nyu.edu)"

OUT_CACHE = "out/cache"
LLM_CACHE_DIR = "out/llm_cache"

os.makedirs(OUT_CACHE, exist_ok=True)
os.makedirs(LLM_CACHE_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/pdf"
})

# ---------------------------------
# Helpers
# ---------------------------------

def canonicalize_url(url: str) -> str:
    """Remove fragment part (#...)."""
    u, _ = urldefrag(url)
    return u


def sha256_bytes(b: bytes) -> str:
    """SHA-256 hash of raw bytes."""
    return hashlib.sha256(b).hexdigest()


def save_snapshot(url: str, content: bytes, content_type: str) -> Tuple[str, str]:
    """
    Save raw response body to OUT_CACHE with timestamp + checksum.

    Returns:
        snapshot_path, checksum_hex
    """
    h = sha256_bytes(content)
    date = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    if "html" in content_type:
        ext = "html"
    elif "pdf" in content_type:
        ext = "pdf"
    else:
        ext = "bin"

    filename = f"{date}_{h[:12]}.{ext}"
    path = os.path.join(OUT_CACHE, filename)

    with open(path, "wb") as f:
        f.write(content)

    return path, h


def crawl_url(url: str, timeout: int = 30, rate_delay: float = 0.3) -> Tuple[str, bytes, str]:
    """
    Simple synchronous crawl + snapshot.

    rate_delay:
        small sleep between requests to avoid hammering sites.
    """
    time.sleep(rate_delay)
    resp = session.get(url, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()

    content = resp.content
    ct = resp.headers.get("Content-Type", "").lower()
    snapshot_path, checksum = save_snapshot(url, content, ct)

    return snapshot_path, content, ct

# ---------------------------------
# Text Extraction
# ---------------------------------

def extract_text_from_html_bytes(html_bytes: bytes) -> Tuple[str, str]:
    """
    Parse HTML, strip scripts/styles/nav/footer, and return (title, main_text).
    """
    soup = BeautifulSoup(html_bytes, "lxml")

    # remove noise tags
    for s in soup(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
        s.decompose()

    title = soup.title.string.strip() if (soup.title and soup.title.string) else ""

    main = soup.find("main") or soup.find("article") or soup.body
    if main:
        text = main.get_text("\n", strip=True)
    else:
        text = soup.get_text("\n", strip=True)

    # collapse excessive blank lines
    text = re.sub(r"\n{2,}", "\n\n", text)

    return title, text


def extract_text_from_pdf_path(pdf_path: str) -> str:
    """
    Extract all text from a PDF using PyMuPDF.
    """
    doc = fitz.open(pdf_path)
    pages = [p.get_text("text") for p in doc]
    return "\n".join(pages)

# ---------------------------------
# Prefilter
# ---------------------------------

def simple_prefilter(text: str) -> bool:
    """
    Quick heuristic: does this look like a program / curriculum / admission page?
    """
    kws = [
        "program", "course", "courses", "curriculum",
        "admission", "admissions", "degree", "master",
        "ms program", "m.s.", "m.sc.",
        "学位", "课程", "项目", "招生", "申请", "要求", "tuition", "学费"
    ]
    t = text.lower()
    return any(k in t for k in kws)

# ---------------------------------
# v2: Find course/curriculum subpages
# ---------------------------------

def find_candidate_course_subpages(
    html_bytes: bytes,
    base_url: str,
    max_links: int = 8
) -> List[str]:
    """
    From curriculum/main page, heuristically find links to course-related subpages.

    We only keep:
      - same-domain links
      - links whose text or href contains curriculum/course-related keywords
      - up to max_links links (to control crawl load)
    """
    soup = BeautifulSoup(html_bytes, "lxml")
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc

    keywords = [
        "curriculum",
        "course",
        "courses",
        "curricula",
        "program requirements",
        "degree requirements",
        "plan of study",
        "program of study",
        "study plan",
        "sample schedule",
        "sample curriculum",
        "course list",
        "course listing",
        "course directory",
        "coursecatalog",  # for sites that have /coursecatalog/ paths
    ]

    def looks_course_like(text: str, href: str) -> bool:
        t = (text or "").lower()
        h = (href or "").lower()
        return any(kw in t or kw in h for kw in keywords)

    found: List[str] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href_raw = a["href"].strip()
        if not href_raw or href_raw.startswith("#"):
            continue
        if href_raw.startswith("mailto:") or href_raw.startswith("tel:"):
            continue

        text = a.get_text(" ", strip=True)
        if not looks_course_like(text, href_raw):
            continue

        full_url = urljoin(base_url, href_raw)
        full_url = canonicalize_url(full_url)

        parsed = urlparse(full_url)
        if parsed.netloc != base_domain:
            continue

        if full_url in seen:
            continue

        seen.add(full_url)
        found.append(full_url)

        if len(found) >= max_links:
            break

    return found

# ---------------------------------
# v3: Course name extraction from plain text
# ---------------------------------

COURSE_NAME_PATTERN = re.compile(
    r"^[A-Z][a-zA-Z0-9&\-\.\s]{2,50}$",
    re.MULTILINE,
)

def extract_course_names_from_text(text: str) -> List[str]:
    """
    Extract likely course names from plain text.
    Examples: "Finance 1", "Marketing", "Data Science & AI for Leaders"
    """
    matches = COURSE_NAME_PATTERN.findall(text)
    seen: set[str] = set()
    results: List[str] = []
    for m in matches:
        m = m.strip()
        # avoid very long lines (probably sentences)
        if len(m.split()) > 6:
            continue
        if m not in seen:
            seen.add(m)
            results.append(m)
    return results

# ---------------------------------
# v3: Generate candidate detail URLs
# ---------------------------------

def generate_possible_detail_urls(base_domain: str, course_name: str) -> List[str]:
    """
    Given a course name like "Finance 1", generate possible detail URLs:
      /coursecatalog/finance-1
      /courses/finance-1
      /course/finance-1
    etc.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", course_name.lower()).strip("-")

    templates = [
        f"/coursecatalog/{slug}",
        f"/coursecatalog/{slug}.html",
        f"/coursecatalog/{slug}.aspx",
        f"/courses/{slug}",
        f"/courses/{slug}.html",
        f"/courses/{slug}.aspx",
        f"/course/{slug}",
        f"/course/{slug}.html",
        f"/course/{slug}.aspx",
    ]

    return [f"https://{base_domain}{t}" for t in templates]

# ---------------------------------
# v3: Try fetching the detail page
# ---------------------------------

def try_fetch_detail_page(url: str) -> str | None:
    """
    Try to fetch candidate course detail page.
    Returns:
        text if valid HTML and length is large enough,
        otherwise None.
    """
    try:
        snap, content, ct = crawl_url(url)
        if "html" not in ct:
            return None
        _, text = extract_text_from_html_bytes(content)
        if len(text) < 100:
            return None
        return text
    except Exception:
        return None

# ---------------------------------
# v3: Discover detail pages for each course name
# ---------------------------------

def discover_course_detail_pages(
    text: str,
    base_domain: str,
    max_courses: int = 8,
    max_candidates_per_course: int = 2,
) -> Dict[str, str]:
    """
    High-frequency optimized version:

    - Only consider at most `max_courses` course names.
    - For each course, only try the first `max_candidates_per_course` candidate URLs.

    This keeps total network calls bounded: max_courses * max_candidates_per_course.
    """
    course_names = extract_course_names_from_text(text)
    course_names = course_names[:max_courses]

    course_to_desc: Dict[str, str] = {}

    for cname in course_names:
        candidates = generate_possible_detail_urls(base_domain, cname)
        for u in candidates[:max_candidates_per_course]:
            desc = try_fetch_detail_page(u)
            if desc:
                course_to_desc[cname] = desc
                break

    return course_to_desc

# ---------------------------------
# Chunking (unchanged)
# ---------------------------------

def chunk_text_by_words(text: str, max_words: int = 400, overlap: int = 100) -> List[dict]:
    """
    Split text into overlapping word-based chunks.
    """
    words = text.split()
    chunks: List[dict] = []
    i = 0
    n = len(words)
    chunk_idx = 0

    joined = " ".join(words)
    char_pos = 0
    word_char_positions: List[int] = []

    for w in words:
        idx = joined.find(w, char_pos)
        word_char_positions.append(idx)
        char_pos = idx + len(w)

    while i < n:
        j = min(i + max_words, n)
        chunk_words = words[i:j]
        chunk_text = " ".join(chunk_words)

        start_char = word_char_positions[i] if i < len(word_char_positions) else 0
        end_char = (
            word_char_positions[j - 1] + len(words[j - 1])
            if (j - 1) < len(words)
            else len(joined)
        )

        chunks.append({
            "chunk_id": f"chunk_{chunk_idx:04d}",
            "text": chunk_text,
            "word_count": len(chunk_words),
            "char_start": start_char,
            "char_end": end_char,
            "token_count": len(chunk_words),
        })

        i = j - overlap if j < n else j
        chunk_idx += 1

    return chunks

# ---------------------------------
# Embedding helper (kept but optional)
# ---------------------------------

_MODEL = None

def get_embedding_for_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2"):
    """
    Batch-encode a list of texts into sentence embeddings using sentence-transformers.

    NOTE: current pipeline_v2 has embeddings DISABLED for speed.
    This helper is kept in case you want to re-enable later.
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(model_name)
    return _MODEL.encode(texts, show_progress_bar=False, convert_to_numpy=True)

# ---------------------------------
# LLM Cache
# ---------------------------------

def save_llm_cache(prompt_hash: str, checksum: str, response_text: str) -> str:
    """
    Save raw LLM response text to a cache file keyed by (prompt_hash, page_checksum).
    """
    key = hashlib.sha256((prompt_hash + checksum).encode("utf-8")).hexdigest()[:16]
    path = os.path.join(LLM_CACHE_DIR, f"{key}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(response_text)
    return path


def load_llm_cache(prompt_hash: str, checksum: str) -> str | None:
    """
    Try to load cached LLM response for (prompt_hash, page_checksum).
    """
    key = hashlib.sha256((prompt_hash + checksum).encode("utf-8")).hexdigest()[:16]
    path = os.path.join(LLM_CACHE_DIR, f"{key}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None
