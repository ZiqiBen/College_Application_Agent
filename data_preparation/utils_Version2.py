"""
Utilities: crawling, snapshot save, html/pdf extract, prefilter, chunking, embedding helper.
"""
import os
import re
import time
import json
import hashlib
from urllib.parse import urlparse, urljoin, urldefrag
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Tuple, List
import requests
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

USER_AGENT = "UniCorpusBot/0.1 (+your-email@example.com)"
OUT_CACHE = "out/cache"
LLM_CACHE_DIR = "out/llm_cache"
os.makedirs(OUT_CACHE, exist_ok=True)
os.makedirs(LLM_CACHE_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/pdf"})

def canonicalize_url(url: str) -> str:
    u, _ = urldefrag(url)
    return u

def sha256_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()

def save_snapshot(url: str, content: bytes, content_type: str) -> Tuple[str, str]:
    h = sha256_bytes(content)
    date = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    ext = "html" if "html" in content_type else "pdf" if "pdf" in content_type else "bin"
    filename = f"{date}_{h[:12]}.{ext}"
    path = os.path.join(OUT_CACHE, filename)
    with open(path, "wb") as f:
        f.write(content)
    return path, h

def crawl_url(url: str, timeout: int = 30, rate_delay: float = 1.0) -> Tuple[str, bytes, str]:
    """
    Simple synchronous crawl + snapshot. Production: add robots.txt checks and retries.
    Returns (snapshot_path, content_bytes, content_type)
    """
    time.sleep(rate_delay)
    resp = session.get(url, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    content = resp.content
    ct = resp.headers.get("Content-Type", "").lower()
    snapshot_path, checksum = save_snapshot(url, content, ct)
    return snapshot_path, content, ct

def extract_text_from_html_bytes(html_bytes: bytes) -> Tuple[str, str]:
    soup = BeautifulSoup(html_bytes, "lxml")
    for s in soup(["script", "style", "noscript", "header", "footer", "svg", "nav"]):
        s.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    main = soup.find("main") or soup.find("article")
    container = main if main else soup.body
    text = container.get_text(separator="\n", strip=True) if container else soup.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return title, text

def extract_text_from_pdf_path(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    return "\n".join(pages)

def simple_prefilter(text: str) -> bool:
    kws = ["program", "course", "admission", "admissions", "degree", "学位", "课程", "项目", "招生", "申请", "要求", "tuition", "学费"]
    s = text.lower()
    return any(k in s for k in kws)

def chunk_text_by_words(text: str, max_words: int = 400, overlap: int = 100) -> List[dict]:
    words = text.split()
    chunks = []
    i = 0
    n = len(words)
    chunk_idx = 0
    # Precompute word char positions for offsets
    joined = " ".join(words)
    char_pos = 0
    word_char_positions = []
    for w in words:
        idx = joined.find(w, char_pos)
        word_char_positions.append(idx)
        char_pos = idx + len(w)
    while i < n:
        j = min(i + max_words, n)
        chunk_words = words[i:j]
        chunk_text = " ".join(chunk_words)
        start_char = word_char_positions[i] if i < len(word_char_positions) else 0
        end_char = (word_char_positions[j-1] + len(words[j-1])) if j-1 < len(words) else len(joined)
        chunks.append({
            "chunk_id": f"chunk_{chunk_idx:04d}",
            "text": chunk_text,
            "word_count": len(chunk_words),
            "char_start": start_char,
            "char_end": end_char,
            "token_count": len(chunk_words)
        })
        i = j - overlap if (j < n) else j
        chunk_idx += 1
    return chunks

# embedding helper (sentence-transformers)
_MODEL = None
def get_embedding_for_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2"):
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(model_name)
    return _MODEL.encode(texts, show_progress_bar=False, convert_to_numpy=True)

def save_llm_cache(prompt_hash: str, checksum: str, response_text: str) -> str:
    key = hashlib.sha256((prompt_hash + checksum).encode("utf-8")).hexdigest()[:16]
    path = os.path.join(LLM_CACHE_DIR, f"{key}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(response_text)
    return path

def load_llm_cache(prompt_hash: str, checksum: str) -> str:
    key = hashlib.sha256((prompt_hash + checksum).encode("utf-8")).hexdigest()[:16]
    path = os.path.join(LLM_CACHE_DIR, f"{key}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None