import os, re, json
from typing import List, Dict


def _chunk_text(text: str, max_len: int = 600) -> List[str]:
    """
    Split a long text into smaller chunks with a maximum character length.

    The function first splits the input text by sentence or newline boundaries,
    then greedily concatenates segments until the max_len limit is reached.
    This ensures that chunks preserve local semantic coherence while staying
    within a fixed length budget.

    Args:
        text (str): The input text string to be split.
        max_len (int): Maximum allowed length of each chunk (default: 600).

    Returns:
        List[str]: A list of non-empty text chunks.
    """
    parts = re.split(r'(?<=[\.\n])\s+', (text or "").strip())
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 <= max_len:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c]


def _boost(text: str, type_: str) -> str:
    """
    Prepend type-aware prefixes to emphasize important sections
    like mission, curriculum, and requirements.

    Args:
        text (str): The section text.
        type_ (str): The type of section, one of {"mission", "curriculum", "requirements", "other"}.

    Returns:
        str: The text prefixed with a type-aware label (e.g., "Mission: ...").
    """
    if type_ == "mission":
        prefix = "Mission: "
    elif type_ == "curriculum":
        prefix = "Core Courses: "
    elif type_ == "requirements":
        prefix = "Admission Requirements: "
    else:
        prefix = ""
    return (prefix + (text or "")).strip()


def _ingest_json(path: str) -> Dict[str, str]:
    """
    Load a structured JSON file containing program sections
    and return a mapping of chunk_id → text.

    Each section may include "text" and optional "items". The section text
    is prefixed using `_boost` and then chunked with `_chunk_text`.

    Args:
        path (str): File path to the .json file.

    Returns:
        Dict[str, str]: A mapping where keys are chunk IDs
        (e.g., "ucb-msds-01#mission#p1") and values are text chunks.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    base_id = os.path.basename(path).replace(".json", "")
    mapping = {}
    sections = data.get("sections", [])
    counter = 1
    for sec in sections:
        t = sec.get("text") or ""
        type_ = sec.get("type", "other")
        # If items exist, concatenate them into retrievable text
        items = sec.get("items") or []
        if items:
            t = t + "\n" + "; ".join(items)
        t = _boost(t, type_)
        for ch in _chunk_text(t, max_len=600):
            cid = f"{base_id}#{type_}#p{counter}"
            mapping[cid] = ch
            counter += 1
    return mapping


def _ingest_txt(path: str) -> Dict[str, str]:
    """
    Load a plain text file and return a mapping of chunk_id → text.

    The file content is split into chunks using `_chunk_text`.

    Args:
        path (str): File path to the .txt file.

    Returns:
        Dict[str, str]: A mapping where keys are chunk IDs
        (e.g., "cmu-ml-00#p2") and values are text chunks.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    base_id = os.path.basename(path).replace(".txt", "")
    mapping = {}
    for i, ch in enumerate(_chunk_text(txt, max_len=600), start=1):
        mapping[f"{base_id}#p{i}"] = ch
    return mapping


def ingest_corpus(corpus_dir: str) -> Dict[str, str]:
    """
    Read all .json and .txt files under a corpus directory and
    return a combined mapping of chunk_id → text.

    JSON files are parsed with `_ingest_json` (type-aware prefixes included),
    TXT files are parsed with `_ingest_txt`.

    Args:
        corpus_dir (str): Path to the directory containing corpus files.

    Returns:
        Dict[str, str]: A mapping of chunk IDs to text chunks, e.g.:
            {
              "ucb-msds-01#mission#p1": "Mission: ...",
              "cmu-ml-00#p2": "Tracks include ..."
            }
    """
    mapping: Dict[str, str] = {}
    for fname in sorted(os.listdir(corpus_dir)):
        fpath = os.path.join(corpus_dir, fname)
        if fname.endswith(".json"):
            mapping.update(_ingest_json(fpath))
        elif fname.endswith(".txt"):
            mapping.update(_ingest_txt(fpath))
        else:
            continue
    return mapping
