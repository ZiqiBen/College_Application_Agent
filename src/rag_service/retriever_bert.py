from typing import List, Tuple, Dict
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

# Pre-trained Sentence-BERT model for encoding documents and queries
_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def build_query(program_text: str, profile_keywords: List[str]) -> str:
    """
    Construct a query string by concatenating program text and applicant profile keywords.

    Args:
        program_text (str): The main program description text (e.g., mission, curriculum).
        profile_keywords (List[str]): A list of applicant-related keywords such as 
                                      skills, courses, or experiences.

    Returns:
        str: A single concatenated query string combining program text and keywords.
    """
    extra = " ".join(profile_keywords or [])
    return f"{program_text or ''} {extra}".strip()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.

    Args:
        a (np.ndarray): First embedding vector.
        b (np.ndarray): Second embedding vector.

    Returns:
        float: Cosine similarity value in the range [-1, 1]. 
               Returns 0.0 if either vector has zero norm.
    """
    denom = (norm(a) * norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieve_topk(corpus: Dict[str, str], query: str, k: int = 5) -> List[Tuple[str, float]]:
    """
    Retrieve the top-k most relevant document chunks from a corpus using Sentence-BERT embeddings.

    Args:
        corpus (Dict[str, str]): A mapping from chunk_id to text chunk (pre-processed program materials).
        query (str): The user query string, typically program text + applicant profile keywords.
        k (int, optional): Number of top-ranked results to return. Defaults to 5.

    Returns:
        List[Tuple[str, float]]: A ranked list of (chunk_id, similarity_score) pairs, 
                                 sorted by descending similarity to the query.
    """
    if not corpus:
        return []
    ids = list(corpus.keys())
    docs = [corpus[i] for i in ids]

    # Encode all document chunks and the query
    doc_emb = _MODEL.encode(docs, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    q_emb = _MODEL.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]

    # Compute cosine similarities (dot product since embeddings are normalized)
    sims = (doc_emb @ q_emb).astype(float) 
    order = np.argsort(sims)[::-1]  # sort in descending order

    return [(ids[i], float(sims[i])) for i in order[:k]]

from typing import List, Tuple, Dict
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def build_query(program_text: str, profile_keywords: List[str]) -> str:
    extra = " ".join(profile_keywords or [])
    return f"{program_text or ''} {extra}".strip()

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (norm(a) * norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)

def retrieve_topk(corpus: Dict[str, str], query: str, k: int = 5) -> List[Tuple[str, float]]:
    if not corpus:
        return []
    ids = list(corpus.keys())
    docs = [corpus[i] for i in ids]

    doc_emb = _MODEL.encode(docs, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    q_emb = _MODEL.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]

    sims = (doc_emb @ q_emb).astype(float)  
    order = np.argsort(sims)[::-1]
    return [(ids[i], float(sims[i])) for i in order[:k]]