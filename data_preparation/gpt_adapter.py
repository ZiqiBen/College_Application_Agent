"""
OpenAI GPT Adapter

This module provides a simple wrapper around OpenAI's Chat Completions API.
It replaces the previous Qwen-based adapter so the main pipeline can call
an LLM through a unified function interface.

Only one function is meant to be imported by the pipeline:

    call_gpt_chat(messages, model=..., ...)

Usage:
  1. Insert your real OpenAI API key into API_KEY.
  2. Set DEFAULT_MODEL to the GPT model you intend to use for extraction.
"""

import json
import requests
from typing import List, Dict

# ============================================================
# 1. Insert your OpenAI API Key here.
#    (For real projects, environment variables are recommended.
#     For a course assignment, hardcoding is acceptable.)
# ============================================================
import os
API_KEY = os.getenv("OPENAI_API_KEY")
API_URL = "https://api.openai.com/v1/chat/completions"

# ============================================================
# 2. Default GPT model to use.
#    Replace with the exact model name available in your account:
#       "gpt-4.1"
#       "gpt-4o"
#       "gpt-4.1-mini"
#       "gpt-5.1"
# ============================================================
DEFAULT_MODEL = "gpt-4.1"


def call_gpt_chat(
    messages: List[Dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    timeout: int = 60,
) -> str:
    """
    Thin wrapper around the OpenAI Chat Completions API.

    Parameters
    ----------
    messages : List[Dict]
        A list of chat messages following OpenAI's schema:
        [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "..."},
          ...
        ]

    model : str
        The GPT model to use for inference.

    temperature : float
        Sampling temperature. For deterministic JSON extraction,
        it is set to 0.0 by default.

    max_tokens : int
        Maximum number of tokens the model is allowed to generate.

    timeout : int
        Maximum time (in seconds) to wait for the HTTP request.

    Returns
    -------
    str
        The model's assistant message content.
        The pipeline expects this to be a JSON-formatted string.

    Notes
    -----
    If parsing fails or the API returns an unexpected structure,
    the function falls back to returning the raw JSON response.
    """
    if not API_KEY or API_KEY == "YOUR_REAL_API_KEY_HERE":
        raise RuntimeError("OpenAI API key is missing. Please set API_KEY in gpt_adapter.py.")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # Expected response format:
    # {
    #   "choices": [
    #       {
    #           "message": {
    #               "role": "assistant",
    #               "content": "..."
    #           }
    #       }
    #   ]
    #   ...
    # }
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        # Fallback: return raw JSON for debugging
        return json.dumps(data, ensure_ascii=False)

