"""
Light Qwen adapter (HTTP). Replace with your platform's SDK or adjust parsing.
Expect environment variables:
  QWEN_API_URL, QWEN_API_KEY

Function:
  call_qwen_chat(messages, model="qwen-7b-chat", temperature=0.0, max_tokens=2000)
returns:
  str (raw textual response from model, expected JSON text)
"""
import os
import json
import requests
from typing import List, Dict

API_URL = os.environ.get("QWEN_API_URL", "")
API_KEY = os.environ.get("QWEN_API_KEY", "")

def call_qwen_chat(messages: List[Dict], model: str = "qwen-7b-chat", temperature: float = 0.0, max_tokens: int = 2000, timeout: int = 60) -> str:
    if not API_URL or not API_KEY:
        raise RuntimeError("Please set QWEN_API_URL and QWEN_API_KEY environment variables or replace this adapter.")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    j = resp.json()
    # NOTE: Adjust the extraction path below according to your Qwen service response structure.
    # Common pattern: {"choices":[{"message":{"content":"..."}}], ...}
    try:
        return j["choices"][0]["message"]["content"]
    except Exception:
        # Fallback: return full JSON string
        return json.dumps(j, ensure_ascii=False)