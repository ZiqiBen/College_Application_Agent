"""
OpenAI GPT Adapter (for pipeline_Version2)

强制使用 gpt-4o（忽略 OPENAI_MODEL 环境变量）
保持与旧接口 /v1/chat/completions 完全兼容
支持 max_tokens，不会出现 gpt-5.1 的 "unsupported parameter" 错误

本文件是整个数据准备流水线（pipeline_Version2）的 LLM 调用适配器。
pipeline 内部所有 GPT 调用都会经过 call_gpt_chat()。
"""

import os
import json
import requests
from typing import List, Dict


# ==========================
# 1. API 基本配置
# ==========================

# 从环境变量读取 OpenAI API Key（必须设置）
API_KEY = os.environ.get("OPENAI_API_KEY")

# OpenAI Chat Completions 接口（旧式 API，与你 pipeline 匹配）
API_URL = "https://api.openai.com/v1/chat/completions"

# 强制写死默认模型为 gpt-4o
# 这样无论本地环境变量 OPENAI_MODEL 设成什么，都不会误用 gpt-5.1
DEFAULT_MODEL = "gpt-4o"


def call_gpt_chat(
    messages: List[Dict],
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 2000,    # gpt-4o 完全支持，不会报错
    timeout: int = 60,
) -> str:
    """
    这是对 OpenAI Chat Completions 的一个薄封装（wrapper）。
    pipeline 会传入 messages（对话历史），本函数会返回 assistant 的文本内容。

    参数说明：
    ----------
    messages : List[Dict]
        形如：
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]
        这是 OpenAI Chat API 的标准输入格式。

    model : str
        pipeline 会传入 model="gpt-4o"，但为了 pipeline 稳定，我们忽略传入参数，
        强制使用 DEFAULT_MODEL（始终为 "gpt-4o"）。

    temperature : float
        控制随机性。抽取结构化信息任务通常设置为 0.0。

    max_tokens : int
        控制最大生成长度。gpt-4o 支持该参数，因此安全。

    timeout : int
        控制 HTTP 请求超时时间。

    返回：
    -------
    str
        GPT 返回的 assistant message 的 content。
        若失败，返回 None。
    """

    # -------- 安全检查：API_KEY 是否存在 --------
    if not API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please set it in your shell environment."
        )

    # -------- 模型名确定（强制使用 gpt-4o）--------
    model_name = DEFAULT_MODEL

    # -------- 构造请求主体 payload --------
    # 注意：这是 old-style Chat Completions 接口格式
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,   # gpt-4o 支持，无需改为 max_completion_tokens
    }

    # -------- HTTP Headers（含 API Key）--------
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # -------- 发送 HTTP POST 请求 --------
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
    except Exception as e:
        # 网络异常（如断网、DNS 错误、连接超时）
        print(f"[GPT Adapter] HTTP request failed: {e}")
        return None

    # -------- 如果状态码 >= 400，打印详细报错 --------
    if resp.status_code >= 400:
        try:
            err_json = resp.json()
        except Exception:
            err_json = {"raw": resp.text}

        print(
            f"[GPT Adapter] OpenAI API error {resp.status_code} for model '{model_name}':\n"
            f"{json.dumps(err_json, ensure_ascii=False, indent=2)}"
        )
        return None

    # -------- 正常返回：解析 JSON --------
    try:
        data = resp.json()
    except Exception:
        print("[GPT Adapter] Failed to parse JSON response from OpenAI.")
        print("Raw response text:", resp.text[:500])
        return None

    # -------- 从标准结构提取 content --------
    # 正常结构：
    # {
    #   "choices": [
    #       {
    #           "message": { "role": "assistant", "content": "..." }
    #       }
    #   ]
    # }
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        print("[GPT Adapter] Unexpected response structure:")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return None
