"""
LLM utilities for Writing Agent
"""

from typing import Any, Literal, Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from ..config import WritingAgentConfig


def get_llm(
    provider: Literal["openai", "anthropic", "qwen"] = "openai",
    model_name: Optional[str] = None,
    temperature: float = 0.7
) -> BaseChatModel:
    """
    Get LLM instance based on provider.
    
    Args:
        provider: LLM provider name
        model_name: Specific model name (optional)
        temperature: Temperature for generation
    
    Returns:
        LangChain chat model instance
    """
    
    if not model_name:
        model_name = WritingAgentConfig.get_model_name(provider)
    
    if provider == "openai":
        api_key = WritingAgentConfig.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )
    
    elif provider == "anthropic":
        api_key = WritingAgentConfig.ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )
    
    elif provider == "qwen":
        # For Qwen, we'll use OpenAI-compatible interface if available
        # Or implement custom wrapper
        api_key = WritingAgentConfig.QWEN_API_KEY
        api_url = WritingAgentConfig.QWEN_API_URL
        
        if not api_key or not api_url:
            raise ValueError("QWEN_API_KEY and QWEN_API_URL must be set in environment variables")
        
        # Use OpenAI wrapper with custom base URL
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=api_url
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def call_llm(llm: BaseChatModel, prompt: str, system_message: Optional[str] = None) -> str:
    """
    Call LLM with prompt and optional system message.
    
    Args:
        llm: LangChain chat model instance
        prompt: User prompt
        system_message: Optional system message
    
    Returns:
        Generated text response
    """
    
    messages = []
    
    if system_message:
        messages.append(SystemMessage(content=system_message))
    
    messages.append(HumanMessage(content=prompt))
    
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        # Log error and return error message
        error_msg = f"LLM call failed: {str(e)}"
        print(f"ERROR: {error_msg}")
        return f"[Error: {error_msg}]"


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text (rough approximation).
    
    Args:
        text: Input text
    
    Returns:
        Estimated token count
    """
    # Rough estimate: ~4 characters per token
    return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximately max_tokens.
    
    Args:
        text: Input text
        max_tokens: Maximum tokens to keep
    
    Returns:
        Truncated text
    """
    max_chars = max_tokens * 4  # Rough approximation
    
    if len(text) <= max_chars:
        return text
    
    return text[:max_chars] + "..."
