"""
Configuration for Writing Agent
"""

import os
from typing import Literal, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class WritingAgentConfig:
    """Configuration class for Writing Agent"""
    
    # ===== LLM Configuration =====
    LLM_PROVIDER: Literal["openai", "anthropic", "qwen"] = os.getenv(
        "WRITING_AGENT_LLM_PROVIDER", "openai"
    )
    
    # Model names for different providers
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
    QWEN_MODEL: str = os.getenv("QWEN_MODEL", "qwen-turbo")
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    QWEN_API_KEY: Optional[str] = os.getenv("QWEN_API_KEY")
    QWEN_API_URL: Optional[str] = os.getenv("QWEN_API_URL")
    
    # Temperature settings
    TEMPERATURE: float = float(os.getenv("WRITING_AGENT_TEMPERATURE", "0.7"))
    TEMPERATURE_REFLECTION: float = 0.3  # Lower temperature for reflection
    
    # ===== RAG Configuration =====
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # ===== Iteration Configuration =====
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "3"))
    QUALITY_THRESHOLD: float = float(os.getenv("QUALITY_THRESHOLD", "0.85"))
    MIN_QUALITY_IMPROVEMENT: float = 0.05  # Minimum improvement per iteration
    
    # ===== Tool Configuration =====
    ENABLE_TOOLS: list = [
        "match_calculator",
        "keyword_extractor",
        "experience_finder",
        "requirement_checker"
    ]
    
    # ===== Reflection Dimensions =====
    REFLECTION_DIMENSIONS: list = [
        "keyword_coverage",      # Are required keywords naturally integrated?
        "personalization",       # Is content personalized with specific examples?
        "coherence",            # Is the narrative logical and well-structured?
        "program_alignment",    # Does it align with program requirements/values?
        "persuasiveness"        # Is it compelling and convincing?
    ]
    
    # Weights for reflection dimensions (must sum to 1.0)
    REFLECTION_WEIGHTS: dict = {
        "keyword_coverage": 0.20,
        "personalization": 0.25,
        "coherence": 0.20,
        "program_alignment": 0.20,
        "persuasiveness": 0.15
    }
    
    # ===== Document Type Specific Settings =====
    PS_MIN_LENGTH: int = 500  # Personal Statement minimum words
    PS_MAX_LENGTH: int = 800
    
    RESUME_MIN_BULLETS: int = 3
    RESUME_MAX_BULLETS: int = 8
    
    RL_MIN_LENGTH: int = 400  # Recommendation Letter minimum words
    RL_MAX_LENGTH: int = 700
    
    # ===== Memory & Caching =====
    ENABLE_MEMORY: bool = True
    CACHE_DIR: str = os.getenv("CACHE_DIR", "out/writing_agent_cache")
    
    # ===== Logging =====
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = os.getenv("LOG_DIR", "logs")
    
    @classmethod
    def get_model_name(cls, provider: Optional[str] = None) -> str:
        """Get model name for the specified provider"""
        provider = provider or cls.LLM_PROVIDER
        
        if provider == "openai":
            return cls.OPENAI_MODEL
        elif provider == "anthropic":
            return cls.ANTHROPIC_MODEL
        elif provider == "qwen":
            return cls.QWEN_MODEL
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    @classmethod
    def get_api_key(cls, provider: Optional[str] = None) -> Optional[str]:
        """Get API key for the specified provider"""
        provider = provider or cls.LLM_PROVIDER
        
        if provider == "openai":
            return cls.OPENAI_API_KEY
        elif provider == "anthropic":
            return cls.ANTHROPIC_API_KEY
        elif provider == "qwen":
            return cls.QWEN_API_KEY
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present"""
        provider = cls.LLM_PROVIDER
        api_key = cls.get_api_key(provider)
        
        if not api_key:
            raise ValueError(
                f"API key not found for provider '{provider}'. "
                f"Please set the appropriate environment variable."
            )
        
        return True
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """Get a summary of current configuration (safe for logging)"""
        return {
            "llm_provider": cls.LLM_PROVIDER,
            "model_name": cls.get_model_name(),
            "temperature": cls.TEMPERATURE,
            "retrieval_top_k": cls.RETRIEVAL_TOP_K,
            "max_iterations": cls.MAX_ITERATIONS,
            "quality_threshold": cls.QUALITY_THRESHOLD,
            "enable_memory": cls.ENABLE_MEMORY,
            "api_key_present": cls.get_api_key() is not None
        }


# Create a default config instance
config = WritingAgentConfig()
