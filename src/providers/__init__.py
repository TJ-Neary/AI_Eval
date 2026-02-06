"""
LLM Provider Abstraction Layer

Unified interface for multiple LLM backends (Ollama, Google, Anthropic, OpenAI).
Based on patterns from benchy and llmperf for consistent multi-provider support.

Usage:
    from src.providers import OllamaProvider, GoogleProvider, ProviderFactory

    # Direct instantiation
    provider = OllamaProvider(model="qwen2.5:32b")
    response = await provider.generate("What is 2+2?")

    # Via factory
    provider = ProviderFactory.create("google", model="gemini-2.5-flash")
    response = await provider.generate("Explain AI")
"""

from .base import (
    BaseProvider,
    GenerationConfig,
    GenerationMetrics,
    GenerationResponse,
    Message,
    ModelInfo,
    ProviderFactory,
    ProviderType,
    calculate_tokens_per_second,
)
from .ollama_provider import OllamaProvider
from .google_provider import GoogleProvider

__all__ = [
    # Base classes and types
    "BaseProvider",
    "GenerationConfig",
    "GenerationMetrics",
    "GenerationResponse",
    "Message",
    "ModelInfo",
    "ProviderFactory",
    "ProviderType",
    "calculate_tokens_per_second",
    # Providers
    "OllamaProvider",
    "GoogleProvider",
]
