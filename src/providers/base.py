"""
Base Provider Abstraction Layer

Defines the interface that all LLM providers must implement.
Based on patterns from benchy and llmperf for consistent multi-provider support.

Usage:
    from src.providers import OllamaProvider, GoogleProvider

    provider = OllamaProvider(model="qwen2.5:32b")
    response = await provider.generate("What is 2+2?")
    print(response.text, response.tokens_per_second)
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported LLM provider types."""

    OLLAMA = auto()
    GOOGLE = auto()
    ANTHROPIC = auto()
    OPENAI = auto()


@dataclass
class ModelInfo:
    """Metadata about a model."""

    name: str
    provider: ProviderType
    context_window: int = 0
    quantization: Optional[str] = None
    parameter_count: Optional[str] = None  # e.g., "32B", "7B"
    is_local: bool = False
    supports_vision: bool = False
    supports_tools: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    temperature: float = 0.1
    max_tokens: int = 2048
    top_p: float = 1.0
    top_k: Optional[int] = None
    stop_sequences: List[str] = field(default_factory=list)
    seed: Optional[int] = None  # For reproducibility


@dataclass
class GenerationMetrics:
    """Performance metrics from a generation request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_eval_duration_ms: float = 0.0
    eval_duration_ms: float = 0.0
    total_duration_ms: float = 0.0
    tokens_per_second: float = 0.0
    time_to_first_token_ms: Optional[float] = None


@dataclass
class GenerationResponse:
    """Response from a generation request."""

    text: str
    model: str
    provider: ProviderType
    metrics: GenerationMetrics
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Whether the generation succeeded."""
        return self.error is None and len(self.text) > 0


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement:
    - generate(): Single prompt generation
    - generate_chat(): Multi-turn conversation
    - get_model_info(): Model metadata
    - list_models(): Available models
    - health_check(): Connectivity test
    """

    def __init__(
        self,
        model: str,
        config: Optional[GenerationConfig] = None,
        timeout: float = 120.0,
    ):
        """
        Args:
            model: Model name/identifier.
            config: Generation configuration (temperature, max_tokens, etc.).
            timeout: Request timeout in seconds.
        """
        self.model = model
        self.config = config or GenerationConfig()
        self.timeout = timeout
        self._request_count = 0
        self._total_tokens = 0

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type enum."""
        ...

    @property
    def is_local(self) -> bool:
        """Whether this provider runs locally (no API costs)."""
        return self.provider_type == ProviderType.OLLAMA

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResponse:
        """
        Generate text from a single prompt.

        Args:
            prompt: The input prompt.
            config: Override default generation config.

        Returns:
            GenerationResponse with text, metrics, and metadata.
        """
        ...

    @abstractmethod
    async def generate_chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResponse:
        """
        Generate text from a multi-turn conversation.

        Args:
            messages: List of Message objects (system, user, assistant).
            config: Override default generation config.

        Returns:
            GenerationResponse with text, metrics, and metadata.
        """
        ...

    @abstractmethod
    async def get_model_info(self) -> ModelInfo:
        """
        Get metadata about the current model.

        Returns:
            ModelInfo with context window, quantization, etc.
        """
        ...

    @abstractmethod
    async def list_models(self) -> List[str]:
        """
        List available models from this provider.

        Returns:
            List of model names/identifiers.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is reachable and working.

        Returns:
            True if healthy, False otherwise.
        """
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this provider instance."""
        return {
            "model": self.model,
            "provider": self.provider_type.name,
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
        }

    def _record_request(self, response: GenerationResponse) -> None:
        """Record metrics from a request."""
        self._request_count += 1
        self._total_tokens += response.metrics.total_tokens


def calculate_tokens_per_second(tokens: int, duration_ms: float) -> float:
    """Calculate tokens per second from token count and duration."""
    if duration_ms <= 0:
        return 0.0
    return (tokens / duration_ms) * 1000


class ProviderFactory:
    """
    Factory for creating provider instances.

    Usage:
        provider = ProviderFactory.create("ollama", model="qwen2.5:32b")
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, provider_class: type) -> None:
        """Register a provider class."""
        cls._registry[name.lower()] = provider_class

    @classmethod
    def create(
        cls,
        provider_name: str,
        model: str,
        config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> BaseProvider:
        """
        Create a provider instance.

        Args:
            provider_name: Provider name (ollama, google, anthropic, openai).
            model: Model name/identifier.
            config: Generation configuration.
            **kwargs: Provider-specific arguments.

        Returns:
            Configured provider instance.

        Raises:
            ValueError: If provider is not registered.
        """
        provider_class = cls._registry.get(provider_name.lower())
        if provider_class is None:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown provider '{provider_name}'. Available: {available}"
            )
        return provider_class(model=model, config=config, **kwargs)

    @classmethod
    def available_providers(cls) -> List[str]:
        """List registered provider names."""
        return list(cls._registry.keys())
