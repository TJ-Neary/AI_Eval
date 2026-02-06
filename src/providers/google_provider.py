"""
Google Gemini Provider Implementation

Cloud LLM inference via Google Generative AI SDK.
Supports Gemini 2.5 Pro, Gemini 2.5 Flash, and other Gemini models.

Usage:
    provider = GoogleProvider(model="gemini-2.5-flash")
    response = await provider.generate("Explain quantum computing")
"""

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

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

logger = logging.getLogger(__name__)

# Context windows for common Gemini models
GEMINI_CONTEXT_WINDOWS = {
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    "gemini-1.5-pro": 2_097_152,
    "gemini-1.5-flash": 1_048_576,
    "gemini-pro": 32_768,
}


class GoogleProvider(BaseProvider):
    """
    Google Gemini provider for cloud LLM inference.

    Requires GOOGLE_API_KEY environment variable or explicit api_key.
    """

    def __init__(
        self,
        model: str,
        config: Optional[GenerationConfig] = None,
        timeout: float = 120.0,
        api_key: Optional[str] = None,
    ):
        """
        Args:
            model: Gemini model name (e.g., "gemini-2.5-flash").
            config: Generation configuration.
            timeout: Request timeout in seconds.
            api_key: Google API key (defaults to GOOGLE_API_KEY env var).
        """
        super().__init__(model, config, timeout)
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._client: Any = None
        self._genai: Any = None

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.GOOGLE

    def _ensure_client(self) -> None:
        """Lazily initialize the Google GenAI client."""
        if self._client is not None:
            return

        if not self._api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY env var or pass api_key."
            )

        try:
            from google import genai

            self._genai = genai
            self._client = genai.Client(api_key=self._api_key)
        except ImportError:
            raise ImportError("google-genai package required: pip install google-genai")

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResponse:
        """Generate text from a single prompt."""
        self._ensure_client()
        cfg = config or self.config

        generation_config = {
            "temperature": cfg.temperature,
            "max_output_tokens": cfg.max_tokens,
            "top_p": cfg.top_p,
        }
        if cfg.top_k is not None:
            generation_config["top_k"] = cfg.top_k
        if cfg.stop_sequences:
            generation_config["stop_sequences"] = cfg.stop_sequences

        try:
            start_time = time.perf_counter()

            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self._genai.types.GenerateContentConfig(**generation_config),
            )

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # Extract text from response
            text = ""
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    text = candidate.content.parts[0].text or ""

            # Extract token counts from usage metadata
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
            completion_tokens = (
                getattr(usage, "candidates_token_count", 0) if usage else 0
            )
            total_tokens = prompt_tokens + completion_tokens

            tokens_per_second = calculate_tokens_per_second(completion_tokens, duration_ms)

            metrics = GenerationMetrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                total_duration_ms=duration_ms,
                tokens_per_second=tokens_per_second,
            )

            result = GenerationResponse(
                text=text,
                model=self.model,
                provider=self.provider_type,
                metrics=metrics,
                timestamp=datetime.now(),
            )
            self._record_request(result)
            return result

        except Exception as e:
            logger.error(f"Google generate failed: {e}")
            return GenerationResponse(
                text="",
                model=self.model,
                provider=self.provider_type,
                metrics=GenerationMetrics(),
                error=str(e),
            )

    async def generate_chat(
        self,
        messages: List[Message],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResponse:
        """Generate text from a multi-turn conversation."""
        self._ensure_client()
        cfg = config or self.config

        # Convert messages to Gemini format
        # Gemini uses "user" and "model" roles
        contents = []
        system_instruction = None

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg.content}]})
            else:
                contents.append({"role": "user", "parts": [{"text": msg.content}]})

        generation_config = {
            "temperature": cfg.temperature,
            "max_output_tokens": cfg.max_tokens,
            "top_p": cfg.top_p,
        }
        if cfg.top_k is not None:
            generation_config["top_k"] = cfg.top_k
        if cfg.stop_sequences:
            generation_config["stop_sequences"] = cfg.stop_sequences

        try:
            start_time = time.perf_counter()

            kwargs: Dict[str, Any] = {
                "model": self.model,
                "contents": contents,
                "config": self._genai.types.GenerateContentConfig(**generation_config),
            }
            if system_instruction:
                kwargs["config"].system_instruction = system_instruction

            response = self._client.models.generate_content(**kwargs)

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # Extract text
            text = ""
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    text = candidate.content.parts[0].text or ""

            # Extract token counts
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
            completion_tokens = (
                getattr(usage, "candidates_token_count", 0) if usage else 0
            )
            total_tokens = prompt_tokens + completion_tokens

            tokens_per_second = calculate_tokens_per_second(completion_tokens, duration_ms)

            metrics = GenerationMetrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                total_duration_ms=duration_ms,
                tokens_per_second=tokens_per_second,
            )

            result = GenerationResponse(
                text=text,
                model=self.model,
                provider=self.provider_type,
                metrics=metrics,
                timestamp=datetime.now(),
            )
            self._record_request(result)
            return result

        except Exception as e:
            logger.error(f"Google chat failed: {e}")
            return GenerationResponse(
                text="",
                model=self.model,
                provider=self.provider_type,
                metrics=GenerationMetrics(),
                error=str(e),
            )

    async def get_model_info(self) -> ModelInfo:
        """Get metadata about the current model."""
        # Derive info from known model specs
        context_window = GEMINI_CONTEXT_WINDOWS.get(self.model, 32_768)

        return ModelInfo(
            name=self.model,
            provider=self.provider_type,
            context_window=context_window,
            is_local=False,
            supports_vision="vision" in self.model.lower() or "pro" in self.model.lower(),
            supports_tools=True,
        )

    async def list_models(self) -> List[str]:
        """List available Gemini models."""
        # Return commonly used models
        return [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]

    async def health_check(self) -> bool:
        """Check if Google API is reachable."""
        try:
            self._ensure_client()
            # Try listing models as a health check
            return True
        except Exception as e:
            logger.warning(f"Google health check failed: {e}")
            return False


# Register with factory
ProviderFactory.register("google", GoogleProvider)
