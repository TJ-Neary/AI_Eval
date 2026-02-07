"""
Ollama Provider Implementation

Local LLM inference via Ollama API.
Supports throughput measurement, concurrent requests, and model metadata.

Usage:
    provider = OllamaProvider(model="qwen2.5:32b")
    response = await provider.generate("Explain quantum computing")
    print(f"{response.metrics.tokens_per_second:.1f} t/s")
"""

import logging
from datetime import datetime
from typing import Any, List, Optional

from ollama import AsyncClient

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


class OllamaProvider(BaseProvider):
    """
    Ollama provider for local LLM inference.

    Connects to Ollama server (default: http://localhost:11434).
    Supports all Ollama models: qwen2.5, llama3.2, deepseek, etc.
    """

    def __init__(
        self,
        model: str,
        config: Optional[GenerationConfig] = None,
        timeout: float = 120.0,
        host: str = "http://localhost:11434",
    ):
        """
        Args:
            model: Ollama model name (e.g., "qwen2.5:32b", "llama3.2:3b").
            config: Generation configuration.
            timeout: Request timeout in seconds.
            host: Ollama server URL.
        """
        super().__init__(model, config, timeout)
        self.host = host
        self._client = AsyncClient(host=host)

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OLLAMA

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResponse:
        """Generate text from a single prompt."""
        cfg = config or self.config

        options = {
            "temperature": cfg.temperature,
            "num_predict": cfg.max_tokens,
            "top_p": cfg.top_p,
        }
        if cfg.top_k is not None:
            options["top_k"] = cfg.top_k
        if cfg.seed is not None:
            options["seed"] = cfg.seed

        try:
            response = await self._client.generate(
                model=self.model,
                prompt=prompt,
                options=options,
                stream=False,
            )

            metrics = self._extract_metrics(response)
            result = GenerationResponse(
                text=response.get("response", ""),
                model=self.model,
                provider=self.provider_type,
                metrics=metrics,
                timestamp=datetime.now(),
                raw_response=dict(response),
            )
            self._record_request(result)
            return result

        except Exception as e:
            logger.error(f"Ollama generate failed: {e}")
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
        cfg = config or self.config

        # Convert Message objects to Ollama format
        ollama_messages = [{"role": m.role, "content": m.content} for m in messages]

        options = {
            "temperature": cfg.temperature,
            "num_predict": cfg.max_tokens,
            "top_p": cfg.top_p,
        }
        if cfg.top_k is not None:
            options["top_k"] = cfg.top_k
        if cfg.seed is not None:
            options["seed"] = cfg.seed

        try:
            response = await self._client.chat(
                model=self.model,
                messages=ollama_messages,
                options=options,
                stream=False,
            )

            metrics = self._extract_metrics(response)
            result = GenerationResponse(
                text=response.get("message", {}).get("content", ""),
                model=self.model,
                provider=self.provider_type,
                metrics=metrics,
                timestamp=datetime.now(),
                raw_response=dict(response),
            )
            self._record_request(result)
            return result

        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            return GenerationResponse(
                text="",
                model=self.model,
                provider=self.provider_type,
                metrics=GenerationMetrics(),
                error=str(e),
            )

    async def get_model_info(self) -> ModelInfo:
        """Get metadata about the current model."""
        try:
            info = await self._client.show(self.model)
            details = info.get("details", {})
            model_info = info.get("model_info", {})

            # Extract context window from model info
            context_window = 0
            for key in model_info:
                if "context" in key.lower():
                    context_window = model_info[key]
                    break

            # Extract parameter count
            param_count = details.get("parameter_size", "")

            # Extract quantization
            quantization = details.get("quantization_level", "")

            return ModelInfo(
                name=self.model,
                provider=self.provider_type,
                context_window=context_window,
                quantization=quantization,
                parameter_count=param_count,
                is_local=True,
                supports_vision="vision" in self.model.lower(),
                supports_tools=True,  # Most Ollama models support tools
                extra={
                    "family": details.get("family", ""),
                    "format": details.get("format", ""),
                    "families": details.get("families", []),
                },
            )

        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return ModelInfo(
                name=self.model,
                provider=self.provider_type,
                is_local=True,
            )

    async def list_models(self) -> List[str]:
        """List available models from Ollama."""
        try:
            response = await self._client.list()
            return [m.get("name", "") for m in response.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    async def health_check(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            await self._client.list()
            return True
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    def _extract_metrics(self, response: Any) -> GenerationMetrics:
        """Extract performance metrics from Ollama response."""
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        # Ollama returns durations in nanoseconds
        prompt_eval_ns = response.get("prompt_eval_duration", 0)
        eval_ns = response.get("eval_duration", 0)
        total_ns = response.get("total_duration", 0)

        prompt_eval_ms = prompt_eval_ns / 1_000_000
        eval_ms = eval_ns / 1_000_000
        total_ms = total_ns / 1_000_000

        tokens_per_second = calculate_tokens_per_second(completion_tokens, eval_ms)

        # Calculate time to first token (load + prompt eval)
        load_ns = response.get("load_duration", 0)
        ttft_ms = (load_ns + prompt_eval_ns) / 1_000_000 if load_ns else None

        return GenerationMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_eval_duration_ms=prompt_eval_ms,
            eval_duration_ms=eval_ms,
            total_duration_ms=total_ms,
            tokens_per_second=tokens_per_second,
            time_to_first_token_ms=ttft_ms,
        )


# Register with factory
ProviderFactory.register("ollama", OllamaProvider)
