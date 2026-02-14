"""
Model Discovery

Queries Ollama for available models (local and remote), maintains a
cached catalog, and detects when newer/better models are available.
"""

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CATALOG_PATH = Path.home() / ".ai_eval" / "models_catalog.json"


@dataclass
class ModelEntry:
    """A model in the catalog."""

    name: str
    size_gb: float = 0.0
    parameter_count: str = ""  # e.g., "32B", "7B"
    family: str = ""  # e.g., "qwen2.5", "llama3"
    capabilities: List[str] = field(default_factory=list)  # text, vision, code, embedding
    quantization: str = ""  # e.g., "Q4_K_M", "FP16"
    is_local: bool = False  # Currently downloaded
    last_seen: str = ""  # ISO date when last detected


@dataclass
class ModelCatalog:
    """Cached catalog of available models."""

    models: List[ModelEntry] = field(default_factory=list)
    last_refreshed: str = ""
    source: str = "ollama"

    def save(self, path: Path = DEFAULT_CATALOG_PATH) -> None:
        """Save catalog to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "models": [asdict(m) for m in self.models],
            "last_refreshed": self.last_refreshed,
            "source": self.source,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path = DEFAULT_CATALOG_PATH) -> "ModelCatalog":
        """Load catalog from JSON, or return empty if not found."""
        if not path.exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        models = [ModelEntry(**m) for m in data.get("models", [])]
        return cls(
            models=models,
            last_refreshed=data.get("last_refreshed", ""),
            source=data.get("source", "ollama"),
        )

    def find(self, name: str) -> Optional[ModelEntry]:
        """Find a model by name."""
        for m in self.models:
            if m.name == name:
                return m
        return None

    def search(
        self,
        capability: Optional[str] = None,
        max_size_gb: Optional[float] = None,
        min_size_gb: Optional[float] = None,
        local_only: bool = False,
    ) -> List[ModelEntry]:
        """Filter catalog by criteria."""
        results = list(self.models)

        if capability:
            results = [m for m in results if capability in m.capabilities]
        if max_size_gb is not None:
            results = [m for m in results if m.size_gb <= max_size_gb]
        if min_size_gb is not None:
            results = [m for m in results if m.size_gb >= min_size_gb]
        if local_only:
            results = [m for m in results if m.is_local]

        return sorted(results, key=lambda m: m.size_gb)


def _parse_size_to_gb(size_str: str) -> float:
    """Parse size string like '19 GB' or '4.7 GB' to float."""
    try:
        parts = size_str.strip().split()
        if not parts:
            return 0.0
        value = float(parts[0])
        if len(parts) > 1:
            unit = parts[1].upper()
            if unit == "MB":
                return value / 1024
            if unit == "GB":
                return value
            if unit == "TB":
                return value * 1024
        return value
    except (ValueError, IndexError):
        return 0.0


def _detect_capabilities(name: str) -> List[str]:
    """Infer model capabilities from name."""
    caps = ["text"]
    name_lower = name.lower()

    if any(kw in name_lower for kw in ("vision", "vl", "llava", "florence")):
        caps.append("vision")
    if any(kw in name_lower for kw in ("code", "coder", "starcoder", "deepseek-coder")):
        caps.append("code")
    if any(kw in name_lower for kw in ("embed", "nomic", "bge", "e5")):
        caps = ["embedding"]
    if any(kw in name_lower for kw in ("r1", "thinking", "reason")):
        caps.append("reasoning")

    return caps


def _extract_param_count(name: str) -> str:
    """Extract parameter count from model name like 'qwen2.5:32b'."""
    if ":" in name:
        tag = name.split(":")[-1].lower()
        for suffix in ("b", "m"):
            if tag.endswith(suffix) and tag[:-1].replace(".", "").isdigit():
                return tag.upper()
    return ""


async def refresh_local_models(catalog_path: Path = DEFAULT_CATALOG_PATH) -> ModelCatalog:
    """Refresh catalog from locally installed Ollama models.

    Calls `ollama list` to get locally downloaded models.
    """
    catalog = ModelCatalog.load(catalog_path)

    # Mark all existing as not local (we'll re-mark found ones)
    for m in catalog.models:
        m.is_local = False

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("ollama list failed: %s", result.stderr)
            return catalog
    except FileNotFoundError:
        logger.warning("ollama not found in PATH")
        return catalog
    except subprocess.TimeoutExpired:
        logger.warning("ollama list timed out")
        return catalog

    today = datetime.now().strftime("%Y-%m-%d")

    for line in result.stdout.strip().split("\n")[1:]:  # Skip header
        parts = line.split()
        if not parts:
            continue

        name = parts[0]
        size_str = ""
        # Ollama list format: NAME ID SIZE MODIFIED
        if len(parts) >= 3:
            size_str = f"{parts[2]} {parts[3]}" if len(parts) >= 4 else parts[2]

        existing = catalog.find(name)
        if existing:
            existing.is_local = True
            existing.last_seen = today
        else:
            catalog.models.append(
                ModelEntry(
                    name=name,
                    size_gb=_parse_size_to_gb(size_str),
                    parameter_count=_extract_param_count(name),
                    family=name.split(":")[0] if ":" in name else name,
                    capabilities=_detect_capabilities(name),
                    is_local=True,
                    last_seen=today,
                )
            )

    catalog.last_refreshed = today
    catalog.save(catalog_path)
    return catalog


def check_for_updates(
    catalog: ModelCatalog,
    current_models: List[str],
) -> List[Dict[str, Any]]:
    """Check if any current models have newer versions or better alternatives.

    Returns list of advisory dicts with model info and recommendation.
    """
    advisories: List[Dict[str, Any]] = []

    for model_name in current_models:
        entry = catalog.find(model_name)
        if not entry:
            continue

        # Find models in same family with different sizes
        family = entry.family
        alternatives = [
            m for m in catalog.models if m.family == family and m.name != model_name and m.is_local
        ]
        if alternatives:
            advisories.append(
                {
                    "current": model_name,
                    "alternatives": [m.name for m in alternatives],
                    "type": "alternative_available",
                }
            )

    return advisories
