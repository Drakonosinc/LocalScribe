"""User configuration persistence in config.json."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .audio import AUDIO_SOURCE_SYSTEM, VALID_AUDIO_SOURCES

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
VALID_MODELS = ("tiny", "base", "small")
MIN_THRESHOLD = 0.001
MAX_THRESHOLD = 0.060
MAX_VOCABULARY_LENGTH = 600

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """User preferences persisted across LocalScribe sessions."""

    model: str = "base"
    threshold: float = 0.007
    audio_source: str = AUDIO_SOURCE_SYSTEM
    always_on_top: bool = False
    vocabulary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AppConfig:
        defaults = cls()

        model = raw.get("model", defaults.model)
        if not isinstance(model, str) or model not in VALID_MODELS:
            model = defaults.model

        raw_threshold = raw.get("threshold", defaults.threshold)
        if isinstance(raw_threshold, (int, float)) and not isinstance(
            raw_threshold, bool
        ):
            threshold = max(MIN_THRESHOLD, min(MAX_THRESHOLD, float(raw_threshold)))
        else:
            threshold = defaults.threshold

        audio_source = raw.get("audio_source", defaults.audio_source)
        if not isinstance(audio_source, str) or audio_source not in VALID_AUDIO_SOURCES:
            audio_source = defaults.audio_source

        raw_top = raw.get("always_on_top", defaults.always_on_top)
        always_on_top = (
            bool(raw_top) if isinstance(raw_top, bool) else defaults.always_on_top
        )

        raw_vocab = raw.get("vocabulary", defaults.vocabulary)
        vocabulary = (
            raw_vocab[:MAX_VOCABULARY_LENGTH]
            if isinstance(raw_vocab, str)
            else defaults.vocabulary
        )

        return cls(
            model=model,
            threshold=threshold,
            audio_source=audio_source,
            always_on_top=always_on_top,
            vocabulary=vocabulary,
        )


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load AppConfig from disk; return defaults if missing or invalid."""
    if not path.exists():
        return AppConfig()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s; using default config: %s", path, exc)
        return AppConfig()
    if not isinstance(data, dict):
        return AppConfig()
    return AppConfig.from_dict(data)


def save_config(config: AppConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Save AppConfig as pretty-printed UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n"
    path.write_text(payload, encoding="utf-8")
