"""User configuration persistence in config.json and per-user data folders."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .audio import AUDIO_SOURCE_SYSTEM, VALID_AUDIO_SOURCES

PROJECT_DIR = Path(__file__).resolve().parent.parent


def user_data_dir() -> Path:
    """Per-user writable folder (``%APPDATA%\\LocalScribe`` on Windows).

    Keeps working when the app is packaged or installed in a read-only folder.
    ``LOCALSCRIBE_HOME`` overrides the location (portable mode / tests).
    """
    override = os.environ.get("LOCALSCRIBE_HOME")
    if override:
        return Path(override)
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home() / ".config"
    return base / "LocalScribe"


DATA_DIR = user_data_dir()
DEFAULT_CONFIG_PATH = DATA_DIR / "config.json"
DEFAULT_SESSIONS_DIR = DATA_DIR / "sessions"
DEFAULT_LOGS_DIR = DATA_DIR / "logs"
DEFAULT_DB_PATH = DATA_DIR / "library.db"
LEGACY_CONFIG_PATH = PROJECT_DIR / "config.json"

VALID_MODELS = ("tiny", "base", "small", "large-v3-turbo")
VALID_DEVICES = ("auto", "cpu", "cuda")
VALID_THEMES = ("dark", "light")
# "auto" lets Whisper detect the language once and then locks it for the session.
LANGUAGES: tuple[tuple[str, str], ...] = (
    ("auto", "Auto-detect"),
    ("es", "Español"),
    ("en", "English"),
    ("pt", "Português"),
    ("fr", "Français"),
    ("de", "Deutsch"),
    ("it", "Italiano"),
    ("ja", "日本語"),
    ("zh", "中文"),
)
VALID_LANGUAGES = tuple(code for code, _ in LANGUAGES)
MIN_THRESHOLD = 0.001
MAX_THRESHOLD = 0.060
MAX_VOCABULARY_LENGTH = 600
MAX_KEYWORDS_LENGTH = 300

logger = logging.getLogger(__name__)


def _pick(raw: dict[str, Any], key: str, valid: tuple[str, ...], default: str) -> str:
    value = raw.get(key, default)
    return value if isinstance(value, str) and value in valid else default


def _text(raw: dict[str, Any], key: str, default: str, limit: int) -> str:
    value = raw.get(key, default)
    return value[:limit] if isinstance(value, str) else default


@dataclass
class AppConfig:
    """User preferences persisted across LocalScribe sessions."""

    model: str = "base"
    threshold: float = 0.007
    audio_source: str = AUDIO_SOURCE_SYSTEM
    always_on_top: bool = False
    vocabulary: str = ""
    language: str = "auto"
    device: str = "auto"
    theme: str = "dark"
    keywords: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AppConfig:
        defaults = cls()

        raw_threshold = raw.get("threshold", defaults.threshold)
        if isinstance(raw_threshold, (int, float)) and not isinstance(
            raw_threshold, bool
        ):
            threshold = max(MIN_THRESHOLD, min(MAX_THRESHOLD, float(raw_threshold)))
        else:
            threshold = defaults.threshold

        raw_top = raw.get("always_on_top", defaults.always_on_top)
        always_on_top = raw_top if isinstance(raw_top, bool) else defaults.always_on_top

        return cls(
            model=_pick(raw, "model", VALID_MODELS, defaults.model),
            threshold=threshold,
            audio_source=_pick(
                raw, "audio_source", VALID_AUDIO_SOURCES, defaults.audio_source
            ),
            always_on_top=always_on_top,
            vocabulary=_text(
                raw, "vocabulary", defaults.vocabulary, MAX_VOCABULARY_LENGTH
            ),
            language=_pick(raw, "language", VALID_LANGUAGES, defaults.language),
            device=_pick(raw, "device", VALID_DEVICES, defaults.device),
            theme=_pick(raw, "theme", VALID_THEMES, defaults.theme),
            keywords=_text(raw, "keywords", defaults.keywords, MAX_KEYWORDS_LENGTH),
        )


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load AppConfig from disk; return defaults if missing or invalid.

    When the default per-user config does not exist yet, a legacy
    ``config.json`` next to the code (v0.3) is migrated transparently.
    """
    if (
        not path.exists()
        and path == DEFAULT_CONFIG_PATH
        and LEGACY_CONFIG_PATH.exists()
    ):
        path = LEGACY_CONFIG_PATH
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
    """Save AppConfig as pretty-printed UTF-8 JSON (atomic replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n"
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(payload, encoding="utf-8")
    os.replace(temp, path)
