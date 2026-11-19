"""Estado de copia independiente de los widgets y exportación multi-formato."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SUPPORTED_EXPORT_FORMATS = ("txt", "md", "srt")


@dataclass(frozen=True)
class TranscriptEntry:
    """Bloque de transcripción confirmado con metadatos de tiempo."""

    timestamp: str
    text: str
    start_seconds: float = 0.0
    end_seconds: float = 0.0


def format_srt_timestamp(seconds: float) -> str:
    """Formatea segundos como marca de tiempo SubRip (HH:MM:SS,mmm)."""
    total_ms = round(max(0.0, seconds) * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _estimate_duration(text: str) -> float:
    words = max(1, len(text.split()))
    return max(2.0, min(16.0, words * 0.45))


def format_txt(entries: Sequence[TranscriptEntry]) -> str:
    """Formatea el historial confirmado como texto plano con marcas de tiempo."""
    lines = [
        f"[{entry.timestamp}] {entry.text}" for entry in entries if entry.text.strip()
    ]
    return "\n".join(lines) + "\n" if lines else ""


def format_markdown(
    entries: Sequence[TranscriptEntry],
    title: str = "Transcripción — LocalScribe",
    exported_at: datetime | None = None,
) -> str:
    """Formatea el historial confirmado en Markdown con marcas de tiempo."""
    valid = [entry for entry in entries if entry.text.strip()]
    if not valid:
        return ""
    stamp = (exported_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {title}",
        "",
        f"_Exportado el {stamp}_",
        "",
    ]
    for entry in valid:
        lines.append(f"- **`[{entry.timestamp}]`** {entry.text}")
    return "\n".join(lines) + "\n"


def format_srt(entries: Sequence[TranscriptEntry]) -> str:
    """Formatea el historial confirmado en subtítulos estándar SubRip (.srt)."""
    valid = [entry for entry in entries if entry.text.strip()]
    if not valid:
        return ""
    blocks: list[str] = []
    previous_end = 0.0
    for idx, entry in enumerate(valid, start=1):
        start = max(previous_end, entry.start_seconds)
        if entry.end_seconds > start:
            end = entry.end_seconds
        else:
            end = start + _estimate_duration(entry.text)
        previous_end = end
        blocks.append(
            f"{idx}\n"
            f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n"
            f"{entry.text}"
        )
    return "\n\n".join(blocks) + "\n"


def export_transcript(
    entries: Sequence[TranscriptEntry],
    path: Path,
    fmt: str | None = None,
    exported_at: datetime | None = None,
) -> str:
    """Exporta entradas confirmadas a .txt, .md o .srt en UTF-8."""
    resolved_fmt = (fmt or path.suffix.lstrip(".") or "txt").lower()
    if resolved_fmt == "markdown":
        resolved_fmt = "md"
    if resolved_fmt not in SUPPORTED_EXPORT_FORMATS:
        raise ValueError(f"Formato de exportación no admitido: {resolved_fmt}")

    if resolved_fmt == "txt":
        content = format_txt(entries)
    elif resolved_fmt == "md":
        content = format_markdown(entries, exported_at=exported_at)
    else:
        content = format_srt(entries)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return resolved_fmt


class RecentTranscript:
    def __init__(self):
        self.group = None
        self.confirmed = []
        self.draft = ""
        self.completed = 0
        self.partial_id = 0

    def _group(self, group: int) -> None:
        if self.group != group:
            self.group = group
            self.confirmed = []
            self.draft = ""

    def partial(self, number: int, text: str, group: int) -> bool:
        if number <= self.completed or number < self.partial_id:
            return False
        self._group(group)
        self.partial_id = number
        self.draft = text
        return True

    def final(self, number: int, text: str, group: int) -> bool:
        if number <= self.completed:
            return False
        self.completed = number
        # Un final anterior puede llegar mientras ya se muestra otra intervención.
        if number >= self.partial_id or group == self.group:
            self._group(group)
            if text:
                self.confirmed.append(text)
            if number >= self.partial_id:
                self.draft = ""
        return True

    def recent(self) -> str:
        # Cada segmento tiene audio disjunto: el borrador sustituye solo a su ID.
        return " ".join([*self.confirmed, self.draft]).strip()
