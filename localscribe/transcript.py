"""Widget-independent copy state and multi-format transcript export."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

SUPPORTED_EXPORT_FORMATS = ("txt", "md", "srt", "vtt", "json")


@dataclass(frozen=True)
class TranscriptEntry:
    """Confirmed transcript block with wall-clock and relative timing metadata."""

    timestamp: str
    text: str
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    starred: bool = False
    speaker: str = ""


def format_srt_timestamp(seconds: float) -> str:
    """Format elapsed seconds as a SubRip timestamp (HH:MM:SS,mmm)."""
    total_ms = round(max(0.0, seconds) * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _estimate_duration(text: str) -> float:
    words = max(1, len(text.split()))
    return max(2.0, min(16.0, words * 0.45))


def _format_speaker_tag(speaker: str) -> str:
    if speaker == "mic":
        return "[You] "
    if speaker == "system":
        return "[Them] "
    return ""


def format_txt(entries: Sequence[TranscriptEntry]) -> str:
    """Format confirmed history as timestamped plain text."""
    lines = [
        f"[{entry.timestamp}] {_format_speaker_tag(entry.speaker)}{entry.text}"
        for entry in entries
        if entry.text.strip()
    ]
    return "\n".join(lines) + "\n" if lines else ""


def format_markdown(
    entries: Sequence[TranscriptEntry],
    title: str = "Transcript — LocalScribe",
    exported_at: datetime | None = None,
) -> str:
    """Format confirmed history as timestamped Markdown."""
    valid = [entry for entry in entries if entry.text.strip()]
    if not valid:
        return ""
    stamp = (exported_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {title}",
        "",
        f"_Exported on {stamp}_",
        "",
    ]
    for entry in valid:
        star = " ⭐" if entry.starred else ""
        speaker = (
            " **🎙 You:**"
            if entry.speaker == "mic"
            else (" **🔊 Them:**" if entry.speaker == "system" else "")
        )
        lines.append(f"- **`[{entry.timestamp}]`**{star}{speaker} {entry.text}")
    return "\n".join(lines) + "\n"


def _timed(entries: Sequence[TranscriptEntry]) -> list[tuple[float, float, str]]:
    """Monotonic, non-overlapping (start, end, text) cues for subtitle formats."""
    cues: list[tuple[float, float, str]] = []
    previous_end = 0.0
    for entry in entries:
        if not entry.text.strip():
            continue
        start = max(previous_end, entry.start_seconds)
        if entry.end_seconds > start:
            end = entry.end_seconds
        else:
            end = start + _estimate_duration(entry.text)
        previous_end = end
        cues.append((start, end, entry.text))
    return cues


def format_srt(entries: Sequence[TranscriptEntry]) -> str:
    """Format confirmed history as standard SubRip (.srt) subtitles."""
    blocks = [
        f"{idx}\n{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n{text}"
        for idx, (start, end, text) in enumerate(_timed(entries), start=1)
    ]
    return "\n\n".join(blocks) + "\n" if blocks else ""


def format_vtt(entries: Sequence[TranscriptEntry]) -> str:
    """Format confirmed history as WebVTT subtitles (HTML5 video players)."""
    cues = _timed(entries)
    if not cues:
        return ""
    blocks = ["WEBVTT"]
    for start, end, text in cues:
        stamp_a = format_srt_timestamp(start).replace(",", ".")
        stamp_b = format_srt_timestamp(end).replace(",", ".")
        blocks.append(f"{stamp_a} --> {stamp_b}\n{text}")
    return "\n\n".join(blocks) + "\n"


def format_json(
    entries: Sequence[TranscriptEntry], exported_at: datetime | None = None
) -> str:
    """Machine-readable export with timing and bookmarks."""
    payload = {
        "app": "LocalScribe",
        "exported_at": (exported_at or datetime.now()).isoformat(timespec="seconds"),
        "entries": [asdict(entry) for entry in entries if entry.text.strip()],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def export_transcript(
    entries: Sequence[TranscriptEntry],
    path: Path,
    fmt: str | None = None,
    exported_at: datetime | None = None,
) -> str:
    """Export confirmed entries to .txt, .md, .srt, .vtt, or .json in UTF-8."""
    resolved_fmt = (fmt or path.suffix.lstrip(".") or "txt").lower()
    if resolved_fmt == "markdown":
        resolved_fmt = "md"
    if resolved_fmt not in SUPPORTED_EXPORT_FORMATS:
        raise ValueError(f"Unsupported export format: {resolved_fmt}")

    if resolved_fmt == "txt":
        content = format_txt(entries)
    elif resolved_fmt == "md":
        content = format_markdown(entries, exported_at=exported_at)
    elif resolved_fmt == "srt":
        content = format_srt(entries)
    elif resolved_fmt == "vtt":
        content = format_vtt(entries)
    else:
        content = format_json(entries, exported_at=exported_at)

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
        # An earlier final can arrive while a newer utterance is already shown.
        if number >= self.partial_id or group == self.group:
            self._group(group)
            if text:
                self.confirmed.append(text)
            if number >= self.partial_id:
                self.draft = ""
        return True

    def recent(self) -> str:
        # Each segment covers disjoint audio: a draft replaces only its own ID.
        return " ".join([*self.confirmed, self.draft]).strip()
