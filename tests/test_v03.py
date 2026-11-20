"""Unit tests for v0.3.0: config.json, incremental sessions, export, and audio."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QLabel

from localscribe.audio import (
    AUDIO_SOURCE_BOTH,
    AUDIO_SOURCE_MIC,
    AUDIO_SOURCE_SYSTEM,
    AudioCapture,
    mix_audio_chunks,
    pcm_to_mono_16k,
)
from localscribe.config import AppConfig, load_config, save_config
from localscribe.pipeline import RecognitionEngine
from localscribe.session import SessionBackupWriter
from localscribe.transcript import (
    TranscriptEntry,
    export_transcript,
    format_markdown,
    format_srt,
    format_srt_timestamp,
    format_txt,
)
from localscribe.ui import MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _Recorder:
    def __init__(self):
        self.values = []

    def emit(self, *args):
        self.values.append(args)


def _dummy_signals():
    return SimpleNamespace(
        **{
            name: _Recorder()
            for name in ("partial", "final", "error", "warning", "status", "metrics")
        }
    )


# --- config.json persistence ---


def test_config_save_and_load_roundtrip(tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    original = AppConfig(
        model="small",
        threshold=0.015,
        audio_source=AUDIO_SOURCE_BOTH,
        always_on_top=True,
        vocabulary="Docker, Kubernetes, FastAPI, PyTorch",
    )
    save_config(original, cfg_path)
    loaded = load_config(cfg_path)
    assert loaded == original


def test_config_defaults_and_sanitization(tmp_path: Path):
    missing = tmp_path / "missing.json"
    assert load_config(missing) == AppConfig()

    corrupted = tmp_path / "corrupted.json"
    corrupted.write_text("{invalid", encoding="utf-8")
    assert load_config(corrupted) == AppConfig()

    invalid_values = tmp_path / "invalid.json"
    invalid_values.write_text(
        '{"model": "large-v3", "threshold": 5.0, "audio_source": "hdmi", '
        '"always_on_top": "yes", "vocabulary": 123}',
        encoding="utf-8",
    )
    loaded = load_config(invalid_values)
    assert loaded.model == "base"
    assert loaded.threshold == 0.060
    assert loaded.audio_source == AUDIO_SOURCE_SYSTEM
    assert loaded.always_on_top is False
    assert loaded.vocabulary == ""


def test_main_window_persists_and_restores_config(qapp, tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    win1 = MainWindow(config_path=cfg_path, sessions_dir=tmp_path / "sessions")
    assert win1.windowTitle() == "LocalScribe — Live Transcriber"
    title_label = win1.findChild(QLabel, "title")
    assert title_label is not None
    assert title_label.text() == "LocalScribe"
    assert "Docker, Kubernetes, FastAPI, PyTorch" in win1.vocabulary.placeholderText()

    win1.model.setCurrentIndex(win1.model.findData("tiny"))
    win1.threshold.setValue(0.021)
    win1.audio_source.setCurrentIndex(win1.audio_source.findData(AUDIO_SOURCE_MIC))
    win1.pin_button.setChecked(True)
    win1.vocabulary.setText("Docker, FastAPI")
    win1.close()

    assert cfg_path.exists()

    win2 = MainWindow(config_path=cfg_path, sessions_dir=tmp_path / "sessions")
    assert win2.model.currentData() == "tiny"
    assert win2.threshold.value() == pytest.approx(0.021)
    assert win2.audio_source.currentData() == AUDIO_SOURCE_MIC
    assert win2.pin_button.isChecked() is True
    assert win2.vocabulary.text() == "Docker, FastAPI"
    win2.close()


# --- Incremental session backups ---


def test_session_backup_writer_flushes_incrementally(tmp_path: Path):
    sessions_dir = tmp_path / "sessions"
    started = datetime(2026, 11, 19, 20, 30, 0)
    writer = SessionBackupWriter(sessions_dir=sessions_dir, started_at=started)

    assert writer.append("20:30:01", "   ") is None
    assert not sessions_dir.exists()

    first_path = writer.append("20:30:05", "First confirmed block.")
    assert first_path == sessions_dir / "session_20261119_203000.txt"
    assert first_path.read_text(encoding="utf-8") == (
        "[20:30:05] First confirmed block.\n"
    )

    second_path = writer.append("20:30:12", "Second confirmed block.")
    assert second_path == first_path
    assert first_path.read_text(encoding="utf-8") == (
        "[20:30:05] First confirmed block.\n[20:30:12] Second confirmed block.\n"
    )


def test_session_backup_writer_avoids_overwriting_same_timestamp(tmp_path: Path):
    sessions_dir = tmp_path / "sessions"
    started = datetime(2026, 11, 19, 20, 30, 0)
    writer_a = SessionBackupWriter(sessions_dir=sessions_dir, started_at=started)
    writer_b = SessionBackupWriter(sessions_dir=sessions_dir, started_at=started)

    path_a = writer_a.append("20:30:01", "Session A")
    path_b = writer_b.append("20:30:02", "Session B")

    assert path_a is not None and path_b is not None
    assert path_a != path_b
    assert path_a.read_text(encoding="utf-8") == "[20:30:01] Session A\n"
    assert path_b.read_text(encoding="utf-8") == "[20:30:02] Session B\n"


def test_recognition_engine_triggers_incremental_backup(tmp_path: Path):
    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, samples, **kwargs):
            return iter([SimpleNamespace(text="persisted block")]), None

    sessions_dir = tmp_path / "sessions"
    writer = SessionBackupWriter(
        sessions_dir=sessions_dir, started_at=datetime(2026, 11, 19, 21, 0, 0)
    )
    done = threading.Event()
    output = _dummy_signals()
    engine = RecognitionEngine(done, output, "tiny", on_final=writer.append)
    engine.submit_final(
        1, np.ones(16_000, dtype=np.float32), "21:00:04", 1, time.monotonic()
    )
    done.set()

    with patch("faster_whisper.WhisperModel", FakeModel):
        engine.start()
        engine.join(5)

    assert not engine.is_alive()
    assert writer.file_path is not None
    assert (
        writer.file_path.read_text(encoding="utf-8") == "[21:00:04] persisted block\n"
    )


# --- Multi-format export (.txt, .md, .srt) ---


def test_format_srt_timestamp_precision():
    assert format_srt_timestamp(0.0) == "00:00:00,000"
    assert format_srt_timestamp(3.5) == "00:00:03,500"
    assert format_srt_timestamp(3661.25) == "01:01:01,250"
    assert format_srt_timestamp(-2.0) == "00:00:00,000"


def test_export_txt_markdown_and_srt(tmp_path: Path):
    entries = [
        TranscriptEntry("12:00:05", "Hello world from LocalScribe.", 0.0, 3.2),
        TranscriptEntry("12:00:10", "Second confirmed line.", 5.0, 8.75),
    ]
    exported_at = datetime(2026, 11, 19, 12, 15, 0)

    txt_content = format_txt(entries)
    assert txt_content == (
        "[12:00:05] Hello world from LocalScribe.\n[12:00:10] Second confirmed line.\n"
    )

    md_content = format_markdown(entries, exported_at=exported_at)
    assert md_content.startswith("# Transcript — LocalScribe\n")
    assert "_Exported on 2026-11-19 12:15:00_" in md_content
    assert "- **`[12:00:05]`** Hello world from LocalScribe." in md_content
    assert "- **`[12:00:10]`** Second confirmed line." in md_content

    srt_content = format_srt(entries)
    assert srt_content == (
        "1\n"
        "00:00:00,000 --> 00:00:03,200\n"
        "Hello world from LocalScribe.\n\n"
        "2\n"
        "00:00:05,000 --> 00:00:08,750\n"
        "Second confirmed line.\n"
    )

    for ext in ("txt", "md", "srt"):
        target = tmp_path / f"output.{ext}"
        fmt = export_transcript(entries, target, exported_at=exported_at)
        assert fmt == ext
        assert target.exists()

    with pytest.raises(ValueError):
        export_transcript(entries, tmp_path / "output.docx")


def test_main_window_export_md_and_srt(qapp, tmp_path: Path):
    win = MainWindow(config_path=tmp_path / "config.json")
    win._on_metrics({"kind": "final", "audio": 2.5})
    win._on_final(1, "Subtitle export test", "14:00:01", 1)

    md_target = tmp_path / "meeting"
    with patch(
        "localscribe.ui.QFileDialog.getSaveFileName",
        return_value=(str(md_target), "Markdown (*.md)"),
    ):
        win._export()
    md_file = tmp_path / "meeting.md"
    assert md_file.exists()
    assert "**`[14:00:01]`** Subtitle export test" in md_file.read_text(
        encoding="utf-8"
    )

    srt_target = tmp_path / "meeting.srt"
    with patch(
        "localscribe.ui.QFileDialog.getSaveFileName",
        return_value=(str(srt_target), "SRT Subtitles (*.srt)"),
    ):
        win._export()
    srt_text = srt_target.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:02,500" in srt_text
    assert "Subtitle export test" in srt_text
    win.close()


# --- Audio utilities and source validation ---


def test_audio_helpers_and_source_validation():
    stereo = np.array([16384, -16384, 32767, 32767], dtype=np.int16).tobytes()
    mono = pcm_to_mono_16k(stereo, channels=2, rate=16_000)
    assert mono.dtype == np.float32
    assert len(mono) == 2
    assert mono[0] == pytest.approx(0.0, abs=1e-4)
    assert mono[1] == pytest.approx(1.0, abs=1e-3)

    a = np.array([0.8, 0.5], dtype=np.float32)
    b = np.array([0.6, -0.5, 0.25], dtype=np.float32)
    mixed = mix_audio_chunks(a, b)
    assert np.allclose(mixed, np.array([1.0, 0.0, 0.25], dtype=np.float32))
    assert np.allclose(mix_audio_chunks(a, None), a)
    assert np.allclose(mix_audio_chunks(None, b), b)
    assert mix_audio_chunks(None, None).size == 0

    with pytest.raises(ValueError):
        AudioCapture(queue.Queue(), threading.Event(), _dummy_signals(), source="other")
