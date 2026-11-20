"""Tests for v0.4: transcription quality, real timing, and the redesigned UI."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from localscribe.audio import TARGET_RATE
from localscribe.config import AppConfig, load_config, save_config
from localscribe.pipeline import (
    RecognitionEngine,
    Segmenter,
    clean_segments,
    is_hallucination,
)
from localscribe.transcript import TranscriptEntry, format_json, format_vtt
from localscribe.ui import MainWindow
from localscribe.widgets import entry_fragments, find_keywords, parse_keywords


class _Recorder:
    def __init__(self):
        self.values = []

    def emit(self, *args):
        self.values.append(args)


def _signals():
    return SimpleNamespace(
        **{
            name: _Recorder()
            for name in ("partial", "final", "error", "warning", "status", "metrics")
        }
    )


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _segmenter(engine=None):
    engine = engine or RecognitionEngine(threading.Event(), _signals(), "tiny")
    seg = Segmenter(
        queue.Queue(), threading.Event(), threading.Event(), engine, _signals()
    )
    return seg, engine


# --- Hallucination filtering -------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Gracias por ver el video.",
        "¡Suscríbete!",
        "Thank you for watching!",
        "Subtítulos realizados por la comunidad de Amara.org",
        "♪ ♪ ♪",
        "...",
    ],
)
def test_known_hallucinations_are_detected(text):
    assert is_hallucination(text)


@pytest.mark.parametrize(
    "text",
    ["Gracias por ver el video que mandé ayer, Juan.", "Hola a todos", "You know what"],
)
def test_real_speech_is_kept(text):
    assert not is_hallucination(text)


def test_clean_segments_drops_low_confidence_and_loops():
    segments = [
        SimpleNamespace(text=" Hola equipo.", no_speech_prob=0.1, avg_logprob=-0.3),
        SimpleNamespace(text=" ruido", no_speech_prob=0.9, avg_logprob=-1.4),
        SimpleNamespace(text=" no no no no no no", compression_ratio=3.1),
        SimpleNamespace(text=" Empecemos."),
    ]
    assert clean_segments(segments) == "Hola equipo. Empecemos."
    assert clean_segments([SimpleNamespace(text="Thanks for watching!")]) == ""


# --- Segmenter: real timing, smart cuts, adaptive noise -----------------------


def test_segments_carry_real_timeline_positions():
    seg, engine = _segmenter()
    silence = np.zeros(1600, np.float32)
    speech = np.full(1600, 0.04, np.float32)
    for _ in range(20):  # 2.0 s of silence
        seg.consume(silence)
    for _ in range(15):  # 1.5 s of speech
        seg.consume(speech)
    for _ in range(10):  # pause closes the segment
        seg.consume(silence)
    task = engine.finals[0]
    # Starts at 2.0 s minus the 0.3 s pre-roll; trailing silence is excluded.
    assert task.start_seconds == pytest.approx(1.7, abs=0.05)
    assert task.end_seconds == pytest.approx(3.5, abs=0.05)


def test_long_utterance_is_split_at_the_quietest_gap():
    seg, engine = _segmenter()
    loud = np.full(1600, 0.05, np.float32)
    dip = np.full(1600, 0.008, np.float32)  # still "speech", but a clear dip
    for i in range(160):
        seg.consume(dip if i == 152 else loud)
    seg._finish()
    first, second = engine.finals[0], engine.finals[1]
    # Cut in the middle of the dip (packet 152), not at the 16 s boundary.
    assert len(first.samples) == 152 * 1600 + 800
    assert first.group == second.group
    total = sum(len(t.samples) for t in engine.finals)
    assert total == 160 * 1600
    assert second.start_seconds == pytest.approx(len(first.samples) / TARGET_RATE)


def test_noise_floor_adapts_so_constant_hum_does_not_trigger():
    seg, engine = _segmenter()
    seg.threshold = 0.002
    hum = np.full(1600, 0.0015, np.float32)
    for _ in range(50):
        seg.consume(hum)
    louder_hum = np.full(1600, 0.0025, np.float32)
    for _ in range(5):
        seg.consume(louder_hum)
    assert seg.effective_threshold > 0.004
    assert not engine.finals and not seg.active
    assert seg.level == pytest.approx(0.0025, rel=1e-3)


# --- Engine: language lock, context prompt, VAD ------------------------------


def test_language_lock_and_context_prompt():
    calls = []

    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, samples, **kwargs):
            calls.append(kwargs)
            info = SimpleNamespace(language="es", language_probability=0.95)
            return iter([SimpleNamespace(text=f"frase {len(calls)}")]), info

    done = threading.Event()
    output = _signals()
    engine = RecognitionEngine(done, output, "tiny", "Kubernetes", language="auto")
    audio = np.ones(TARGET_RATE * 3, np.float32)
    engine.submit_final(1, audio, "10:00", 1, time.monotonic(), 0.0, 3.0)
    engine.submit_final(2, audio, "10:01", 1, time.monotonic(), 3.0, 6.0)
    done.set()
    with patch("faster_whisper.WhisperModel", Model):
        engine.start()
        engine.join(5)
    assert calls[0]["language"] is None
    assert calls[0]["vad_filter"] is True
    assert calls[1]["language"] == "es"
    assert calls[1]["initial_prompt"] == "Kubernetes frase 1"
    assert output.final.values[1] == (2, "frase 2", "10:01", 1, 3.0, 6.0)
    assert any("locked" in v[0] for v in output.status.values)


def test_vad_failure_falls_back_without_vad():
    calls = []

    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, samples, **kwargs):
            calls.append(kwargs["vad_filter"])
            if kwargs["vad_filter"]:
                raise RuntimeError("onnxruntime missing")
            return iter([SimpleNamespace(text="ok")]), None

    done = threading.Event()
    output = _signals()
    engine = RecognitionEngine(done, output, "tiny")
    engine.submit_final(1, np.ones(TARGET_RATE, np.float32), "", 1)
    done.set()
    with patch("faster_whisper.WhisperModel", Model):
        engine.start()
        engine.join(5)
    assert calls == [True, False]
    assert output.final.values[0][1] == "ok"
    assert not output.error.values


# --- Keywords, export, config -----------------------------------------------


def test_keyword_parsing_and_fragments():
    keywords = parse_keywords(" Cancino, deadline ,x, ")
    assert keywords == ["Cancino", "deadline"]
    text = "Cancino, the deadline is Friday; deadlines matter."
    assert find_keywords(text, keywords) == ["Cancino", "deadline"]
    fragments = entry_fragments("12:00", text, starred=True, keywords=keywords)
    roles = [role for _, role in fragments]
    assert roles[:2] == ["star", "time"]
    assert [t for t, r in fragments if r == "keyword"] == ["Cancino", "deadline"]
    assert "".join(t for t, _ in fragments) == f"★ [12:00] {text}"


def test_vtt_and_json_export():
    entries = [
        TranscriptEntry("12:00:05", "Hola.", 0.0, 1.5),
        TranscriptEntry("12:00:07", "Seguimos.", 2.0, 3.25, starred=True),
    ]
    vtt = format_vtt(entries)
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHola.")
    assert "00:00:02.000 --> 00:00:03.250\nSeguimos." in vtt
    data = json.loads(format_json(entries))
    assert data["entries"][1]["starred"] is True


def test_config_new_fields_roundtrip_and_sanitization(tmp_path: Path):
    path = tmp_path / "config.json"
    cfg = AppConfig(language="es", device="cuda", theme="light", keywords="Cancino")
    save_config(cfg, path)
    assert load_config(path) == cfg
    path.write_text(
        '{"language": "klingon", "device": "tpu", "theme": "neon", "model": "turbo"}',
        encoding="utf-8",
    )
    loaded = load_config(path)
    assert (loaded.language, loaded.device, loaded.theme, loaded.model) == (
        "auto",
        "auto",
        "dark",
        "base",
    )


# --- Main window ------------------------------------------------------------


def test_real_timing_is_used_for_entries(qapp, tmp_path: Path):
    win = MainWindow(config_path=tmp_path / "config.json")
    win._on_final(1, "primero", "10:00:00", 1, 1.0, 2.5)
    win._on_final(2, "segundo", "10:00:03", 1, 3.0, 4.0)
    assert [(e.start_seconds, e.end_seconds) for e in win.entries] == [
        (1.0, 2.5),
        (3.0, 4.0),
    ]
    win.close()


def test_star_keyword_and_theme(qapp, tmp_path: Path):
    win = MainWindow(config_path=tmp_path / "config.json")
    win.keywords_edit.setText("presupuesto")
    with patch("localscribe.ui.QApplication.alert") as alert:
        win._on_final(1, "Hablemos del presupuesto", "10:00:00", 1)
    alert.assert_called_once()
    assert "presupuesto" in win.status.text()
    win._star_last()
    assert win.entries[0].starred
    assert win.transcript.toPlainText().startswith("★ [10:00:00]")
    win._toggle_theme()
    assert win.current_config().theme == "light"
    assert win.transcript.toPlainText().startswith("★ [10:00:00]")
    win.search.setText("presupuesto")
    assert win.search_count.text() == "1 match"
    win.close()


def test_unseen_pill_appears_when_reading_older_text(qapp, tmp_path: Path):
    win = MainWindow(config_path=tmp_path / "config.json")
    win.resize(900, 700)
    win.show()
    for i in range(60):
        win._on_final(i + 1, f"line {i}", "10:00:00", i + 1)
    qapp.processEvents()
    win.transcript.verticalScrollBar().setValue(0)
    win._on_final(61, "fresh", "10:00:01", 61)
    assert win.jump_button.isVisible()
    assert win.jump_button.text() == "↓ 1 new"
    win.transcript.go_to_latest()
    assert not win.jump_button.isVisible()
    win.close()


def test_close_while_recording_closes_automatically(qapp, tmp_path: Path):
    win = MainWindow(config_path=tmp_path / "config.json")
    win.show()
    win.session = SimpleNamespace(stop=lambda: None, segmenter=None)
    win.is_recording = True
    win.close()
    assert win.isVisible() and win._closing
    win._on_finished()
    qapp.processEvents()
    assert not win.isVisible()


def test_caption_overlay_and_speaker_attribution(qapp, tmp_path: Path):
    seg, engine = _segmenter()
    speech = np.full(1600, 0.04, np.float32)
    silence = np.zeros(1600, np.float32)
    for _ in range(10):
        seg.consume(speech, speaker="mic")
    for _ in range(10):
        seg.consume(silence)
    assert engine.finals[0].speaker == "mic"

    win = MainWindow(config_path=tmp_path / "config.json")
    win.overlay_button.setChecked(True)
    assert win.caption_overlay.isVisible()
    win._on_partial(1, "probando subtítulos en vivo", 1)
    assert win.caption_overlay._draft == "probando subtítulos en vivo"

    win.session = SimpleNamespace(
        audio_source="both",
        engine=SimpleNamespace(last_speaker="mic"),
    )
    win._on_final(1, "Hola desde mi micrófono", "10:00:00", 1, 0.0, 2.0)
    assert win.entries[0].speaker == "mic"
    assert "🎙 You" in win.transcript.toPlainText()
    assert win.caption_overlay._confirmed == "Hola desde mi micrófono"
    win.overlay_button.setChecked(False)
    assert not win.caption_overlay.isVisible()
    win.session = None
    win.close()
