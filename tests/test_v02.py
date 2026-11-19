"""Regresiones de concurrencia, copia y widgets Qt sin capturar audio real."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QApplication

from localscribe.pipeline import MAX_PENDING_SECONDS, RecognitionEngine, Segmenter
from localscribe.transcript import RecentTranscript
from localscribe.ui import MainWindow, ProtectedLive, SafeTranscript


class Recorder:
    def __init__(self):
        self.values = []

    def emit(self, *args):
        self.values.append(args)


def signals():
    return SimpleNamespace(
        **{
            name: Recorder()
            for name in ("partial", "final", "error", "status", "metrics")
        }
    )


class CopyTests(unittest.TestCase):
    def test_partial_replaced_by_final_and_continuation_combined(self):
        state = RecentTranscript()
        state.partial(1, "palabra incorrecta", 1)
        state.final(1, "palabra correcta", 1)
        state.partial(2, "y continuación", 1)
        self.assertEqual(state.recent(), "palabra correcta y continuación")
        state.final(2, "y continuación corregida", 1)
        self.assertEqual(state.recent(), "palabra correcta y continuación corregida")
        self.assertFalse(state.partial(1, "viejo", 1))
        self.assertFalse(state.final(2, "duplicado", 1))

    def test_new_pause_group_does_not_copy_previous_question(self):
        state = RecentTranscript()
        state.final(1, "anterior", 1)
        state.partial(2, "pregunta actual", 2)
        self.assertEqual(state.recent(), "pregunta actual")

    def test_late_final_does_not_erase_newer_draft(self):
        state = RecentTranscript()
        state.partial(2, "actual", 2)
        state.final(1, "anterior", 1)
        self.assertEqual(state.recent(), "actual")


class EngineTests(unittest.TestCase):
    def test_bounded_final_queue_and_stale_partial(self):
        engine = RecognitionEngine(threading.Event(), signals(), "tiny")
        audio = np.ones(16_000 * 16, np.float32)
        for i in range(int(MAX_PENDING_SECONDS / 16)):
            engine.submit_final(i + 1, audio, "")
        with self.assertRaises(RuntimeError):
            engine.submit_final(99, audio, "")
        engine.submit_partial(1, audio, "")
        self.assertIsNone(engine.latest_partial)

    def test_lazy_inference_metrics_and_single_model(self):
        calls = []

        class Model:
            def __init__(self, *args, **kwargs):
                calls.append("model")

            def transcribe(self, samples, **kwargs):
                def results():
                    calls.append(kwargs)
                    yield SimpleNamespace(text=" texto ")

                return results(), None

        done = threading.Event()
        output = signals()
        engine = RecognitionEngine(done, output, "tiny", "Kubernetes")
        engine.submit_final(1, np.ones(16000, np.float32), "12:00", 1, time.monotonic())
        done.set()
        with patch("faster_whisper.WhisperModel", Model):
            engine.start()
            engine.join(5)
        self.assertFalse(engine.is_alive())
        self.assertFalse(output.error.values)
        self.assertEqual(output.final.values, [(1, "texto", "12:00", 1)])
        self.assertEqual(calls[1]["initial_prompt"], "Kubernetes")
        self.assertEqual(output.metrics.values[0][0]["audio"], 1)

    def test_capture_queue_drains_while_inference_is_busy(self):
        entered, release, done = threading.Event(), threading.Event(), threading.Event()

        class SlowModel:
            def __init__(self, *args, **kwargs):
                pass

            def transcribe(self, samples, **kwargs):
                def results():
                    entered.set()
                    release.wait(3)
                    yield SimpleNamespace(text="texto")

                return results(), None

        output = signals()
        engine = RecognitionEngine(done, output, "tiny")
        engine.submit_partial(1, np.ones(16000, np.float32), "", 1)
        producer = threading.Event()
        audio = queue.Queue()
        segmenter = Segmenter(
            audio, threading.Event(), done, engine, output, producer_done=producer
        )
        with patch("faster_whisper.WhisperModel", SlowModel):
            engine.start()
            self.assertTrue(entered.wait(2))
            for _ in range(20):
                audio.put((np.full(1600, 0.04, np.float32), time.monotonic()))
            producer.set()
            segmenter.start()
            segmenter.join(2)
            self.assertFalse(segmenter.is_alive())
            self.assertTrue(audio.empty())
            self.assertTrue(engine.is_alive())
            release.set()
            engine.join(5)
        self.assertFalse(engine.is_alive())
        self.assertEqual(len(output.partial.values), 1)
        self.assertEqual(len(output.final.values), 1)

    def test_model_failure_stops_segmenter(self):
        done, stop = threading.Event(), threading.Event()
        engine = RecognitionEngine(done, signals(), "tiny")
        engine.failed.set()
        segmenter = Segmenter(queue.Queue(), stop, done, engine, signals())
        segmenter.start()
        segmenter.join(2)
        self.assertTrue(stop.is_set())
        self.assertTrue(done.is_set())

    def test_long_pause_after_forced_boundary_starts_new_group(self):
        engine = RecognitionEngine(threading.Event(), signals(), "tiny")
        segmenter = Segmenter(
            queue.Queue(), threading.Event(), threading.Event(), engine, signals()
        )
        for _ in range(160):
            segmenter.consume(np.full(1600, 0.04, np.float32))
        for _ in range(10):
            segmenter.consume(np.zeros(1600, np.float32))
        for _ in range(10):
            segmenter.consume(np.full(1600, 0.04, np.float32))
        segmenter._finish()
        self.assertEqual([t.group for t in engine.finals], [1, 2])

    def test_stop_drains_last_packet_after_stop_request(self):
        stop, producer, done = threading.Event(), threading.Event(), threading.Event()
        engine = RecognitionEngine(done, signals(), "tiny")
        audio = queue.Queue()
        segmenter = Segmenter(
            audio, stop, done, engine, signals(), producer_done=producer
        )
        segmenter.start()
        stop.set()
        audio.put((np.full(16000, 0.04, np.float32), time.monotonic()))
        producer.set()
        segmenter.join(3)
        self.assertFalse(segmenter.is_alive())
        self.assertEqual(len(engine.finals), 1)

    def test_overload_stops_capture_and_keeps_queued_audio(self):
        stop, producer, done = threading.Event(), threading.Event(), threading.Event()
        output = signals()
        engine = RecognitionEngine(done, output, "tiny")
        audio = queue.Queue()
        for _ in range(500):
            audio.put((np.full(1600, 0.04, np.float32), time.monotonic()))
        producer.set()
        segmenter = Segmenter(audio, stop, done, engine, output, producer_done=producer)
        with self.assertLogs("localscribe.pipeline", level="WARNING"):
            segmenter.start()
            segmenter.join(3)
        self.assertFalse(segmenter.is_alive())
        self.assertTrue(stop.is_set())
        self.assertTrue(audio.empty())
        self.assertEqual(sum(len(t.samples) for t in engine.finals), 500 * 1600)
        self.assertEqual(len(output.error.values), 1)

    def test_forced_boundary_does_not_repeat_audio_packet(self):
        engine = RecognitionEngine(threading.Event(), signals(), "tiny")
        segmenter = Segmenter(
            queue.Queue(), threading.Event(), threading.Event(), engine, signals()
        )
        for _ in range(170):
            segmenter.consume(np.full(1600, 0.04, np.float32))
        segmenter._finish()
        self.assertEqual(sum(len(t.samples) for t in engine.finals), 170 * 1600)
        self.assertEqual([t.group for t in engine.finals], [1, 1])


class WidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_history_selection_and_scrolling(self):
        widget = SafeTranscript()
        widget.resize(400, 180)
        widget.show()
        for i in range(70):
            widget.add_entry(f"Pregunta {i}")
        self.app.processEvents()
        bar = widget.verticalScrollBar()
        self.assertEqual(bar.value(), bar.maximum())
        bar.setValue(10)
        cursor = widget.textCursor()
        cursor.setPosition(4)
        cursor.setPosition(18, QTextCursor.MoveMode.KeepAnchor)
        widget.setTextCursor(cursor)
        bar.setValue(10)
        selected = widget.textCursor().selectedText()
        widget.add_entry("Nuevo texto")
        self.app.processEvents()
        self.assertEqual(bar.value(), 10)
        self.assertEqual(widget.textCursor().selectedText(), selected)
        cursor.clearSelection()
        widget.setTextCursor(cursor)
        bar.setValue(bar.maximum())
        widget.add_entry("Otra actualización")
        self.assertEqual(bar.value(), bar.maximum())
        widget.close()

    def test_draft_selection_is_frozen_until_released(self):
        widget = ProtectedLive()
        widget.update_text("borrador anterior")
        widget.selectAll()
        widget.update_text("borrador corregido")
        self.assertEqual(widget.textCursor().selectedText(), "borrador anterior")
        self.assertEqual(widget.latest_text, "borrador corregido")
        cursor = widget.textCursor()
        cursor.clearSelection()
        widget.setTextCursor(cursor)
        self.assertEqual(widget.toPlainText(), "borrador corregido")

    def test_mouse_selection_prevents_follow_even_without_selected_text(self):
        widget = SafeTranscript()
        widget.resize(400, 180)
        widget.show()
        for _ in range(50):
            widget.add_entry("texto")
        self.app.processEvents()
        bar = widget.verticalScrollBar()
        position = bar.value()
        widget._mouse_selecting = True
        widget.add_entry("nuevo")
        self.assertEqual(bar.value(), position)
        self.assertLess(bar.value(), bar.maximum())
        widget.close()

    def test_copy_buttons_and_final_transition(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            window = MainWindow(config_path=Path(folder) / "config.json")
            window._on_partial(1, "pregunta provisional", 1)
            window._copy_draft()
            self.assertEqual(self.app.clipboard().text(), "pregunta provisional")
            window._on_final(1, "pregunta confirmada", "12:00", 1)
            window._on_partial(2, "continuación", 1)
            window._copy_recent()
            self.assertEqual(
                self.app.clipboard().text(), "pregunta confirmada continuación"
            )
            self.assertEqual(window.live.toPlainText(), "continuación")
            window.close()

    def test_search_export_and_topmost_toggle(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            window = MainWindow(config_path=Path(folder) / "config.json")
            window._on_final(1, "pregunta guardada", "12:00", 1)
            window.search.setText("guardada")
            window._find_next()
            self.assertEqual(window.transcript.textCursor().selectedText(), "guardada")
            target = Path(folder) / "transcripcion.txt"
            with patch(
                "localscribe.ui.QFileDialog.getSaveFileName",
                return_value=(str(target), "Texto plano (*.txt)"),
            ):
                window._export()
            self.assertEqual(
                target.read_text(encoding="utf-8"), "[12:00] pregunta guardada\n"
            )
            from PySide6.QtCore import Qt

            window.pin_button.setChecked(True)
            self.assertTrue(window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
            window.pin_button.setChecked(False)
            self.assertFalse(window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
            window.close()


if __name__ == "__main__":
    unittest.main()
