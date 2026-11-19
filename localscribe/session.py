"""Coordinación de captura, segmentación, reconocimiento y respaldo incremental."""

from __future__ import annotations

import logging
import os
import queue
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from .audio import AUDIO_SOURCE_SYSTEM, AudioCapture
from .pipeline import RecognitionEngine, Segmenter

DEFAULT_SESSIONS_DIR = Path(__file__).resolve().parent.parent / "sessions"


class Signals(QObject):
    partial = Signal(int, str, int)
    final = Signal(int, str, str, int)
    metrics = Signal(dict)
    status = Signal(str)
    device = Signal(str)
    warning = Signal(str)
    error = Signal(str)
    finished = Signal()


class SessionBackupWriter:
    """Vuelca incrementalmente cada bloque confirmado en un archivo bajo sessions/."""

    def __init__(
        self,
        sessions_dir: Path | None = None,
        started_at: datetime | None = None,
    ):
        self.sessions_dir = (
            sessions_dir if sessions_dir is not None else DEFAULT_SESSIONS_DIR
        )
        self.started_at = started_at or datetime.now()
        self.file_path: Path | None = None
        self._lock = threading.Lock()

    def _resolve_path(self) -> Path:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        base = f"sesion_{self.started_at:%Y%m%d_%H%M%S}"
        candidate = self.sessions_dir / f"{base}.txt"
        suffix = 1
        while candidate.exists():
            candidate = self.sessions_dir / f"{base}_{suffix}.txt"
            suffix += 1
        return candidate

    def append(self, timestamp: str, text: str) -> Path | None:
        """Agrega un bloque confirmado y sincroniza inmediatamente a disco."""
        cleaned = text.strip()
        if not cleaned:
            return self.file_path
        with self._lock:
            if self.file_path is None:
                self.file_path = self._resolve_path()
            with self.file_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{timestamp}] {cleaned}\n")
                handle.flush()
                os.fsync(handle.fileno())
            return self.file_path


class Session:
    def __init__(
        self,
        signals: Signals,
        model_name: str,
        threshold: float,
        initial_prompt: str = "",
        audio_source: str = AUDIO_SOURCE_SYSTEM,
        sessions_dir: Path | None = None,
    ):
        self.signals = signals
        self.audio_source = audio_source
        self.capture_stop = threading.Event()
        self.segmenter_completed = threading.Event()
        self.audio_queue: queue.Queue = queue.Queue(maxsize=160)
        self.backup = SessionBackupWriter(sessions_dir=sessions_dir)
        self.engine = RecognitionEngine(
            self.segmenter_completed,
            signals,
            model_name,
            initial_prompt,
            on_final=self._record_final,
        )
        self.capture = AudioCapture(
            self.audio_queue, self.capture_stop, signals, source=audio_source
        )
        self.segmenter = Segmenter(
            self.audio_queue,
            self.capture_stop,
            self.segmenter_completed,
            self.engine,
            signals,
            energy_threshold=threshold,
            producer_done=self.capture.finished,
        )
        self.monitor = threading.Thread(target=self._monitor, daemon=True)

    def _record_final(self, timestamp: str, text: str) -> None:
        try:
            self.backup.append(timestamp, text)
        except OSError as exc:
            self.signals.warning.emit(
                f"No se pudo guardar el respaldo incremental de sesión: {exc}"
            )

    def start(self) -> None:
        logger = logging.getLogger("localscribe")
        if not logger.handlers:
            try:
                folder = Path(__file__).resolve().parent.parent / "logs"
                folder.mkdir(exist_ok=True)
                handler = RotatingFileHandler(
                    folder / "diagnostico.log",
                    maxBytes=1_000_000,
                    backupCount=2,
                    encoding="utf-8",
                )
                handler.setFormatter(
                    logging.Formatter("%(asctime)s %(levelname)s %(message)s")
                )
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
            except OSError as exc:
                self.signals.warning.emit(
                    f"No se pudo abrir el registro de diagnóstico: {exc}"
                )
        self.engine.start()
        self.segmenter.start()
        self.capture.start()
        self.monitor.start()

    def stop(self) -> None:
        self.capture_stop.set()

    def _monitor(self) -> None:
        self.capture.join()
        self.segmenter.join()
        self.engine.join()
        self.signals.finished.emit()
