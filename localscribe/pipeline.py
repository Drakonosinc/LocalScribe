"""Segmentación por pausas y reconocimiento en un hilo independiente."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .audio import TARGET_RATE

logger = logging.getLogger(__name__)
MAX_PENDING_SECONDS = 128.0
BACKLOG_STOP_SECONDS = 48.0


@dataclass
class RecognitionTask:
    number: int
    samples: np.ndarray
    when: str
    group: int
    captured_end: float

    @property
    def duration(self) -> float:
        return len(self.samples) / TARGET_RATE


class RecognitionEngine(threading.Thread):
    """Usa un único modelo, priorizando fragmentos definitivos.

    Conserva una sola solicitud provisional (la más reciente) para no generar
    una cola interminable de tareas al transcribir con CPU.
    """

    def __init__(
        self,
        completed: threading.Event,
        signals,
        model_name: str,
        initial_prompt: str = "",
        on_final: Callable[[str, str], None] | None = None,
    ):
        super().__init__(name="Reconocimiento", daemon=True)
        self.completed = completed
        self.signals = signals
        self.model_name = model_name
        self.condition = threading.Condition()
        self.finals: deque[RecognitionTask] = deque()
        self.latest_partial: RecognitionTask | None = None
        self.initial_prompt = initial_prompt.strip() or None
        self.on_final = on_final
        self.failed = threading.Event()
        self.busy = False
        self.partial_interval = 0.8
        self.last_final = 0
        self.capture_pending = 0.0
        self.inference_started: float | None = None

    def wants_partial(self) -> bool:
        with self.condition:
            return not self.busy and not self.finals and not self.failed.is_set()

    def pending_seconds(self) -> float:
        with self.condition:
            # El borrador solapa el segmento activo; no se cuenta dos veces.
            return sum(task.duration for task in self.finals)

    def submit_partial(
        self,
        number: int,
        samples: np.ndarray,
        when: str,
        group: int = 0,
        captured_end: float | None = None,
    ) -> None:
        with self.condition:
            if self.failed.is_set() or number <= self.last_final:
                return
            self.latest_partial = RecognitionTask(
                number, samples, when, group, captured_end or time.monotonic()
            )
            self.condition.notify()

    def submit_final(
        self,
        number: int,
        samples: np.ndarray,
        when: str,
        group: int = 0,
        captured_end: float | None = None,
    ) -> None:
        with self.condition:
            if self.failed.is_set():
                return
            if (
                self.pending_seconds() + len(samples) / TARGET_RATE
                > MAX_PENDING_SECONDS
            ):
                raise RuntimeError(
                    "La cola superó 128 s: se detiene la captura para limitar memoria. "
                    "El último segmento no se pudo encolar; probá el modelo tiny."
                )
            if self.latest_partial and self.latest_partial.number == number:
                self.latest_partial = None
            self.last_final = number
            self.finals.append(
                RecognitionTask(
                    number, samples, when, group, captured_end or time.monotonic()
                )
            )
            self.condition.notify()

    def run(self) -> None:
        try:
            from faster_whisper import WhisperModel

            self.signals.status.emit(
                "Cargando modelo de voz. La primera ejecución necesita internet..."
            )
            model = WhisperModel(
                self.model_name, device="cpu", compute_type="int8", cpu_threads=6
            )
            self.signals.status.emit("Modelo listo: transcribiendo audio")

            while True:
                with self.condition:
                    self.condition.wait_for(
                        lambda: (
                            bool(self.finals)
                            or self.latest_partial is not None
                            or self.completed.is_set()
                        ),
                        timeout=0.5,
                    )
                    if self.finals:
                        task = self.finals.popleft()
                        final = True
                    elif self.latest_partial is not None:
                        task = self.latest_partial
                        self.latest_partial = None
                        final = False
                    elif self.completed.is_set():
                        break
                    else:
                        continue
                    self.busy = True

                # Los generadores de faster-whisper requieren iteración para
                # ejecutar realmente la inferencia.
                started = time.monotonic()
                self.inference_started = started
                segments, _ = model.transcribe(
                    task.samples,
                    language="es",
                    beam_size=3 if final else 1,
                    condition_on_previous_text=False,
                    vad_filter=False,
                    without_timestamps=True,
                    temperature=0,
                    initial_prompt=self.initial_prompt,
                )
                content = " ".join(s.text.strip() for s in segments).strip()
                ended = time.monotonic()
                elapsed = ended - started
                with self.condition:
                    self.busy = False
                    self.inference_started = None
                    self.partial_interval = max(0.8, min(4.0, elapsed * 1.25))
                metrics = {
                    "kind": "final" if final else "borrador",
                    "inference": elapsed,
                    "audio": task.duration,
                    "rtf": elapsed / task.duration,
                    "lag": max(0.0, ended - task.captured_end),
                    "pending": self.pending_seconds() + self.capture_pending,
                }
                logger.info("ASR id=%s %s", task.number, metrics)
                if metrics["rtf"] > 1 or metrics["pending"] > 16:
                    logger.warning("Reconocimiento retrasado: %s", metrics)
                self.signals.metrics.emit(metrics)
                if final:
                    if content and self.on_final is not None:
                        try:
                            self.on_final(task.when, content)
                        except OSError as exc:
                            logger.warning(
                                "No se pudo guardar el respaldo incremental: %s", exc
                            )
                    self.signals.final.emit(task.number, content, task.when, task.group)
                else:
                    # Aunque el final ya esté en cola, este resultado sirve para
                    # copiar ahora; el final lo reemplazará por su mismo ID.
                    self.signals.partial.emit(task.number, content, task.group)
        except Exception as exc:
            self.failed.set()
            logger.exception("Falló el reconocimiento")
            self.signals.error.emit(f"Error en el reconocimiento de voz: {exc}")
        finally:
            with self.condition:
                self.busy = False
                self.inference_started = None


class Segmenter(threading.Thread):
    """Agrupa audio por pausas y envía fragmentos a la cola ASR."""

    def __init__(
        self,
        audio_queue: queue.Queue,
        capture_stop: threading.Event,
        completed: threading.Event,
        engine: RecognitionEngine,
        signals,
        energy_threshold: float = 0.007,
        producer_done: threading.Event | None = None,
    ):
        super().__init__(name="Segmentador", daemon=True)
        self.audio_queue = audio_queue
        self.capture_stop = capture_stop
        self.completed = completed
        self.engine = engine
        self.signals = signals
        self.threshold = energy_threshold
        self.producer_done = producer_done or capture_stop
        self.pre_roll: deque[np.ndarray] = deque()
        self.pre_roll_length = 0
        self.utterance: list[np.ndarray] = []
        self.active = False
        self.utterance_size = 0
        self.silence_size = 0
        self.next_partial_at = 0.8
        self.number = 0
        self.started_at = ""
        self.group = 0
        self.continuation = False
        self.captured_end = time.monotonic()
        self.idle_silence = 0

    def _remember_pre_roll(self, samples: np.ndarray) -> None:
        self.pre_roll.append(samples)
        self.pre_roll_length += len(samples)
        while self.pre_roll_length > TARGET_RATE * 0.3 and len(self.pre_roll) > 1:
            old = self.pre_roll.popleft()
            self.pre_roll_length -= len(old)

    def _begin(self) -> None:
        self.number += 1
        if not self.continuation:
            self.group += 1
        self.continuation = False
        self.started_at = datetime.now().strftime("%H:%M:%S")
        self.utterance = list(self.pre_roll)
        self.active = True
        self.utterance_size = sum(len(x) for x in self.utterance)
        self.silence_size = 0
        self.next_partial_at = 0.8

    def _finish(self) -> None:
        if self.utterance_size >= int(TARGET_RATE * 0.45):
            samples = np.concatenate(self.utterance).astype(np.float32)
            self.engine.submit_final(
                self.number, samples, self.started_at, self.group, self.captured_end
            )
        self.utterance = []
        self.active = False
        self.utterance_size = 0
        self.silence_size = 0
        self.next_partial_at = 0.8

    def consume(self, samples: np.ndarray, captured_end: float | None = None) -> None:
        if samples.size == 0:
            return
        self.captured_end = captured_end or time.monotonic()
        level = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        is_loud = level >= self.threshold
        if not self.active and not is_loud:
            self.idle_silence += len(samples)
            if self.idle_silence >= TARGET_RATE * 0.8:
                self.continuation = False
        if not self.active and is_loud:
            self._begin()
            self.idle_silence = 0
        if self.active:
            self.utterance.append(samples)
            self.utterance_size += len(samples)
            if is_loud:
                self.silence_size = 0
            else:
                self.silence_size += len(samples)
            duration = self.utterance_size / TARGET_RATE
            if (
                duration >= self.next_partial_at
                and self.silence_size < TARGET_RATE * 0.65
                and self.engine.wants_partial()
            ):
                self.engine.submit_partial(
                    self.number,
                    np.concatenate(self.utterance).astype(np.float32),
                    self.started_at,
                    self.group,
                    self.captured_end,
                )
                self.next_partial_at = duration + self.engine.partial_interval
            if self.silence_size >= TARGET_RATE * 0.8 or duration >= 16.0:
                self._finish()
                if duration >= 16.0:
                    self.continuation = True
                    self.pre_roll.clear()
                    self.pre_roll_length = 0
                    # No repetir el último paquete al comenzar el segmento siguiente.
                    return
        self._remember_pre_roll(samples)

    def run(self) -> None:
        try:
            while not self.producer_done.is_set() or not self.audio_queue.empty():
                if self.engine.failed.is_set():
                    self.capture_stop.set()
                    break
                try:
                    packet = self.audio_queue.get(timeout=0.15)
                except queue.Empty:
                    continue
                samples, captured_end = packet
                self.engine.capture_pending = max(0.0, time.monotonic() - captured_end)
                self.consume(samples, captured_end)
                if (
                    self.engine.pending_seconds() >= BACKLOG_STOP_SECONDS
                    and not self.capture_stop.is_set()
                ):
                    self.capture_stop.set()
                    message = (
                        "Se acumularon 48 s de audio: se detiene la captura y se procesa "
                        "lo pendiente. Probá tiny para la próxima sesión."
                    )
                    logger.warning(message)
                    self.signals.error.emit(message)
                self.engine.capture_pending = self.utterance_size / TARGET_RATE + max(
                    0.0, time.monotonic() - captured_end
                )
            if self.active:
                self._finish()
        except Exception as exc:
            self.capture_stop.set()
            logger.exception("Falló la segmentación")
            self.signals.error.emit(f"Error al segmentar el audio: {exc}")
        finally:
            self.completed.set()
            with self.engine.condition:
                self.engine.condition.notify_all()
