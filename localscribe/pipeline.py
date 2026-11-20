"""Pause-based audio segmentation and recognition on a dedicated thread."""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .audio import TARGET_RATE

logger = logging.getLogger(__name__)
MAX_PENDING_SECONDS = 128.0
BACKLOG_STOP_SECONDS = 48.0
MAX_UTTERANCE_SECONDS = 16.0
SMART_CUT_WINDOW_SECONDS = 1.5
CONTEXT_TAIL_CHARS = 200
LANGUAGE_LOCK_PROBABILITY = 0.8
LANGUAGE_LOCK_MIN_AUDIO = 2.0

# Phrases Whisper is known to invent on silence, music, or noise. A segment is
# only discarded when its *entire* text matches one of them.
KNOWN_HALLUCINATIONS = frozenset(
    {
        "gracias por ver el video",
        "gracias por ver el vídeo",
        "gracias por ver",
        "gracias por su atención",
        "suscríbete",
        "suscribete",
        "suscríbete al canal",
        "no olvides suscribirte",
        "thank you for watching",
        "thanks for watching",
        "thank you so much for watching",
        "please subscribe",
        "like and subscribe",
        "subscribe to my channel",
        "you",
        "music",
        "música",
        "musica",
        "aplausos",
        "applause",
        "silence",
        "silencio",
        "obrigado por assistir",
        "merci davoir regardé",
        "untertitel im auftrag des zdf",
        "sottotitoli creati dalla comunità amaraorg",
    }
)
_HALLUCINATION_MARKERS = ("amara.org", "subtítulos realizados por", "subtitulos por")
_NORMALIZE_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize(text: str) -> str:
    return " ".join(_NORMALIZE_RE.sub("", text.lower()).split())


def is_hallucination(text: str) -> bool:
    """True when the whole text is a well-known Whisper hallucination."""
    lowered = text.lower().strip()
    if not lowered:
        return False
    if any(marker in lowered for marker in _HALLUCINATION_MARKERS):
        return True
    normalized = _normalize(lowered)
    if not normalized or set(normalized) <= {"♪", " "}:
        return True
    return normalized in KNOWN_HALLUCINATIONS


def clean_segments(segments: Iterable) -> str:
    """Join Whisper segments, skipping low-confidence or looping output."""
    parts: list[str] = []
    for segment in segments:
        text = str(getattr(segment, "text", "")).strip()
        if not text:
            continue
        no_speech = getattr(segment, "no_speech_prob", 0.0) or 0.0
        avg_logprob = getattr(segment, "avg_logprob", 0.0) or 0.0
        compression = getattr(segment, "compression_ratio", 0.0) or 0.0
        if no_speech > 0.6 and avg_logprob < -1.0:
            continue  # Most likely silence/noise decoded as words.
        if compression > 2.4:
            continue  # Repetition loop ("no no no no no ...").
        if avg_logprob < -1.6:
            continue
        parts.append(text)
    content = " ".join(parts).strip()
    return "" if is_hallucination(content) else content


def default_cpu_threads() -> int:
    return max(2, min(8, (os.cpu_count() or 4) - 2))


@dataclass
class RecognitionTask:
    number: int
    samples: np.ndarray
    when: str
    group: int
    captured_end: float
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    speaker: str = ""

    @property
    def duration(self) -> float:
        return len(self.samples) / TARGET_RATE


class RecognitionEngine(threading.Thread):
    """Runs a single Whisper model while prioritizing confirmed segments.

    Keeps at most one provisional draft request (the latest) so inference
    never accumulates an unbounded backlog of stale partials.
    """

    def __init__(
        self,
        completed: threading.Event,
        signals,
        model_name: str,
        initial_prompt: str = "",
        on_final: Callable[[str, str], None] | None = None,
        language: str | None = None,
        device: str = "cpu",
    ):
        super().__init__(name="Recognizer", daemon=True)
        self.completed = completed
        self.signals = signals
        self.model_name = model_name
        self.condition = threading.Condition()
        self.finals: deque[RecognitionTask] = deque()
        self.latest_partial: RecognitionTask | None = None
        self.initial_prompt = initial_prompt.strip() or None
        self.on_final = on_final
        self.language = None if language in (None, "", "auto") else language
        self.language_locked = self.language is not None
        self.device = device
        self.active_device = "cpu"
        self.failed = threading.Event()
        self.busy = False
        self.partial_interval = 0.8
        self.last_final = 0
        self.capture_pending = 0.0
        self.inference_started: float | None = None
        self.context_tail = ""
        self.use_vad = True
        self.last_speaker = ""

    def wants_partial(self) -> bool:
        with self.condition:
            return not self.busy and not self.finals and not self.failed.is_set()

    def pending_seconds(self) -> float:
        with self.condition:
            # Drafts overlap the active segment and are not double-counted.
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
        start_seconds: float = 0.0,
        end_seconds: float | None = None,
        speaker: str = "",
    ) -> None:
        with self.condition:
            if self.failed.is_set():
                return
            if (
                self.pending_seconds() + len(samples) / TARGET_RATE
                > MAX_PENDING_SECONDS
            ):
                raise RuntimeError(
                    "Queue exceeded 128s: stopping capture to bound memory usage. "
                    "The last segment could not be queued; try a faster model."
                )
            if self.latest_partial and self.latest_partial.number == number:
                self.latest_partial = None
            self.last_final = number
            duration = len(samples) / TARGET_RATE
            self.finals.append(
                RecognitionTask(
                    number,
                    samples,
                    when,
                    group,
                    captured_end or time.monotonic(),
                    start_seconds,
                    end_seconds
                    if end_seconds is not None
                    else start_seconds + duration,
                    speaker=speaker,
                )
            )
            self.condition.notify()

    # --- Model loading -------------------------------------------------

    def _resolve_device(self) -> str:
        if self.device in ("cpu", "cuda"):
            return self.device
        try:
            import ctranslate2

            return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:  # noqa: BLE001 - any driver issue means "no GPU"
            return "cpu"

    def _load_model(self):
        from faster_whisper import WhisperModel

        device = self._resolve_device()
        self.signals.status.emit(
            f"Loading '{self.model_name}' on {device.upper()}… "
            "(first use downloads the model)"
        )
        if device == "cuda":
            try:
                model = WhisperModel(
                    self.model_name, device="cuda", compute_type="float16"
                )
                # CUDA libraries load lazily: run a tiny warm-up to fail fast.
                segments, _ = model.transcribe(
                    np.zeros(TARGET_RATE // 2, np.float32), without_timestamps=True
                )
                list(segments)
                self.active_device = "cuda"
                return model
            except Exception as exc:  # noqa: BLE001 - CUDA libs fail in many ways
                logger.warning("CUDA unavailable, falling back to CPU: %s", exc)
                self.signals.status.emit("GPU unavailable; using CPU instead…")
        model = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=default_cpu_threads(),
        )
        self.active_device = "cpu"
        return model

    # --- Inference -----------------------------------------------------

    def _prompt(self) -> str | None:
        parts = [p for p in (self.initial_prompt, self.context_tail) if p]
        return " ".join(parts) if parts else None

    def _transcribe(self, model, task: RecognitionTask, final: bool):
        kwargs = dict(
            language=self.language,
            beam_size=3 if final else 1,
            condition_on_previous_text=False,
            without_timestamps=True,
            temperature=0,
            initial_prompt=self._prompt() if final else self.initial_prompt,
            vad_filter=bool(final and self.use_vad),
        )
        if kwargs["vad_filter"]:
            kwargs["vad_parameters"] = {
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 300,
            }
        try:
            segments, info = model.transcribe(task.samples, **kwargs)
            content = clean_segments(segments)
        except Exception as exc:
            if not kwargs["vad_filter"]:
                raise
            # Silero VAD needs onnxruntime; degrade gracefully without it.
            logger.warning("VAD filter disabled after failure: %s", exc)
            self.use_vad = False
            kwargs["vad_filter"] = False
            kwargs.pop("vad_parameters", None)
            segments, info = model.transcribe(task.samples, **kwargs)
            content = clean_segments(segments)
        return content, info

    def _maybe_lock_language(self, info, task: RecognitionTask, content: str) -> None:
        if self.language_locked or info is None or not content:
            return
        detected = getattr(info, "language", None)
        probability = getattr(info, "language_probability", 0.0) or 0.0
        if (
            isinstance(detected, str)
            and probability >= LANGUAGE_LOCK_PROBABILITY
            and task.duration >= LANGUAGE_LOCK_MIN_AUDIO
        ):
            self.language = detected
            self.language_locked = True
            self.signals.status.emit(
                f"Language detected: {detected} ({probability:.0%}) · locked"
            )

    def run(self) -> None:
        try:
            model = self._load_model()
            self.signals.status.emit(
                f"Listening · {self.model_name} on {self.active_device.upper()}"
            )

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

                started = time.monotonic()
                self.inference_started = started
                content, info = self._transcribe(model, task, final)
                ended = time.monotonic()
                elapsed = ended - started
                with self.condition:
                    self.busy = False
                    self.inference_started = None
                    self.partial_interval = max(0.8, min(4.0, elapsed * 1.25))
                metrics = {
                    "kind": "final" if final else "draft",
                    "inference": elapsed,
                    "audio": task.duration,
                    "rtf": elapsed / task.duration,
                    "lag": max(0.0, ended - task.captured_end),
                    "pending": self.pending_seconds() + self.capture_pending,
                    "device": self.active_device,
                }
                logger.info("ASR id=%s %s", task.number, metrics)
                if metrics["rtf"] > 1 or metrics["pending"] > 16:
                    logger.warning("Recognition falling behind: %s", metrics)
                self.signals.metrics.emit(metrics)
                if final:
                    self._maybe_lock_language(info, task, content)
                    if content:
                        self.context_tail = content[-CONTEXT_TAIL_CHARS:]
                        if self.on_final is not None:
                            try:
                                self.on_final(task.when, content)
                            except OSError as exc:
                                logger.warning(
                                    "Could not write incremental session backup: %s",
                                    exc,
                                )
                    self.last_speaker = task.speaker
                    self.signals.final.emit(
                        task.number,
                        content,
                        task.when,
                        task.group,
                        task.start_seconds,
                        task.end_seconds,
                    )
                else:
                    # Even if the final is already queued, emit the draft so users
                    # can copy immediately; the final will replace it by ID.
                    self.signals.partial.emit(task.number, content, task.group)
        except Exception as exc:
            self.failed.set()
            logger.exception("Speech recognition failed")
            self.signals.error.emit(f"Speech recognition error: {exc}")
        finally:
            with self.condition:
                self.busy = False
                self.inference_started = None


class Segmenter(threading.Thread):
    """Groups incoming audio by silence pauses and submits segments to ASR.

    Tracks an absolute sample counter so every segment carries its real
    position in the session timeline (used for accurate SRT export), adapts
    the speech threshold to the background noise floor, and splits overly
    long utterances at the quietest moment instead of mid-word.
    """

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
        super().__init__(name="Segmenter", daemon=True)
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
        self.total_samples = 0
        self.utterance_start_sample = 0
        self.noise_floor = 0.0
        self.level = 0.0  # Latest RMS, polled by the UI level meter.
        self._speaker_votes: dict[str, int] = {}

    @property
    def effective_threshold(self) -> float:
        return max(self.threshold, min(0.05, self.noise_floor * 3.0))

    def _remember_pre_roll(self, samples: np.ndarray) -> None:
        self.pre_roll.append(samples)
        self.pre_roll_length += len(samples)
        while self.pre_roll_length > TARGET_RATE * 0.3 and len(self.pre_roll) > 1:
            old = self.pre_roll.popleft()
            self.pre_roll_length -= len(old)

    def _open(self, chunks: list[np.ndarray], start_sample: int) -> None:
        self.number += 1
        if not self.continuation:
            self.group += 1
        self.continuation = False
        self.started_at = datetime.now().strftime("%H:%M:%S")
        self.utterance = list(chunks)
        self.active = True
        self.utterance_size = sum(len(x) for x in self.utterance)
        self.utterance_start_sample = max(0, start_sample)
        self.silence_size = 0
        self.next_partial_at = 0.8

    def _begin(self, packet_start: int | None = None) -> None:
        start = (
            packet_start if packet_start is not None else self.total_samples
        ) - self.pre_roll_length
        self._open(list(self.pre_roll), start)

    def _finish(self) -> None:
        if self.utterance_size >= int(TARGET_RATE * 0.45):
            samples = np.concatenate(self.utterance).astype(np.float32)
            start = self.utterance_start_sample / TARGET_RATE
            voiced = max(0, self.utterance_size - self.silence_size)
            speaker = (
                max(self._speaker_votes, key=self._speaker_votes.get)
                if self._speaker_votes
                else ""
            )
            try:
                self.engine.submit_final(
                    self.number,
                    samples,
                    self.started_at,
                    self.group,
                    self.captured_end,
                    start,
                    start + voiced / TARGET_RATE,
                    speaker=speaker,
                )
            except TypeError:
                self.engine.submit_final(
                    self.number,
                    samples,
                    self.started_at,
                    self.group,
                    self.captured_end,
                    start,
                    start + voiced / TARGET_RATE,
                )
        self.utterance = []
        self._speaker_votes.clear()
        self.active = False
        self.utterance_size = 0
        self.silence_size = 0
        self.next_partial_at = 0.8

    @staticmethod
    def best_cut(audio: np.ndarray) -> int:
        """Index where to split a long utterance: the quietest 100 ms frame."""
        frame = TARGET_RATE // 10
        length = len(audio)
        frames = int(min(SMART_CUT_WINDOW_SECONDS * 10, length // (2 * frame)))
        if frames < 2:
            return length
        energies = []
        for k in range(frames):
            end = length - k * frame
            chunk = audio[end - frame : end]
            energies.append(float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64)))))
        quietest = int(np.argmin(energies))
        # Prefer the boundary unless a clearly quieter gap exists.
        if quietest == 0 or energies[quietest] >= energies[0] * 0.8:
            return length
        return length - quietest * frame - frame // 2

    def _split_long_utterance(self) -> None:
        audio = np.concatenate(self.utterance)
        cut = self.best_cut(audio)
        head, tail = audio[:cut], audio[cut:]
        tail_start = self.utterance_start_sample + len(head)
        self.utterance = [head]
        self.utterance_size = len(head)
        self.silence_size = 0
        self._finish()
        self.continuation = True
        self.pre_roll.clear()
        self.pre_roll_length = 0
        if tail.size:
            self._open([tail], tail_start)

    def consume(
        self,
        samples: np.ndarray,
        captured_end: float | None = None,
        speaker: str = "",
    ) -> None:
        if samples.size == 0:
            return
        packet_start = self.total_samples
        self.total_samples += len(samples)
        self.captured_end = captured_end or time.monotonic()
        level = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        self.level = level
        is_loud = level >= self.effective_threshold
        if not self.active and not is_loud:
            # Slowly track the background noise so fans/hum stop triggering.
            self.noise_floor = (
                level
                if self.noise_floor == 0.0
                else self.noise_floor * 0.95 + level * 0.05
            )
            self.idle_silence += len(samples)
            if self.idle_silence >= TARGET_RATE * 0.8:
                self.continuation = False
        if not self.active and is_loud:
            self._begin(packet_start)
            self.idle_silence = 0
        if self.active:
            self.utterance.append(samples)
            self.utterance_size += len(samples)
            if is_loud:
                self.silence_size = 0
                if speaker:
                    self._speaker_votes[speaker] = (
                        self._speaker_votes.get(speaker, 0) + 1
                    )
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
            if self.silence_size >= TARGET_RATE * 0.8:
                self._finish()
            elif duration >= MAX_UTTERANCE_SECONDS:
                self._split_long_utterance()
                # Do not repeat the boundary packet in the next segment.
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
                if len(packet) == 3:
                    samples, captured_end, speaker = packet
                else:
                    samples, captured_end = packet
                    speaker = ""
                self.engine.capture_pending = max(0.0, time.monotonic() - captured_end)
                self.consume(samples, captured_end, speaker=speaker)
                if (
                    self.engine.pending_seconds() >= BACKLOG_STOP_SECONDS
                    and not self.capture_stop.is_set()
                ):
                    self.capture_stop.set()
                    message = (
                        "48s of audio backlog accumulated: stopping capture and "
                        "processing queued audio. Try a faster model next session."
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
            logger.exception("Audio segmentation failed")
            self.signals.error.emit(f"Audio segmentation error: {exc}")
        finally:
            self.level = 0.0
            self.completed.set()
            with self.engine.condition:
                self.engine.condition.notify_all()


SUPPORTED_MEDIA_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
    ".mkv",
    ".ogg",
    ".flac",
    ".webm",
    ".aac",
    ".wma",
)


class FileTranscriptionJob(threading.Thread):
    """Transcribes an audio/video file locally and emits confirmed blocks."""

    def __init__(
        self,
        media_path: Path,
        signals,
        model_name: str,
        initial_prompt: str = "",
        language: str | None = None,
        device: str = "auto",
        on_final: Callable[[str, str], None] | None = None,
    ):
        super().__init__(name="FileTranscriber", daemon=True)
        self.media_path = Path(media_path)
        self.signals = signals
        self.model_name = model_name
        self.initial_prompt = initial_prompt.strip() or None
        self.language = None if language in (None, "", "auto") else language
        self.device = device
        self.on_final = on_final
        self.cancel_event = threading.Event()
        self.active_device = "cpu"

    def stop(self) -> None:
        self.cancel_event.set()

    def _load_model(self):
        helper = RecognitionEngine(
            threading.Event(),
            self.signals,
            self.model_name,
            self.initial_prompt or "",
            language=self.language,
            device=self.device,
        )
        model = helper._load_model()
        self.active_device = helper.active_device
        return model

    def run(self) -> None:
        try:
            if not self.media_path.exists():
                raise FileNotFoundError(f"File not found: {self.media_path}")
            model = self._load_model()
            self.signals.status.emit(
                f"Transcribing file '{self.media_path.name}' on {self.active_device.upper()}…"
            )
            started = time.monotonic()
            kwargs = dict(
                language=self.language,
                beam_size=3,
                condition_on_previous_text=True,
                without_timestamps=False,
                temperature=0,
                initial_prompt=self.initial_prompt,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 300,
                },
            )
            try:
                segments, _info = model.transcribe(str(self.media_path), **kwargs)
                self._consume_segments(segments, started)
            except Exception as exc:
                if not kwargs.get("vad_filter"):
                    raise
                logger.warning("File VAD filter failed, retrying without VAD: %s", exc)
                kwargs["vad_filter"] = False
                kwargs.pop("vad_parameters", None)
                segments, _info = model.transcribe(str(self.media_path), **kwargs)
                self._consume_segments(segments, started)
        except Exception as exc:
            logger.exception("File transcription failed")
            self.signals.error.emit(f"File transcription error: {exc}")
        finally:
            self.signals.finished.emit()

    def _consume_segments(self, segments: Iterable, started: float) -> None:
        number = 0
        for seg in segments:
            if self.cancel_event.is_set():
                break
            text = clean_segments([seg])
            if not text:
                continue
            number += 1
            start_s = float(getattr(seg, "start", 0.0) or 0.0)
            end_s = float(getattr(seg, "end", start_s + 1.0) or (start_s + 1.0))
            mins, secs = divmod(int(max(0.0, start_s)), 60)
            hrs, mins = divmod(mins, 60)
            when = f"{hrs:02d}:{mins:02d}:{secs:02d}"
            if self.on_final is not None:
                try:
                    self.on_final(when, text)
                except OSError as exc:
                    logger.warning("Could not write file session backup: %s", exc)
            self.signals.final.emit(number, text, when, 1, start_s, end_s)
        elapsed = max(0.001, time.monotonic() - started)
        self.signals.metrics.emit(
            {
                "kind": "file",
                "inference": elapsed,
                "audio": max(0.1, elapsed),
                "rtf": 0.0,
                "lag": 0.0,
                "pending": 0.0,
                "device": self.active_device,
            }
        )
