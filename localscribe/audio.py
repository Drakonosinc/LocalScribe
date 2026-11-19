"""Windows audio capture: WASAPI loopback, microphone, or dual-stream mix."""

from __future__ import annotations

import logging
import math
import queue
import threading
import time

import numpy as np
from scipy.signal import resample_poly

TARGET_RATE = 16_000
CHUNK_FRAMES = 2048
MIX_CHUNK_SAMPLES = 1600

AUDIO_SOURCE_SYSTEM = "system"
AUDIO_SOURCE_MIC = "mic"
AUDIO_SOURCE_BOTH = "both"
VALID_AUDIO_SOURCES = (
    AUDIO_SOURCE_SYSTEM,
    AUDIO_SOURCE_MIC,
    AUDIO_SOURCE_BOTH,
)

logger = logging.getLogger(__name__)


def pcm_to_mono_16k(raw: bytes, channels: int, rate: int) -> np.ndarray:
    """Convert interleaved int16 PCM bytes to mono float32 resampled at 16 kHz."""
    if not raw or channels < 1:
        return np.empty(0, dtype=np.float32)
    pcm = np.frombuffer(raw, dtype=np.int16)
    frame_count = len(pcm) // channels
    if frame_count == 0:
        return np.empty(0, dtype=np.float32)
    pcm = pcm[: frame_count * channels]
    mono = pcm.reshape(-1, channels).astype(np.float32).mean(axis=1)
    mono /= 32768.0
    if rate != TARGET_RATE:
        divisor = math.gcd(rate, TARGET_RATE)
        mono = resample_poly(mono, TARGET_RATE // divisor, rate // divisor).astype(
            np.float32
        )
    return mono


def mix_audio_chunks(
    system_chunk: np.ndarray | None,
    mic_chunk: np.ndarray | None,
) -> np.ndarray:
    """Mix 16 kHz system and microphone audio chunks with clipping protection."""
    has_sys = system_chunk is not None and system_chunk.size > 0
    has_mic = mic_chunk is not None and mic_chunk.size > 0
    if not has_sys and not has_mic:
        return np.empty(0, dtype=np.float32)
    if has_sys and not has_mic:
        return system_chunk.astype(np.float32, copy=False)
    if has_mic and not has_sys:
        return mic_chunk.astype(np.float32, copy=False)

    assert system_chunk is not None and mic_chunk is not None
    length = max(len(system_chunk), len(mic_chunk))
    mixed = np.zeros(length, dtype=np.float32)
    mixed[: len(system_chunk)] += system_chunk.astype(np.float32, copy=False)
    mixed[: len(mic_chunk)] += mic_chunk.astype(np.float32, copy=False)
    return np.clip(mixed, -1.0, 1.0)


class AudioCapture(threading.Thread):
    """Capture Windows audio (WASAPI loopback, microphone, or both).

    Runs on a dedicated thread and never blocks waiting on speech recognition.
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        stop_event: threading.Event,
        signals,
        source: str = AUDIO_SOURCE_SYSTEM,
    ):
        if source not in VALID_AUDIO_SOURCES:
            raise ValueError(f"Unsupported audio source: {source}")
        super().__init__(name="AudioCapture", daemon=True)
        self.audio_queue = audio_queue
        self.stop_event = stop_event
        self.signals = signals
        self.source = source
        self.finished = threading.Event()
        self.dropped_seconds = 0.0
        self._full_warning_sent = False

    def _enqueue(self, mono: np.ndarray, captured_end: float) -> None:
        if mono.size == 0:
            return
        try:
            self.audio_queue.put_nowait((mono, captured_end))
            if (
                self.audio_queue.maxsize > 0
                and self.audio_queue.qsize() < self.audio_queue.maxsize // 2
            ):
                self._full_warning_sent = False
        except queue.Full:
            self.dropped_seconds += len(mono) / TARGET_RATE
            logger.warning(
                "Capture queue full; cumulative dropped audio=%.3fs",
                self.dropped_seconds,
            )
            # Never block capture or flood Qt with repeated warnings.
            if not self._full_warning_sent:
                self.signals.warning.emit(
                    "Processing is falling behind: some audio frames were dropped."
                )
                self._full_warning_sent = True

    @staticmethod
    def _system_device(manager) -> tuple[dict, int, int]:
        device = manager.get_default_wasapi_loopback()
        channels = int(device["maxInputChannels"])
        rate = int(device["defaultSampleRate"])
        if channels < 1 or rate < 8_000:
            raise RuntimeError("Unsupported system output device format")
        return device, channels, rate

    @staticmethod
    def _mic_device(manager) -> tuple[dict, int, int]:
        device = manager.get_default_input_device_info()
        max_channels = int(device["maxInputChannels"])
        rate = int(device["defaultSampleRate"])
        if max_channels < 1 or rate < 8_000:
            raise RuntimeError("Unsupported default microphone format")
        channels = min(max_channels, 2)
        return device, channels, rate

    def _run_single_stream(
        self,
        pyaudio,
        manager,
        device: dict,
        channels: int,
        rate: int,
        label: str,
    ) -> None:
        self.signals.device.emit(label)
        with manager.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=int(device["index"]),
            frames_per_buffer=CHUNK_FRAMES,
        ) as stream:
            while not self.stop_event.is_set():
                raw = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
                captured_end = time.monotonic()
                mono = pcm_to_mono_16k(raw, channels, rate)
                self._enqueue(mono, captured_end)

    def _run_dual_stream(self, pyaudio, manager) -> None:
        sys_dev, sys_ch, sys_rate = self._system_device(manager)
        mic_dev, mic_ch, mic_rate = self._mic_device(manager)
        self.signals.device.emit(
            f"System: {sys_dev['name']} + Microphone: {mic_dev['name']}"
        )

        sys_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        mic_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        reader_errors: list[Exception] = []

        def read_loop(stream, channels: int, rate: int, target_q: queue.Queue) -> None:
            try:
                while not self.stop_event.is_set():
                    raw = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
                    mono = pcm_to_mono_16k(raw, channels, rate)
                    if mono.size == 0:
                        continue
                    try:
                        target_q.put_nowait(mono)
                    except queue.Full:
                        # Drop the oldest chunk so readers never stall.
                        try:
                            target_q.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            target_q.put_nowait(mono)
                        except queue.Full:
                            pass
            except Exception as exc:
                if not self.stop_event.is_set():
                    logger.exception("Dual capture reader thread failed")
                    reader_errors.append(exc)
                    self.stop_event.set()

        with (
            manager.open(
                format=pyaudio.paInt16,
                channels=sys_ch,
                rate=sys_rate,
                input=True,
                input_device_index=int(sys_dev["index"]),
                frames_per_buffer=CHUNK_FRAMES,
            ) as sys_stream,
            manager.open(
                format=pyaudio.paInt16,
                channels=mic_ch,
                rate=mic_rate,
                input=True,
                input_device_index=int(mic_dev["index"]),
                frames_per_buffer=CHUNK_FRAMES,
            ) as mic_stream,
        ):
            sys_thread = threading.Thread(
                target=read_loop,
                args=(sys_stream, sys_ch, sys_rate, sys_q),
                name="SystemReader",
                daemon=True,
            )
            mic_thread = threading.Thread(
                target=read_loop,
                args=(mic_stream, mic_ch, mic_rate, mic_q),
                name="MicReader",
                daemon=True,
            )
            sys_thread.start()
            mic_thread.start()

            sys_buf = np.empty(0, dtype=np.float32)
            mic_buf = np.empty(0, dtype=np.float32)

            def drain(q: queue.Queue[np.ndarray], buf: np.ndarray) -> np.ndarray:
                chunks = [buf] if buf.size else []
                while True:
                    try:
                        chunks.append(q.get_nowait())
                    except queue.Empty:
                        break
                return (
                    np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
                )

            while not self.stop_event.is_set():
                time.sleep(0.05)
                sys_buf = drain(sys_q, sys_buf)
                mic_buf = drain(mic_q, mic_buf)

                while (
                    sys_buf.size >= MIX_CHUNK_SAMPLES
                    or mic_buf.size >= MIX_CHUNK_SAMPLES
                ):
                    take_sys = min(sys_buf.size, MIX_CHUNK_SAMPLES)
                    take_mic = min(mic_buf.size, MIX_CHUNK_SAMPLES)
                    sys_part = sys_buf[:take_sys] if take_sys else None
                    mic_part = mic_buf[:take_mic] if take_mic else None
                    sys_buf = sys_buf[take_sys:]
                    mic_buf = mic_buf[take_mic:]
                    mixed = mix_audio_chunks(sys_part, mic_part)
                    self._enqueue(mixed, time.monotonic())

            sys_thread.join(timeout=1.5)
            mic_thread.join(timeout=1.5)
            sys_buf = drain(sys_q, sys_buf)
            mic_buf = drain(mic_q, mic_buf)
            if sys_buf.size or mic_buf.size:
                self._enqueue(mix_audio_chunks(sys_buf, mic_buf), time.monotonic())

        if reader_errors:
            raise reader_errors[0]

    def run(self) -> None:
        try:
            import pyaudiowpatch as pyaudio

            with pyaudio.PyAudio() as manager:
                if self.source == AUDIO_SOURCE_SYSTEM:
                    dev, channels, rate = self._system_device(manager)
                    self._run_single_stream(
                        pyaudio, manager, dev, channels, rate, f"System: {dev['name']}"
                    )
                elif self.source == AUDIO_SOURCE_MIC:
                    dev, channels, rate = self._mic_device(manager)
                    self._run_single_stream(
                        pyaudio,
                        manager,
                        dev,
                        channels,
                        rate,
                        f"Microphone: {dev['name']}",
                    )
                else:
                    self._run_dual_stream(pyaudio, manager)
        except Exception as exc:
            logger.exception("Audio capture failed (%s)", self.source)
            self.signals.error.emit(f"Could not capture audio: {exc}")
        finally:
            self.stop_event.set()
            self.finished.set()


SystemAudioCapture = AudioCapture
