"""Headless unit tests independent of Windows audio devices and GUI."""

import queue
import threading
import unittest

import numpy as np

from localscribe.audio import TARGET_RATE
from localscribe.pipeline import RecognitionEngine, Segmenter


class DummySignal:
    def emit(self, *args):
        pass


class DummySignals:
    error = DummySignal()
    status = DummySignal()
    partial = DummySignal()
    final = DummySignal()
    metrics = DummySignal()


class FakeEngine:
    def __init__(self):
        self.partials = []
        self.finals = []
        self.partial_interval = 0.8

    def wants_partial(self):
        return True

    def submit_partial(self, number, samples, when, *metadata):
        self.partials.append((number, samples.copy(), when))

    def submit_final(self, number, samples, when, *metadata):
        self.finals.append((number, samples.copy(), when))


class SegmenterTests(unittest.TestCase):
    def test_recognizes_voice_and_silence_as_one_block(self):
        engine = FakeEngine()
        segmenter = Segmenter(
            queue.Queue(), threading.Event(), threading.Event(), engine, DummySignals()
        )
        speech = np.full(1600, 0.04, np.float32)  # 100 ms of audio
        silence = np.zeros(1600, np.float32)
        for _ in range(18):
            segmenter.consume(speech)
        for _ in range(10):
            segmenter.consume(silence)
        self.assertEqual(len(engine.finals), 1)
        self.assertGreaterEqual(len(engine.partials), 1)
        self.assertEqual(engine.finals[0][0], 1)
        self.assertEqual(engine.finals[0][1].dtype, np.float32)

    def test_does_not_transcribe_silence(self):
        engine = FakeEngine()
        segmenter = Segmenter(
            queue.Queue(), threading.Event(), threading.Event(), engine, DummySignals()
        )
        for _ in range(30):
            segmenter.consume(np.zeros(TARGET_RATE // 10, dtype=np.float32))
        self.assertFalse(engine.finals)
        self.assertFalse(engine.partials)

    def test_last_partial_replaces_older_partial(self):
        engine = RecognitionEngine(threading.Event(), DummySignals(), "tiny")
        a = np.ones(1600, dtype=np.float32)
        b = np.ones(3200, dtype=np.float32)
        engine.submit_partial(1, a, "12:00:00")
        engine.submit_partial(1, b, "12:00:00")
        self.assertEqual(len(engine.latest_partial.samples), 3200)
        engine.submit_final(1, b, "12:00:00")
        self.assertIsNone(engine.latest_partial)
        self.assertEqual(len(engine.finals), 1)


if __name__ == "__main__":
    unittest.main()
