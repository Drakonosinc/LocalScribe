"""LocalScribe main window: stable history, smart scroll, and multi-format export."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from time import monotonic

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .audio import AUDIO_SOURCE_BOTH, AUDIO_SOURCE_MIC, AUDIO_SOURCE_SYSTEM
from .config import DEFAULT_CONFIG_PATH, AppConfig, load_config, save_config
from .session import Session, Signals
from .transcript import RecentTranscript, TranscriptEntry, export_transcript

logger = logging.getLogger(__name__)

AUDIO_SOURCE_HINTS = {
    AUDIO_SOURCE_SYSTEM: "Audio: default system output (WASAPI)",
    AUDIO_SOURCE_MIC: "Audio: default microphone",
    AUDIO_SOURCE_BOTH: "Audio: system (WASAPI) + default microphone",
}

EXPORT_FILTERS = "Plain text (*.txt);;Markdown (*.md);;SRT Subtitles (*.srt)"


class SafeTranscript(QPlainTextEdit):
    """Never jumps to the bottom while reading or selecting earlier text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mouse_selecting = False
        self.setReadOnly(True)
        self.setCenterOnScroll(False)
        self.setUndoRedoEnabled(False)
        self.setPlaceholderText("Confirmed transcript history will appear here...")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_selecting = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_selecting = False

    def add_entry(self, text: str) -> None:
        bar = self.verticalScrollBar()
        old_position = bar.value()
        old_maximum = bar.maximum()
        old_cursor = self.textCursor()
        follow = (
            old_position >= old_maximum - 2
            and not old_cursor.hasSelection()
            and not self._mouse_selecting
        )

        # Avoid setPlainText: it would overwrite the document and wipe active
        # selections, cursor anchors, and scroll position.
        anchor, position = old_cursor.anchor(), old_cursor.position()
        writer = QTextCursor(self.document())
        writer.movePosition(QTextCursor.MoveOperation.End)
        if not self.document().isEmpty():
            writer.insertBlock()
        writer.insertText(text)
        old_cursor.setPosition(anchor)
        old_cursor.setPosition(position, QTextCursor.MoveMode.KeepAnchor)
        self.setTextCursor(old_cursor)
        if follow:
            bar.setValue(bar.maximum())
        else:
            bar.setValue(old_position)

    def go_to_latest(self) -> None:
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class ProtectedLive(QPlainTextEdit):
    """Defers provisional draft updates while the user selects text to copy."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self._mouse_selecting = False
        self._pending_text: str | None = None
        self.latest_text = ""
        self.selectionChanged.connect(self._flush_pending)

    def update_text(self, text: str) -> None:
        self.latest_text = text
        if self._mouse_selecting or self.textCursor().hasSelection():
            self._pending_text = text
            return
        self._pending_text = None
        if text != self.toPlainText():
            bar = self.verticalScrollBar()
            position = bar.value()
            follow = position == bar.maximum()
            self.setPlainText(text)
            bar.setValue(bar.maximum() if follow else position)

    def _flush_pending(self) -> None:
        if (
            self._pending_text is not None
            and not self._mouse_selecting
            and not self.textCursor().hasSelection()
        ):
            text = self._pending_text
            self._pending_text = None
            self.update_text(text)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_selecting = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_selecting = False
            self._flush_pending()


class MainWindow(QMainWindow):
    def __init__(
        self,
        config_path: Path | None = None,
        sessions_dir: Path | None = None,
    ):
        super().__init__()
        self.setWindowTitle("LocalScribe — Live Transcriber")
        self.resize(780, 640)
        self.setMinimumSize(540, 420)
        self.config_path = (
            config_path if config_path is not None else DEFAULT_CONFIG_PATH
        )
        self.sessions_dir = sessions_dir
        self.session: Session | None = None
        self.signals: Signals | None = None
        self.last_completed_id = 0
        self.latest_partial_id = 0
        self.last_block = ""
        self.is_recording = False
        self.is_stopping = False
        self._closing = False
        self._last_error = None
        self._loading_config = False
        self.recent = RecentTranscript()
        self.entries: list[TranscriptEntry] = []
        self._timeline_origin: float | None = None
        self._timeline_end: float = 0.0
        self.latest_metrics = None

        self._build_ui()
        self._apply_style()
        self._load_persisted_config()
        self._connect_config_persistence()

        self.metrics_timer = QTimer(self)
        self.metrics_timer.setInterval(1000)
        self.metrics_timer.timeout.connect(self._refresh_metrics)
        self.metrics_timer.start()
        self._set_status("Ready. Select an audio source and click Start.")

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(13, 13, 13, 13)
        layout.setSpacing(9)

        header = QHBoxLayout()
        title = QLabel("LOCALSCRIBE")
        title.setObjectName("title")
        header.addWidget(title, stretch=1)
        self.pin_button = QPushButton("📌 Always on top")
        self.pin_button.setCheckable(True)
        self.pin_button.setToolTip("Keep this window above other desktop windows")
        self.pin_button.toggled.connect(self._toggle_topmost)
        header.addWidget(self.pin_button)
        layout.addLayout(header)

        controls = QHBoxLayout()
        self.start_button = QPushButton("▶ Start")
        self.start_button.setObjectName("start")
        self.start_button.clicked.connect(self.start_or_stop)
        controls.addWidget(self.start_button)

        self.audio_source = QComboBox()
        self.audio_source.addItem("System Audio (WASAPI)", AUDIO_SOURCE_SYSTEM)
        self.audio_source.addItem("Microphone", AUDIO_SOURCE_MIC)
        self.audio_source.addItem("Both (System + Microphone)", AUDIO_SOURCE_BOTH)
        self.audio_source.setToolTip("Choose the audio input source for this session")
        self.audio_source.currentIndexChanged.connect(self._on_audio_source_changed)
        controls.addWidget(self.audio_source)

        self.model = QComboBox()
        self.model.addItem("Fast (tiny)", "tiny")
        self.model.addItem("Balanced (base)", "base")
        self.model.addItem("Accurate, slower (small)", "small")
        self.model.setCurrentIndex(1)
        self.model.setToolTip("Downloaded once and reused offline")
        controls.addWidget(self.model)

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.001, 0.060)
        self.threshold.setDecimals(3)
        self.threshold.setSingleStep(0.002)
        self.threshold.setValue(0.007)
        self.threshold.setPrefix("Threshold ")
        self.threshold.setToolTip(
            "Voice activity sensitivity; lower if speech is missed, "
            "raise if background noise triggers segments."
        )
        controls.addWidget(self.threshold)
        layout.addLayout(controls)

        self.vocabulary = QLineEdit()
        self.vocabulary.setMaxLength(600)
        self.vocabulary.setPlaceholderText(
            "Optional vocabulary: Docker, Kubernetes, FastAPI, PyTorch..."
        )
        self.vocabulary.setToolTip(
            "Context hint for Whisper (initial_prompt); optional."
        )
        layout.addWidget(self.vocabulary)

        self.device_label = QLabel(AUDIO_SOURCE_HINTS[AUDIO_SOURCE_SYSTEM])
        self.device_label.setObjectName("hint")
        layout.addWidget(self.device_label)

        layout.addWidget(QLabel("Confirmed History (selectable and persistent)"))
        self.transcript = SafeTranscript()
        self.transcript.setFont(QFont("Segoe UI", 11))
        layout.addWidget(self.transcript, stretch=1)

        nav = QHBoxLayout()
        self.jump_button = QPushButton("↓ Jump to latest")
        self.jump_button.clicked.connect(self.transcript.go_to_latest)
        nav.addWidget(self.jump_button)
        nav.addStretch()
        self.copy_selected = QPushButton("Copy selection")
        self.copy_selected.clicked.connect(self._copy_selection)
        nav.addWidget(self.copy_selected)
        self.copy_latest = QPushButton("Copy last block")
        self.copy_latest.clicked.connect(self._copy_last)
        nav.addWidget(self.copy_latest)
        layout.addLayout(nav)

        layout.addWidget(QLabel("Live · provisional draft (may be revised)"))
        self.live = ProtectedLive()
        self.live.setReadOnly(True)
        self.live.setMaximumHeight(95)
        self.live.setMinimumHeight(64)
        self.live.setPlaceholderText("Live speech hypothesis will appear here...")
        self.live.setFont(QFont("Segoe UI", 10))
        self.live.setObjectName("live")
        layout.addWidget(self.live)

        copy_live = QHBoxLayout()
        self.copy_draft = QPushButton("Copy current draft")
        self.copy_draft.clicked.connect(self._copy_draft)
        copy_live.addWidget(self.copy_draft)
        self.copy_recent = QPushButton("Copy recent utterance")
        self.copy_recent.clicked.connect(self._copy_recent)
        copy_live.addWidget(self.copy_recent)
        layout.addLayout(copy_live)

        self.diagnostics_button = QPushButton("Show diagnostics")
        self.diagnostics_button.setCheckable(True)
        layout.addWidget(self.diagnostics_button)
        self.diagnostics = QLabel("Waiting for speech recognition metrics…")
        self.diagnostics.setWordWrap(True)
        self.diagnostics.hide()
        self.diagnostics_button.toggled.connect(self.diagnostics.setVisible)
        layout.addWidget(self.diagnostics)

        bottom = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search history...")
        self.search.returnPressed.connect(self._find_next)
        bottom.addWidget(self.search, stretch=1)
        search_button = QPushButton("Search")
        search_button.clicked.connect(self._find_next)
        bottom.addWidget(search_button)
        self.export_button = QPushButton("Export...")
        self.export_button.setToolTip("Export confirmed history as .txt, .md, or .srt")
        self.export_button.clicked.connect(self._export)
        bottom.addWidget(self.export_button)
        layout.addLayout(bottom)

        self.status = QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        QShortcut(QKeySequence("Ctrl+F"), self, activated=self.search.setFocus)
        QShortcut(QKeySequence("Ctrl+L"), self, activated=self.transcript.go_to_latest)

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QWidget { background: #121821; color: #e4edf7; font: 10pt 'Segoe UI'; }
            QLabel#title { font-weight: 700; font-size: 13pt; color: #f5f9ff; }
            QLabel#hint, QLabel#status { color: #aabbd0; font-size: 9pt; }
            QPlainTextEdit, QLineEdit, QComboBox, QDoubleSpinBox {
                background: #1c2633; color: #edf4fb;
                border: 1px solid #33445a; border-radius: 6px; padding: 6px;
                selection-background-color: #3c73bc;
            }
            QPlainTextEdit#live { background: #192b31; border-color: #315e61; }
            QPushButton { background: #28384b; color: #e7f1fc;
                border: 1px solid #415670; border-radius: 6px; padding: 7px 10px; }
            QPushButton:hover { background: #354d67; }
            QPushButton:checked { background: #156f65; border-color: #32b5a2; }
            QPushButton#start { background: #176f5c; border-color: #1e9f80; font-weight: 600; }
            QPushButton#start:hover { background: #218670; }
            QPushButton:disabled { color: #7d8d9e; background: #202c3a; }
        """)

    def _load_persisted_config(self) -> None:
        self._loading_config = True
        try:
            config = load_config(self.config_path)
            model_idx = self.model.findData(config.model)
            if model_idx >= 0:
                self.model.setCurrentIndex(model_idx)
            self.threshold.setValue(config.threshold)
            source_idx = self.audio_source.findData(config.audio_source)
            if source_idx >= 0:
                self.audio_source.setCurrentIndex(source_idx)
            self.vocabulary.setText(config.vocabulary)
            self.pin_button.setChecked(config.always_on_top)
            self._on_audio_source_changed()
        finally:
            self._loading_config = False

    def _connect_config_persistence(self) -> None:
        self.model.currentIndexChanged.connect(self._save_current_config)
        self.threshold.valueChanged.connect(self._save_current_config)
        self.audio_source.currentIndexChanged.connect(self._save_current_config)
        self.pin_button.toggled.connect(self._save_current_config)
        self.vocabulary.textChanged.connect(self._save_current_config)

    def current_config(self) -> AppConfig:
        return AppConfig(
            model=str(self.model.currentData()),
            threshold=float(self.threshold.value()),
            audio_source=str(self.audio_source.currentData()),
            always_on_top=bool(self.pin_button.isChecked()),
            vocabulary=self.vocabulary.text(),
        )

    def _save_current_config(self, *_args) -> None:
        if self._loading_config:
            return
        try:
            save_config(self.current_config(), self.config_path)
        except OSError as exc:
            logger.warning("Could not save config.json: %s", exc)

    def _on_audio_source_changed(self) -> None:
        if self.is_recording:
            return
        source = str(self.audio_source.currentData())
        self.device_label.setText(
            AUDIO_SOURCE_HINTS.get(source, AUDIO_SOURCE_HINTS[AUDIO_SOURCE_SYSTEM])
        )

    def _set_status(self, value: str) -> None:
        self.status.setText(value)

    def _toggle_topmost(self, enabled: bool) -> None:
        was_visible = self.isVisible()
        geometry = self.geometry()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
        if was_visible:
            self.show()
            self.setGeometry(geometry)
        self.pin_button.setText(
            "📌 Always on top: ON" if enabled else "📌 Always on top"
        )

    def start_or_stop(self) -> None:
        if self.is_recording:
            self._stop()
        elif not self.is_stopping:
            self._start()

    def _start(self) -> None:
        self.signals = Signals(self)
        self.signals.status.connect(self._set_status)
        self.signals.device.connect(
            lambda name: self.device_label.setText("Audio: " + name)
        )
        self.signals.partial.connect(self._on_partial)
        self.signals.final.connect(self._on_final)
        self.signals.warning.connect(self._set_status)
        self.signals.error.connect(self._on_error)
        self.signals.finished.connect(self._on_finished)
        self.signals.metrics.connect(self._on_metrics)

        # Each session resets segment IDs while keeping confirmed history intact.
        self.last_completed_id = 0
        self.latest_partial_id = 0
        self.recent = RecentTranscript()
        self.latest_metrics = None
        if self._timeline_origin is None:
            self._timeline_origin = monotonic()

        self.session = Session(
            self.signals,
            str(self.model.currentData()),
            self.threshold.value(),
            self.vocabulary.text(),
            audio_source=str(self.audio_source.currentData()),
            sessions_dir=self.sessions_dir,
        )
        self.audio_source.setEnabled(False)
        self.model.setEnabled(False)
        self.threshold.setEnabled(False)
        self.vocabulary.setEnabled(False)
        self.is_recording = True
        self.start_button.setText("■ Stop")
        self._set_status("Starting audio capture...")
        self.live.update_text("")
        self._last_error = None
        self.session.start()

    def _stop(self) -> None:
        if not self.session or self.is_stopping:
            return
        self.is_stopping = True
        self.is_recording = False
        self.start_button.setEnabled(False)
        self.start_button.setText("Stopping...")
        self._set_status("Stopping capture and processing remaining segments...")
        self.session.stop()

    def _on_partial(self, number: int, content: str, group: int = 0) -> None:
        if not self.recent.partial(number, content, group):
            return
        self.latest_partial_id = number
        # Drafts live in a separate panel so live revisions never break history selection.
        self.live.update_text(content)

    def _on_final(self, number: int, content: str, when: str, group: int = 0) -> None:
        if not self.recent.final(number, content, group):
            return
        self.last_completed_id = number
        if content:
            duration = None
            if (
                isinstance(self.latest_metrics, dict)
                and self.latest_metrics.get("kind") == "final"
                and isinstance(self.latest_metrics.get("audio"), (int, float))
            ):
                duration = float(self.latest_metrics["audio"])
            if duration is None or duration <= 0:
                words = max(1, len(content.split()))
                duration = max(2.0, min(16.0, words * 0.45))

            if self._timeline_origin is not None:
                now_elapsed = max(0.0, monotonic() - self._timeline_origin)
                start = max(self._timeline_end, max(0.0, now_elapsed - duration))
            else:
                start = self._timeline_end
            end = start + duration
            self._timeline_end = end

            self.entries.append(
                TranscriptEntry(
                    timestamp=when,
                    text=content,
                    start_seconds=start,
                    end_seconds=end,
                )
            )
            block = f"[{when}] {content}"
            self.last_block = block
            self.transcript.add_entry(block)
        if number >= self.latest_partial_id:
            self.live.update_text("")

    def _on_error(self, message: str) -> None:
        self._last_error = message
        if self.is_recording:
            self._stop()
        self._set_status(message)

    def _on_finished(self) -> None:
        self.is_recording = False
        self.is_stopping = False
        self.start_button.setEnabled(True)
        self.start_button.setText("▶ Start")
        self.audio_source.setEnabled(True)
        self.model.setEnabled(True)
        self.threshold.setEnabled(True)
        self.vocabulary.setEnabled(True)
        self._on_audio_source_changed()
        if not self._closing:
            self._set_status(
                self._last_error
                or "Stopped. Confirmed history remains available to copy or export."
            )
        self.session = None

    def _copy_draft(self) -> None:
        if self.recent.draft:
            QApplication.clipboard().setText(self.recent.draft)
            self._set_status("Draft copied; wording may still be revised.")
        else:
            self._set_status("No live draft available yet.")

    def _copy_recent(self) -> None:
        text = self.recent.recent()
        if text:
            QApplication.clipboard().setText(text)
            self._set_status("Recent utterance copied (includes active draft if any).")

    def _on_metrics(self, metrics: dict) -> None:
        self.latest_metrics = metrics

    def _refresh_metrics(self) -> None:
        if not self.latest_metrics:
            if self.session:
                pending = (
                    self.session.engine.pending_seconds()
                    + self.session.engine.capture_pending
                )
                self.diagnostics.setText(
                    f"Waiting for first inference · Pending ≈{pending:.1f}s"
                )
            return
        m = self.latest_metrics
        pending = m["pending"]
        if self.session:
            pending = (
                self.session.engine.pending_seconds()
                + self.session.engine.capture_pending
            )
        warning = " · BACKLOG: try tiny" if m["rtf"] > 1 or pending > 16 else ""
        running = ""
        started = self.session.engine.inference_started if self.session else None
        if started is not None:
            running = f" · Active inference: {monotonic() - started:.1f}s"
        self.diagnostics.setText(
            f"{m['kind']}: inference {m['inference']:.2f}s / audio {m['audio']:.2f}s "
            f"· RTF {m['rtf']:.2f} (measured)\n"
            f"Latest audio age at emit: {m['lag']:.2f}s · "
            f"Pending ≈{pending:.1f}s{running}{warning}"
        )

    def _copy_selection(self) -> None:
        content = self.transcript.textCursor().selectedText().replace("\u2029", "\n")
        if content:
            QApplication.clipboard().setText(content)
            self._set_status("Selection copied.")
        else:
            self._set_status("Select some text first to copy.")

    def _copy_last(self) -> None:
        if self.last_block:
            QApplication.clipboard().setText(self.last_block)
            self._set_status("Last confirmed block copied.")
        else:
            self._set_status("No confirmed block available yet.")

    def _find_next(self) -> None:
        query = self.search.text().strip()
        if not query:
            return
        if not self.transcript.find(query):
            cursor = self.transcript.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            self.transcript.setTextCursor(cursor)
            if not self.transcript.find(query):
                self._set_status(f"Not found: {query}")
                return
        self._set_status("Match found; scrolling remains under your control.")

    @staticmethod
    def _resolve_export_target(filename: str, selected_filter: str) -> tuple[Path, str]:
        path = Path(filename)
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".srt"}:
            return path, suffix.lstrip(".")
        if "*.md" in selected_filter:
            return path.with_suffix(".md"), "md"
        if "*.srt" in selected_filter:
            return path.with_suffix(".srt"), "srt"
        return path.with_suffix(".txt"), "txt"

    def _export(self) -> None:
        if not self.entries:
            self._set_status("No confirmed content to export.")
            return
        proposed = f"localscribe_{datetime.now():%Y%m%d_%H%M%S}.txt"
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Transcript",
            str(Path.home() / proposed),
            EXPORT_FILTERS,
        )
        if filename:
            target, fmt = self._resolve_export_target(filename, selected_filter)
            try:
                export_transcript(self.entries, target, fmt=fmt)
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "Export Error", str(exc))
                return
            self._set_status(f"Transcript exported successfully (.{fmt}).")

    def closeEvent(self, event) -> None:
        self._save_current_config()
        if self.session is not None:
            # Wait for worker threads to drain queued audio before closing.
            self._closing = True
            self._stop()
            self._set_status("Finishing pending audio. Close again once stopped.")
            event.ignore()
            return
        event.accept()
