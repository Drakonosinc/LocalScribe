"""LocalScribe main window: modern layout, live captions, smart history, export."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from time import monotonic

from PySide6.QtCore import (
    QEasingCurve,
    QElapsedTimer,
    QEvent,
    QPropertyAnimation,
    Qt,
    QTimer,
    QUrl,
)
from PySide6.QtGui import QAction, QColor, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .audio import AUDIO_SOURCE_BOTH, AUDIO_SOURCE_MIC, AUDIO_SOURCE_SYSTEM
from .config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_SESSIONS_DIR,
    LANGUAGES,
    AppConfig,
    load_config,
    save_config,
)
from .session import Session, Signals
from .theme import PALETTES, Palette, app_icon, stylesheet
from .transcript import RecentTranscript, TranscriptEntry, export_transcript
from .widgets import (
    CaptionOverlay,
    LevelMeter,
    ProtectedLive,
    SafeTranscript,
    entry_fragments,
    find_keywords,
    parse_keywords,
)

__all__ = ["MainWindow", "ProtectedLive", "SafeTranscript"]

logger = logging.getLogger(__name__)

AUDIO_SOURCE_HINTS = {
    AUDIO_SOURCE_SYSTEM: "System output (WASAPI loopback)",
    AUDIO_SOURCE_MIC: "Default microphone",
    AUDIO_SOURCE_BOTH: "System output + microphone",
}

EXPORT_FILTERS = (
    "Plain text (*.txt);;Markdown (*.md);;SRT Subtitles (*.srt);;"
    "WebVTT Subtitles (*.vtt);;JSON (*.json)"
)
EXPORT_SUFFIXES = (".txt", ".md", ".srt", ".vtt", ".json")

EMPTY_STATE = (
    "Press  ●  (or Ctrl+R) to start listening.\n\n"
    "Everything is transcribed locally on this computer: no cloud, no API keys.\n"
    "Tip: pick your language below for the best accuracy."
)


def _icon_button(text: str, tooltip: str, checkable: bool = False) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("icon")
    button.setToolTip(tooltip)
    button.setCheckable(checkable)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


class MainWindow(QMainWindow):
    def __init__(
        self,
        config_path: Path | None = None,
        sessions_dir: Path | None = None,
    ):
        super().__init__()
        self.setWindowTitle("LocalScribe — Live Transcriber")
        self.setWindowIcon(app_icon())
        self.resize(940, 720)
        self.setMinimumSize(640, 520)
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
        self._session_offset: float = 0.0
        self.latest_metrics = None
        self.palette_: Palette = PALETTES["dark"]
        self.keywords: list[str] = []
        self._search_hits = 0
        self._search_index = 0
        self.clock = QElapsedTimer()
        self.caption_overlay = CaptionOverlay()
        self.caption_overlay.closed.connect(
            lambda: self.overlay_button.setChecked(False)
        )

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self._write_config)

        self._build_ui()
        self._load_persisted_config()
        self._connect_config_persistence()
        self._build_shortcuts()

        self.metrics_timer = QTimer(self)
        self.metrics_timer.setInterval(1000)
        self.metrics_timer.timeout.connect(self._refresh_metrics)
        self.metrics_timer.start()

        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(60)
        self.ui_timer.timeout.connect(self._tick)
        self.ui_timer.start()

        self._set_status("Ready. Choose a source and press ● to start.")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)

        layout.addLayout(self._build_header())
        self.drawer = self._build_drawer()
        layout.addWidget(self.drawer)
        layout.addWidget(self._build_transcript_card(), stretch=1)
        layout.addWidget(self._build_live_card())
        layout.addWidget(self._build_control_bar())
        layout.addLayout(self._build_footer())

    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setSpacing(8)
        logo = QLabel()
        logo.setPixmap(app_icon().pixmap(26, 26))
        header.addWidget(logo)
        title = QLabel("LocalScribe")
        title.setObjectName("title")
        header.addWidget(title)
        header.addSpacing(14)

        self.rec_dot = QLabel("●")
        self.rec_dot.setObjectName("recClock")
        header.addWidget(self.rec_dot)
        self.rec_clock = QLabel("00:00:00")
        self.rec_clock.setObjectName("recClock")
        self.rec_clock.setToolTip("Recording time for the current session")
        header.addWidget(self.rec_clock)
        self.level_meter = LevelMeter()
        header.addWidget(self.level_meter)
        header.addStretch(1)

        self.search = QLineEdit()
        self.search.setObjectName("search")
        self.search.setPlaceholderText("⌕  Search history  (Ctrl+F)")
        self.search.setClearButtonEnabled(True)
        self.search.returnPressed.connect(self._find_next)
        self.search.textChanged.connect(self._on_search_changed)
        header.addWidget(self.search)
        self.search_count = QLabel("")
        self.search_count.setObjectName("muted")
        header.addWidget(self.search_count)

        self.overlay_button = _icon_button(
            "💬",
            "Floating live captions overlay  (Ctrl+Shift+O)",
            checkable=True,
        )
        self.overlay_button.toggled.connect(self._toggle_overlay)
        header.addWidget(self.overlay_button)
        self.theme_button = _icon_button("☾", "Toggle light / dark theme")
        self.theme_button.clicked.connect(self._toggle_theme)
        header.addWidget(self.theme_button)
        self.settings_button = _icon_button("⚙", "Settings  (Ctrl+,)", checkable=True)
        self.settings_button.toggled.connect(self._toggle_drawer)
        header.addWidget(self.settings_button)
        self.pin_button = _icon_button(
            "📌", "Keep this window above other windows", checkable=True
        )
        self.pin_button.toggled.connect(self._toggle_topmost)
        header.addWidget(self.pin_button)
        return header

    def _build_drawer(self) -> QFrame:
        drawer = QFrame()
        drawer.setObjectName("drawer")
        grid = QGridLayout(drawer)
        grid.setContentsMargins(16, 14, 16, 14)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        def section(text: str) -> QLabel:
            label = QLabel(text.upper())
            label.setObjectName("sectionTitle")
            return label

        grid.addWidget(section("Vocabulary hint"), 0, 0)
        self.vocabulary = QLineEdit()
        self.vocabulary.setMaxLength(600)
        self.vocabulary.setPlaceholderText(
            "Names, jargon, acronyms: Docker, Kubernetes, FastAPI, PyTorch..."
        )
        self.vocabulary.setToolTip(
            "Context hint for Whisper (initial_prompt). Helps spell names and terms."
        )
        grid.addWidget(self.vocabulary, 1, 0, 1, 2)

        grid.addWidget(section("Keyword alerts"), 0, 2)
        self.keywords_edit = QLineEdit()
        self.keywords_edit.setMaxLength(300)
        self.keywords_edit.setPlaceholderText("Your name, deadline, budget…")
        self.keywords_edit.setToolTip(
            "Comma-separated words. When someone says one, it is highlighted "
            "and the taskbar flashes."
        )
        self.keywords_edit.textChanged.connect(self._on_keywords_changed)
        grid.addWidget(self.keywords_edit, 1, 2, 1, 2)

        grid.addWidget(section("Speech sensitivity"), 2, 0)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.001, 0.060)
        self.threshold.setDecimals(3)
        self.threshold.setSingleStep(0.002)
        self.threshold.setValue(0.007)
        self.threshold.setPrefix("Min. level ")
        self.threshold.setToolTip(
            "Minimum voice level. It adapts automatically to background noise; "
            "lower it if quiet speech is missed."
        )
        grid.addWidget(self.threshold, 3, 0)

        grid.addWidget(section("Compute"), 2, 1)
        self.device = QComboBox()
        self.device.addItem("Auto (GPU if available)", "auto")
        self.device.addItem("CPU only", "cpu")
        self.device.addItem("NVIDIA GPU (CUDA)", "cuda")
        self.device.setToolTip("GPU makes the Turbo model usable in real time.")
        grid.addWidget(self.device, 3, 1)

        grid.addWidget(section("Data"), 2, 2)
        open_sessions = QPushButton("📁 Session backups")
        open_sessions.setToolTip("Open the folder with automatic session backups")
        open_sessions.clicked.connect(self._open_sessions_folder)
        grid.addWidget(open_sessions, 3, 2)

        self.diagnostics_button = QCheckBox("Show diagnostics")
        grid.addWidget(self.diagnostics_button, 3, 3)
        self.diagnostics = QLabel("Waiting for speech recognition metrics…")
        self.diagnostics.setObjectName("muted")
        self.diagnostics.setWordWrap(True)
        self.diagnostics.hide()
        self.diagnostics_button.toggled.connect(self.diagnostics.setVisible)
        grid.addWidget(self.diagnostics, 4, 0, 1, 4)
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(2, 2)
        drawer.hide()
        return drawer

    def _build_transcript_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        box = QVBoxLayout(card)
        box.setContentsMargins(4, 10, 4, 4)
        box.setSpacing(0)

        heading = QHBoxLayout()
        heading.setContentsMargins(14, 0, 8, 0)
        label = QLabel("TRANSCRIPT")
        label.setObjectName("sectionTitle")
        heading.addWidget(label)
        self.entry_count = QLabel("")
        self.entry_count.setObjectName("muted")
        heading.addWidget(self.entry_count)
        heading.addStretch(1)

        self.copy_menu_button = QToolButton()
        self.copy_menu_button.setObjectName("icon")
        self.copy_menu_button.setText("⧉ Copy")
        self.copy_menu_button.setToolTip("Copy options")
        self.copy_menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        copy_menu = QMenu(self)
        copy_menu.addAction("Copy selection", self._copy_selection)
        copy_menu.addAction("Copy last block\tCtrl+Shift+C", self._copy_last)
        copy_menu.addAction("Copy everything", self._copy_all)
        copy_menu.addSeparator()
        copy_menu.addAction("Copy live draft", self._copy_draft)
        copy_menu.addAction("Copy recent utterance", self._copy_recent)
        self.copy_menu_button.setMenu(copy_menu)
        heading.addWidget(self.copy_menu_button)

        self.star_button = _icon_button("☆", "Bookmark the last block  (Ctrl+B)")
        self.star_button.clicked.connect(self._star_last)
        heading.addWidget(self.star_button)
        self.new_session_button = _icon_button("🗋", "New transcript  (Ctrl+N)")
        self.new_session_button.clicked.connect(self._new_transcript)
        heading.addWidget(self.new_session_button)
        box.addLayout(heading)

        self.transcript = SafeTranscript()
        self.transcript.setPlaceholderText(EMPTY_STATE)
        self.transcript.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.transcript.customContextMenuRequested.connect(self._transcript_menu)
        self.transcript.unseen_changed.connect(self._on_unseen_changed)
        box.addWidget(self.transcript, stretch=1)

        # Floating "new messages" pill, Slack-style, instead of a static button.
        self.jump_button = QPushButton("↓ Latest", self.transcript)
        self.jump_button.setObjectName("newPill")
        self.jump_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.jump_button.clicked.connect(self.transcript.go_to_latest)
        self.jump_button.hide()
        self.transcript.installEventFilter(self)
        return card

    def _build_live_card(self) -> QFrame:
        self.live_card = QFrame()
        self.live_card.setObjectName("liveCard")
        self.live_card.setProperty("active", False)
        box = QVBoxLayout(self.live_card)
        box.setContentsMargins(14, 10, 10, 8)
        box.setSpacing(4)
        top = QHBoxLayout()
        self.live_pill = QLabel("● LIVE")
        self.live_pill.setObjectName("livePill")
        self.live_pill.setProperty("active", False)
        top.addWidget(self.live_pill)
        hint = QLabel("provisional · refined when the speaker pauses")
        hint.setObjectName("muted")
        top.addWidget(hint)
        top.addStretch(1)
        self.copy_draft = _icon_button("⧉", "Copy the live draft")
        self.copy_draft.clicked.connect(self._copy_draft)
        top.addWidget(self.copy_draft)
        self.copy_recent = _icon_button("⧉⧉", "Copy the recent utterance (incl. draft)")
        self.copy_recent.clicked.connect(self._copy_recent)
        top.addWidget(self.copy_recent)
        box.addLayout(top)

        self.live = ProtectedLive()
        self.live.setObjectName("live")
        self.live.setReadOnly(True)
        self.live.setMaximumHeight(78)
        self.live.setMinimumHeight(48)
        self.live.setPlaceholderText("Live captions appear here while someone speaks…")
        box.addWidget(self.live)
        return self.live_card

    def _build_control_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("controlBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(12, 10, 14, 10)
        row.setSpacing(10)

        self.start_button = QPushButton("●")
        self.start_button.setObjectName("record")
        self.start_button.setProperty("recording", False)
        self.start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_button.setToolTip("Start listening  (Ctrl+R)")
        self.start_button.clicked.connect(self.start_or_stop)
        self.glow = QGraphicsDropShadowEffect(self.start_button)
        self.glow.setOffset(0, 0)
        self.glow.setBlurRadius(0)
        self.start_button.setGraphicsEffect(self.glow)
        self.pulse = QPropertyAnimation(self.glow, b"blurRadius", self)
        self.pulse.setDuration(1100)
        self.pulse.setStartValue(4.0)
        self.pulse.setKeyValueAt(0.5, 30.0)
        self.pulse.setEndValue(4.0)
        self.pulse.setEasingCurve(QEasingCurve.Type.InOutSine)
        self.pulse.setLoopCount(-1)
        row.addWidget(self.start_button)

        state = QVBoxLayout()
        state.setSpacing(0)
        self.state_label = QLabel("Ready")
        self.state_label.setObjectName("title")
        state.addWidget(self.state_label)
        self.device_label = QLabel(AUDIO_SOURCE_HINTS[AUDIO_SOURCE_SYSTEM])
        self.device_label.setObjectName("hint")
        self.device_label.setMaximumWidth(260)
        state.addWidget(self.device_label)
        row.addLayout(state)
        row.addStretch(1)

        self.audio_source = QComboBox()
        self.audio_source.addItem("🔊  System audio", AUDIO_SOURCE_SYSTEM)
        self.audio_source.addItem("🎙  Microphone", AUDIO_SOURCE_MIC)
        self.audio_source.addItem("🔊+🎙  Both", AUDIO_SOURCE_BOTH)
        self.audio_source.setToolTip("What to listen to")
        self.audio_source.currentIndexChanged.connect(self._on_audio_source_changed)
        row.addWidget(self.audio_source)

        self.model = QComboBox()
        self.model.addItem("⚡  Fast · tiny", "tiny")
        self.model.addItem("⚖  Balanced · base", "base")
        self.model.addItem("🎯  Accurate · small", "small")
        self.model.addItem("🚀  Turbo · large-v3 (GPU)", "large-v3-turbo")
        self.model.setCurrentIndex(1)
        self.model.setToolTip(
            "Speech model. Downloaded once, then works offline. "
            "Turbo needs a GPU for real time."
        )
        row.addWidget(self.model)

        self.language = QComboBox()
        for code, label in LANGUAGES:
            self.language.addItem(f"🌐  {label}", code)
        self.language.setToolTip(
            "Spoken language. Setting it avoids wrong-language output on short "
            "phrases; Auto-detect locks the language once it is confident."
        )
        row.addWidget(self.language)

        self.export_button = QPushButton("⤓ Export")
        self.export_button.setObjectName("primary")
        self.export_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_button.setToolTip(
            "Export as .txt, .md, .srt, .vtt or .json  (Ctrl+E)"
        )
        self.export_button.clicked.connect(self._export)
        row.addWidget(self.export_button)
        for combo in (self.audio_source, self.model, self.language):
            combo.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        return bar

    def _build_footer(self) -> QHBoxLayout:
        footer = QHBoxLayout()
        footer.setContentsMargins(4, 0, 4, 0)
        self.status = QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        footer.addWidget(self.status, stretch=1)
        self.health = QLabel("●")
        self.health.setObjectName("status")
        footer.addWidget(self.health)
        self.engine_badge = QLabel("idle")
        self.engine_badge.setObjectName("status")
        footer.addWidget(self.engine_badge)
        return footer

    def _build_shortcuts(self) -> None:
        bindings = {
            "Ctrl+F": self._focus_search,
            "F3": self._find_next,
            "Ctrl+L": self.transcript.go_to_latest,
            "Ctrl+R": self.start_or_stop,
            "Ctrl+E": self._export,
            "Ctrl+B": self._star_last,
            "Ctrl+N": self._new_transcript,
            "Ctrl+Shift+C": self._copy_last,
            "Ctrl+Shift+O": self.overlay_button.toggle,
            "Ctrl+,": self.settings_button.toggle,
            "Escape": self._escape,
        }
        for keys, handler in bindings.items():
            QShortcut(QKeySequence(keys), self, activated=handler)

    def eventFilter(self, obj, event):
        if obj is self.transcript and event.type() == QEvent.Type.Resize:
            self._place_jump_pill()
        return super().eventFilter(obj, event)

    def _place_jump_pill(self) -> None:
        self.jump_button.adjustSize()
        margin = 14
        x = (self.transcript.width() - self.jump_button.width()) // 2
        y = self.transcript.height() - self.jump_button.height() - margin
        self.jump_button.move(max(0, x), max(0, y))

    # ------------------------------------------------------------------
    # Theme & window state
    # ------------------------------------------------------------------

    def _apply_theme(self, name: str) -> None:
        self.palette_ = PALETTES.get(name, PALETTES["dark"])
        self.setStyleSheet(stylesheet(self.palette_))
        self.transcript.set_palette(self.palette_)
        self.level_meter.set_palette(self.palette_)
        self.caption_overlay.set_palette(self.palette_)
        self.glow.setColor(QColor(self.palette_.danger))
        self.theme_button.setText("☀" if self.palette_.name == "dark" else "☾")
        self._rerender_transcript()
        self._update_health(None)

    def _toggle_overlay(self, enabled: bool) -> None:
        if enabled:
            last_text = self.entries[-1].text if self.entries else ""
            self.caption_overlay.set_caption(
                confirmed=last_text, draft=self.recent.draft
            )
            self.caption_overlay.show()
            self.caption_overlay.raise_()
            self._set_status(
                "Floating live captions ON (drag to move, wheel to resize, double-click to hide)."
            )
        else:
            self.caption_overlay.hide()

    def _toggle_theme(self) -> None:
        self._apply_theme("light" if self.palette_.name == "dark" else "dark")
        self._save_current_config()

    def _repolish(self, *widgets: QWidget) -> None:
        for widget in widgets:
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def _toggle_drawer(self, visible: bool) -> None:
        self.drawer.setVisible(visible)

    def _toggle_topmost(self, enabled: bool) -> None:
        was_visible = self.isVisible()
        geometry = self.geometry()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
        if was_visible:
            self.show()
            self.setGeometry(geometry)
        self.pin_button.setToolTip(
            "Always on top: ON" if enabled else "Keep this window above other windows"
        )

    def _escape(self) -> None:
        if self.search.hasFocus() or self.search.text():
            self.search.clear()
            self.transcript.setFocus()
        elif self.drawer.isVisible():
            self.settings_button.setChecked(False)

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def _load_persisted_config(self) -> None:
        self._loading_config = True
        try:
            config = load_config(self.config_path)
            for combo, value in (
                (self.model, config.model),
                (self.audio_source, config.audio_source),
                (self.language, config.language),
                (self.device, config.device),
            ):
                index = combo.findData(value)
                if index >= 0:
                    combo.setCurrentIndex(index)
            self.threshold.setValue(config.threshold)
            self.vocabulary.setText(config.vocabulary)
            self.keywords_edit.setText(config.keywords)
            self.pin_button.setChecked(config.always_on_top)
            self._apply_theme(config.theme)
            self._on_audio_source_changed()
        finally:
            self._loading_config = False

    def _connect_config_persistence(self) -> None:
        self.model.currentIndexChanged.connect(self._save_current_config)
        self.language.currentIndexChanged.connect(self._save_current_config)
        self.device.currentIndexChanged.connect(self._save_current_config)
        self.threshold.valueChanged.connect(self._save_current_config)
        self.audio_source.currentIndexChanged.connect(self._save_current_config)
        self.pin_button.toggled.connect(self._save_current_config)
        self.vocabulary.textChanged.connect(self._save_current_config)
        self.keywords_edit.textChanged.connect(self._save_current_config)

    def current_config(self) -> AppConfig:
        return AppConfig(
            model=str(self.model.currentData()),
            threshold=float(self.threshold.value()),
            audio_source=str(self.audio_source.currentData()),
            always_on_top=bool(self.pin_button.isChecked()),
            vocabulary=self.vocabulary.text(),
            language=str(self.language.currentData()),
            device=str(self.device.currentData()),
            theme=self.palette_.name,
            keywords=self.keywords_edit.text(),
        )

    def _save_current_config(self, *_args) -> None:
        """Debounced: typing in a text field does not hit the disk per key."""
        if self._loading_config:
            return
        self._save_timer.start()

    def _write_config(self) -> None:
        self._save_timer.stop()
        try:
            save_config(self.current_config(), self.config_path)
        except OSError as exc:
            logger.warning("Could not save config.json: %s", exc)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _set_status(self, value: str) -> None:
        self.status.setText(value)

    def _on_audio_source_changed(self) -> None:
        if self.is_recording:
            return
        source = str(self.audio_source.currentData())
        self.device_label.setText(
            AUDIO_SOURCE_HINTS.get(source, AUDIO_SOURCE_HINTS[AUDIO_SOURCE_SYSTEM])
        )

    def _on_device_name(self, name: str) -> None:
        self.device_label.setText(name)
        self.device_label.setToolTip(name)

    def _on_keywords_changed(self, text: str) -> None:
        self.keywords = parse_keywords(text)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.audio_source,
            self.model,
            self.language,
            self.device,
            self.threshold,
            self.vocabulary,
            self.new_session_button,
        ):
            widget.setEnabled(enabled)

    def _set_recording_visuals(self, recording: bool) -> None:
        self.start_button.setProperty("recording", recording)
        self.start_button.setText("■" if recording else "●")
        self.start_button.setToolTip(
            "Stop  (Ctrl+R)" if recording else "Start listening  (Ctrl+R)"
        )
        self.rec_clock.setProperty("recording", recording)
        self.rec_dot.setProperty("recording", recording)
        self.live_card.setProperty("active", recording)
        self.live_pill.setProperty("active", recording)
        self._repolish(
            self.start_button,
            self.rec_clock,
            self.rec_dot,
            self.live_card,
            self.live_pill,
        )
        if recording:
            self.pulse.start()
        else:
            self.pulse.stop()
            self.glow.setBlurRadius(0)

    def _tick(self) -> None:
        """60 ms UI heartbeat: level meter, clock, blinking REC dot."""
        segmenter = self.session.segmenter if self.session else None
        if segmenter is not None and self.is_recording:
            level = segmenter.level
            self.level_meter.push(level, level >= segmenter.effective_threshold)
        if self.is_recording and self.clock.isValid():
            seconds = self.clock.elapsed() // 1000
            self.rec_clock.setText(
                f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
            )
            visible = (self.clock.elapsed() // 600) % 2 == 0
            self.rec_dot.setText("●" if visible else " ")

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_or_stop(self) -> None:
        if self.is_recording:
            self._stop()
        elif not self.is_stopping:
            self._start()

    def _start(self) -> None:
        self.signals = Signals(self)
        self.signals.status.connect(self._set_status)
        self.signals.device.connect(self._on_device_name)
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
        self._session_offset = self._timeline_end
        if self._timeline_origin is None:
            self._timeline_origin = monotonic()

        language = str(self.language.currentData())
        self.session = Session(
            self.signals,
            str(self.model.currentData()),
            self.threshold.value(),
            self.vocabulary.text(),
            audio_source=str(self.audio_source.currentData()),
            sessions_dir=self.sessions_dir,
            language=None if language == "auto" else language,
            device=str(self.device.currentData()),
        )
        self._set_controls_enabled(False)
        self.is_recording = True
        self._set_recording_visuals(True)
        self.state_label.setText("Listening")
        self.clock.start()
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
        self._set_recording_visuals(False)
        self.state_label.setText("Finishing…")
        self._set_status("Stopping capture and processing remaining speech...")
        self.session.stop()

    def _on_finished(self) -> None:
        self.is_recording = False
        self.is_stopping = False
        self.start_button.setEnabled(True)
        self._set_recording_visuals(False)
        self._set_controls_enabled(True)
        self.state_label.setText("Ready")
        self.level_meter.reset()
        self._on_audio_source_changed()
        self.engine_badge.setText("idle")
        self.session = None
        if self._closing:
            # The user already asked to close: finish now instead of a 2nd click.
            QTimer.singleShot(0, self.close)
            return
        self._set_status(
            self._last_error
            or "Stopped. Your transcript is saved and ready to copy or export."
        )

    def _on_error(self, message: str) -> None:
        self._last_error = message
        if self.is_recording:
            self._stop()
        self._set_status(message)

    # ------------------------------------------------------------------
    # Recognition results
    # ------------------------------------------------------------------

    def _on_partial(self, number: int, content: str, group: int = 0) -> None:
        if not self.recent.partial(number, content, group):
            return
        self.latest_partial_id = number
        # Drafts live in a separate panel so live revisions never break history selection.
        self.live.update_text(content)
        last_text = self.entries[-1].text if self.entries else ""
        self.caption_overlay.set_caption(confirmed=last_text, draft=content)

    def _fallback_timing(self, content: str) -> tuple[float, float]:
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
        return start, start + duration

    def _on_final(
        self,
        number: int,
        content: str,
        when: str,
        group: int = 0,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
    ) -> None:
        if not self.recent.final(number, content, group):
            return
        self.last_completed_id = number
        if content:
            if (
                start_seconds is not None
                and end_seconds is not None
                and end_seconds > start_seconds
            ):
                # Real audio positions from the segmenter's sample counter.
                start = max(self._timeline_end, self._session_offset + start_seconds)
                end = max(start + 0.3, self._session_offset + end_seconds)
            else:
                start, end = self._fallback_timing(content)
            self._timeline_end = end

            speaker = ""
            if (
                self.session is not None
                and getattr(self.session, "audio_source", "") == AUDIO_SOURCE_BOTH
            ):
                speaker = getattr(self.session.engine, "last_speaker", "")
            entry = TranscriptEntry(
                timestamp=when,
                text=content,
                start_seconds=start,
                end_seconds=end,
                speaker=speaker,
            )
            self.entries.append(entry)
            self.last_block = f"[{when}] {content}"
            hits = find_keywords(content, self.keywords)
            self.transcript.add_fragments(
                entry_fragments(when, content, False, hits, speaker=speaker)
            )
            self.caption_overlay.set_caption(confirmed=content, draft=self.recent.draft)
            self._update_entry_count()
            if hits:
                self._keyword_alert(hits)
            if self.search.text().strip():
                self._on_search_changed(self.search.text())
        if number >= self.latest_partial_id:
            self.live.update_text("")

    def _keyword_alert(self, hits: list[str]) -> None:
        QApplication.alert(self, 0)
        QApplication.beep()
        self._set_status(f"🔔 Mentioned: {', '.join(hits)}")

    def _on_metrics(self, metrics: dict) -> None:
        self.latest_metrics = metrics

    def _update_health(self, metrics: dict | None, pending: float = 0.0) -> None:
        p = self.palette_
        if not metrics:
            color = p.muted
        elif metrics["rtf"] > 1 or pending > 16:
            color = p.danger
        elif metrics["rtf"] > 0.6 or pending > 6:
            color = p.warning
        else:
            color = p.success
        self.health.setStyleSheet(f"color: {color};")

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
        warning = (
            " · BACKLOG: try a faster model" if m["rtf"] > 1 or pending > 16 else ""
        )
        running = ""
        started = self.session.engine.inference_started if self.session else None
        if started is not None:
            running = f" · Active inference: {monotonic() - started:.1f}s"
        detail = (
            f"{m['kind']}: inference {m['inference']:.2f}s / audio {m['audio']:.2f}s "
            f"· RTF {m['rtf']:.2f} (measured)\n"
            f"Latest audio age at emit: {m['lag']:.2f}s · "
            f"Pending ≈{pending:.1f}s{running}{warning}"
        )
        self.diagnostics.setText(detail)
        device = str(m.get("device", "cpu")).upper()
        self.engine_badge.setText(
            f"{self.model.currentData()} · {device} · RTF {m['rtf']:.2f}"
        )
        self.engine_badge.setToolTip(detail)
        self.health.setToolTip(detail)
        self._update_health(m, pending)

    # ------------------------------------------------------------------
    # Transcript actions
    # ------------------------------------------------------------------

    def _update_entry_count(self) -> None:
        count = len(self.entries)
        starred = sum(1 for e in self.entries if e.starred)
        text = f"· {count} block{'s' if count != 1 else ''}" if count else ""
        if starred:
            text += f" · ★ {starred}"
        self.entry_count.setText(text)

    def _rerender_transcript(self) -> None:
        if not self.entries:
            return
        self.transcript.clear_entries()
        for entry in self.entries:
            hits = find_keywords(entry.text, self.keywords)
            self.transcript.add_fragments(
                entry_fragments(
                    entry.timestamp,
                    entry.text,
                    entry.starred,
                    hits,
                    speaker=entry.speaker,
                )
            )

    def _toggle_star(self, index: int) -> None:
        if not 0 <= index < len(self.entries):
            return
        old = self.entries[index]
        entry = TranscriptEntry(
            old.timestamp,
            old.text,
            old.start_seconds,
            old.end_seconds,
            not old.starred,
            old.speaker,
        )
        self.entries[index] = entry
        hits = find_keywords(entry.text, self.keywords)
        self.transcript.replace_block(
            index,
            entry_fragments(
                entry.timestamp,
                entry.text,
                entry.starred,
                hits,
                speaker=entry.speaker,
            ),
        )
        self._update_entry_count()
        self._set_status("★ Bookmarked." if entry.starred else "Bookmark removed.")

    def _star_last(self) -> None:
        if self.entries:
            self._toggle_star(len(self.entries) - 1)
        else:
            self._set_status("Nothing to bookmark yet.")

    def _transcript_menu(self, pos) -> None:
        cursor = self.transcript.cursorForPosition(pos)
        index = cursor.blockNumber()
        menu = QMenu(self)
        has_selection = self.transcript.textCursor().hasSelection()
        copy_sel = QAction("Copy selection", menu)
        copy_sel.setEnabled(has_selection)
        copy_sel.triggered.connect(self._copy_selection)
        menu.addAction(copy_sel)
        if 0 <= index < len(self.entries):
            entry = self.entries[index]
            menu.addAction(
                "Copy this block",
                lambda: self._copy_text(
                    f"[{entry.timestamp}] {entry.text}", "Block copied."
                ),
            )
            menu.addAction(
                "Remove bookmark" if entry.starred else "★ Bookmark this block",
                lambda: self._toggle_star(index),
            )
        menu.addSeparator()
        menu.addAction("Copy everything", self._copy_all)
        menu.addAction("Export…", self._export)
        menu.exec(self.transcript.viewport().mapToGlobal(pos))

    def _on_unseen_changed(self, count: int) -> None:
        if count:
            self.jump_button.setText(f"↓ {count} new")
            self.jump_button.show()
            self.jump_button.raise_()
            self._place_jump_pill()
        else:
            self.jump_button.hide()

    def _new_transcript(self) -> None:
        if self.is_recording or self.is_stopping:
            self._set_status(
                "Stop the current session before starting a new transcript."
            )
            return
        if self.entries:
            answer = QMessageBox.question(
                self,
                "New transcript",
                "Clear the current transcript? (A backup is already saved in "
                "the sessions folder.)",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.entries.clear()
        self.transcript.clear_entries()
        self.live.update_text("")
        self.recent = RecentTranscript()
        self.last_block = ""
        self._timeline_origin = None
        self._timeline_end = 0.0
        self._session_offset = 0.0
        self.rec_clock.setText("00:00:00")
        self._update_entry_count()
        self._set_status("New transcript ready.")

    def _open_sessions_folder(self) -> None:
        folder = self.sessions_dir or DEFAULT_SESSIONS_DIR
        folder.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(folder)  # noqa: S606 - opens Explorer on a local folder
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # --- Copy -----------------------------------------------------------

    def _copy_text(self, text: str, message: str) -> None:
        QApplication.clipboard().setText(text)
        self._set_status(message)

    def _copy_draft(self) -> None:
        if self.recent.draft:
            self._copy_text(
                self.recent.draft, "Draft copied; wording may still be revised."
            )
        else:
            self._set_status("No live draft available yet.")

    def _copy_recent(self) -> None:
        text = self.recent.recent()
        if text:
            self._copy_text(
                text, "Recent utterance copied (includes active draft if any)."
            )

    def _copy_selection(self) -> None:
        content = self.transcript.textCursor().selectedText().replace("\u2029", "\n")
        if content:
            self._copy_text(content, "Selection copied.")
        else:
            self._set_status("Select some text first to copy.")

    def _copy_last(self) -> None:
        if self.last_block:
            self._copy_text(self.last_block, "Last confirmed block copied.")
        else:
            self._set_status("No confirmed block available yet.")

    def _copy_all(self) -> None:
        if not self.entries:
            self._set_status("Nothing to copy yet.")
            return
        text = "\n".join(f"[{e.timestamp}] {e.text}" for e in self.entries)
        self._copy_text(text, f"Copied {len(self.entries)} blocks.")

    # --- Search ---------------------------------------------------------

    def _focus_search(self) -> None:
        self.search.setFocus()
        self.search.selectAll()

    def _on_search_changed(self, text: str) -> None:
        query = text.strip()
        self._search_hits = self.transcript.highlight_all(query)
        self._search_index = 0
        if not query:
            self.search_count.setText("")
        else:
            self.search_count.setText(
                f"{self._search_hits} match{'es' if self._search_hits != 1 else ''}"
                if self._search_hits
                else "no matches"
            )

    def _find_next(self) -> None:
        query = self.search.text().strip()
        if not query:
            return
        if not self.transcript.find(query):
            cursor = self.transcript.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            self.transcript.setTextCursor(cursor)
            self._search_index = 0
            if not self.transcript.find(query):
                self._set_status(f"Not found: {query}")
                return
        self._search_index = self._search_index % max(1, self._search_hits) + 1
        if self._search_hits:
            self.search_count.setText(f"{self._search_index}/{self._search_hits}")
        self._set_status("Match found; scrolling remains under your control.")

    # --- Export ---------------------------------------------------------

    @staticmethod
    def _resolve_export_target(filename: str, selected_filter: str) -> tuple[Path, str]:
        path = Path(filename)
        suffix = path.suffix.lower()
        if suffix in EXPORT_SUFFIXES:
            return path, suffix.lstrip(".")
        for ext in ("md", "srt", "vtt", "json"):
            if f"*.{ext}" in selected_filter:
                return path.with_suffix(f".{ext}"), ext
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
            self._set_status(
                f"Transcript exported successfully (.{fmt}) → {target.name}"
            )

    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._write_config()
        if self.session is not None:
            # Drain queued audio first; the window closes itself when finished.
            self._closing = True
            self._stop()
            self._set_status(
                "Finishing pending audio… the window will close by itself."
            )
            event.ignore()
            return
        self.caption_overlay.close()
        event.accept()
