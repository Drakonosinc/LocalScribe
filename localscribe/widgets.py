"""Custom widgets: smart-scrolling transcript, protected live draft, level meter."""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Iterable

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
    QTextFormat,
)
from PySide6.QtWidgets import QPlainTextEdit, QTextEdit, QWidget

from .theme import DARK, Palette

Fragment = tuple[str, str]  # (text, role) where role selects a char format.


def parse_keywords(raw: str) -> list[str]:
    return [k.strip() for k in raw.split(",") if len(k.strip()) >= 2]


def find_keywords(text: str, keywords: Iterable[str]) -> list[str]:
    """Keywords present in text (case-insensitive, whole-word when possible)."""
    hits = []
    for keyword in keywords:
        pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(keyword)
    return hits


def entry_fragments(
    timestamp: str | None,
    text: str,
    starred: bool = False,
    keywords: Iterable[str] = (),
    speaker: str = "",
) -> list[Fragment]:
    fragments: list[Fragment] = []
    if starred:
        fragments.append(("★ ", "star"))
    if timestamp:
        fragments.append((f"[{timestamp}] ", "time"))
    if speaker == "mic":
        fragments.append(("🎙 You · ", "speaker_mic"))
    elif speaker == "system":
        fragments.append(("🔊 Them · ", "speaker_sys"))
    keywords = [k for k in keywords if k]
    if not keywords:
        fragments.append((text, "body"))
        return fragments
    pattern = "|".join(rf"(?<!\w){re.escape(k)}(?!\w)" for k in keywords)
    position = 0
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        if match.start() > position:
            fragments.append((text[position : match.start()], "body"))
        fragments.append((match.group(0), "keyword"))
        position = match.end()
    if position < len(text):
        fragments.append((text[position:], "body"))
    return fragments


class SafeTranscript(QTextEdit):
    """Confirmed history that never jumps while reading or selecting.

    New blocks are appended with a cursor (never ``setPlainText``) so active
    selections, anchors, and the scroll position survive live updates.
    """

    unseen_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mouse_selecting = False
        self._palette: Palette = DARK
        self.unseen = 0
        self.setObjectName("transcript")
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.setAcceptRichText(False)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.document().setDocumentMargin(14)
        self.setPlaceholderText("Confirmed transcript history will appear here...")
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)

    # --- Formatting ----------------------------------------------------

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette

    def _format(self, role: str) -> QTextCharFormat:
        p = self._palette
        fmt = QTextCharFormat()
        if role == "time":
            fmt.setForeground(QColor(p.accent))
            fmt.setFontWeight(QFont.Weight.DemiBold)
            fmt.setFontPointSize(9.5)
            fmt.setFontFamilies(["Cascadia Mono", "Consolas"])
        elif role == "star":
            fmt.setForeground(QColor(p.warning))
        elif role == "speaker_mic":
            fmt.setForeground(QColor(p.success))
            fmt.setFontWeight(QFont.Weight.DemiBold)
            fmt.setFontPointSize(9.5)
        elif role == "speaker_sys":
            fmt.setForeground(QColor(p.muted))
            fmt.setFontWeight(QFont.Weight.DemiBold)
            fmt.setFontPointSize(9.5)
        elif role == "keyword":
            fmt.setForeground(QColor(p.text))
            fmt.setBackground(QColor(p.highlight))
            fmt.setFontWeight(QFont.Weight.DemiBold)
        else:
            fmt.setForeground(QColor(p.text))
        return fmt

    @staticmethod
    def _block_format() -> QTextBlockFormat:
        block = QTextBlockFormat()
        block.setTopMargin(4)
        block.setBottomMargin(6)
        block.setLineHeight(
            135, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value
        )
        return block

    def _write(self, cursor: QTextCursor, fragments: list[Fragment]) -> None:
        for text, role in fragments:
            cursor.insertText(text, self._format(role))

    # --- Appending -----------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_selecting = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_selecting = False

    def add_entry(self, text: str) -> None:
        self.add_fragments([(text, "body")])

    def add_fragments(self, fragments: list[Fragment]) -> None:
        bar = self.verticalScrollBar()
        old_position = bar.value()
        old_maximum = bar.maximum()
        old_cursor = self.textCursor()
        follow = (
            old_position >= old_maximum - 2
            and not old_cursor.hasSelection()
            and not self._mouse_selecting
        )
        anchor, position = old_cursor.anchor(), old_cursor.position()
        writer = QTextCursor(self.document())
        writer.movePosition(QTextCursor.MoveOperation.End)
        if self.document().isEmpty():
            writer.setBlockFormat(self._block_format())
        else:
            writer.insertBlock(self._block_format())
        self._write(writer, fragments)
        old_cursor.setPosition(anchor)
        old_cursor.setPosition(position, QTextCursor.MoveMode.KeepAnchor)
        self.setTextCursor(old_cursor)
        if follow:
            bar.setValue(bar.maximum())
        else:
            bar.setValue(old_position)
            self.unseen += 1
            self.unseen_changed.emit(self.unseen)

    def replace_block(self, index: int, fragments: list[Fragment]) -> None:
        """Re-render one entry (e.g. after starring) without moving the view."""
        block = self.document().findBlockByNumber(index)
        if not block.isValid():
            return
        bar = self.verticalScrollBar()
        position = bar.value()
        cursor = QTextCursor(block)
        cursor.movePosition(
            QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor
        )
        cursor.removeSelectedText()
        self._write(cursor, fragments)
        bar.setValue(position)

    def clear_entries(self) -> None:
        self.clear()
        self.unseen = 0
        self.unseen_changed.emit(0)

    # --- Navigation & search -------------------------------------------

    def _on_scroll(self, value: int) -> None:
        if self.unseen and value >= self.verticalScrollBar().maximum() - 2:
            self.unseen = 0
            self.unseen_changed.emit(0)

    def go_to_latest(self) -> None:
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        if self.unseen:
            self.unseen = 0
            self.unseen_changed.emit(0)

    def highlight_all(self, query: str) -> int:
        """Paint every match of query; returns the number of matches."""
        selections = []
        if query:
            doc = self.document()
            cursor = QTextCursor(doc)
            fmt = QTextCharFormat()
            fmt.setBackground(QColor(self._palette.highlight))
            fmt.setProperty(QTextFormat.Property.FullWidthSelection, False)
            while True:
                cursor = doc.find(query, cursor)
                if cursor.isNull():
                    break
                selection = QTextEdit.ExtraSelection()
                selection.cursor = cursor
                selection.format = fmt
                selections.append(selection)
        self.setExtraSelections(selections)
        return len(selections)


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


class LevelMeter(QWidget):
    """Scrolling mini-waveform showing live input level and speech activity."""

    BARS = 36

    def __init__(self, parent=None):
        super().__init__(parent)
        self.levels: deque[tuple[float, bool]] = deque(
            [(0.0, False)] * self.BARS, maxlen=self.BARS
        )
        self._palette: Palette = DARK
        self.setToolTip("Input level · bars light up when speech is detected")

    def sizeHint(self) -> QSize:
        return QSize(150, 28)

    def minimumSizeHint(self) -> QSize:
        return QSize(90, 22)

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette
        self.update()

    def push(self, level: float, speaking: bool) -> None:
        self.levels.append((max(0.0, level), speaking))
        self.update()

    def reset(self) -> None:
        self.levels.extend([(0.0, False)] * self.BARS)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        width, height = self.width(), self.height()
        step = width / self.BARS
        bar_w = max(2.0, step * 0.55)
        idle = QColor(self._palette.border)
        voice = QColor(self._palette.accent)
        for i, (level, speaking) in enumerate(self.levels):
            # Perceptual scaling: quiet speech still produces visible bars.
            norm = min(1.0, math.sqrt(level / 0.12)) if level > 0 else 0.0
            bar_h = max(3.0, norm * (height - 2))
            color = QColor(voice if speaking else idle)
            color.setAlphaF(0.35 + 0.65 * (i + 1) / self.BARS)
            painter.setBrush(color)
            x = i * step + (step - bar_w) / 2
            painter.drawRoundedRect(
                QRectF(x, (height - bar_h) / 2, bar_w, bar_h), bar_w / 2, bar_w / 2
            )
        painter.end()


class CaptionOverlay(QWidget):
    """Floating, translucent, draggable live-caption window (always on top)."""

    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(
            parent,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setObjectName("captionOverlay")
        self._drag_offset = None
        self._font_pt = 16
        self._confirmed = ""
        self._draft = ""
        self._palette: Palette = DARK
        self.resize(780, 110)
        self.setMinimumSize(360, 74)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setToolTip(
            "Drag to move · Scroll wheel to resize text · Double-click to close"
        )

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette
        self.update()

    def set_caption(self, confirmed: str = "", draft: str = "") -> None:
        self._confirmed = confirmed.strip()
        self._draft = draft.strip()
        self.update()

    def adjust_font_size(self, delta: int) -> None:
        self._font_pt = max(11, min(32, self._font_pt + delta))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            self._drag_offset is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self.move(event.globalPosition().toPoint() - self._drag_offset)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, _event):
        self.hide()
        self.closed.emit()

    def wheelEvent(self, event):
        step = 1 if event.angleDelta().y() > 0 else -1
        self.adjust_font_size(step)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(2, 2, self.width() - 4, self.height() - 4)
        bg = QColor(13, 17, 23, 215)
        border = QColor(self._palette.accent)
        border.setAlpha(150)
        painter.setBrush(bg)
        painter.setPen(border)
        painter.drawRoundedRect(rect, 16, 16)

        inner = rect.adjusted(18, 12, -18, -12)
        font = QFont("Segoe UI Variable Text", self._font_pt)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)

        if self._draft:
            painter.setPen(QColor(self._palette.accent))
            text = self._draft
        elif self._confirmed:
            painter.setPen(QColor("#f0f6fc"))
            text = self._confirmed
        else:
            painter.setPen(QColor(self._palette.muted))
            text = "● Live captions ready — waiting for speech…"

        painter.drawText(
            inner,
            int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap),
            text,
        )
        painter.end()
