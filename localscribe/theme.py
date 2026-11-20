"""Visual identity: color palettes, Qt stylesheet, and the generated app icon."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPixmap,
)


@dataclass(frozen=True)
class Palette:
    name: str
    bg: str
    surface: str
    surface2: str
    border: str
    text: str
    muted: str
    accent: str
    accent_text: str
    accent_soft: str
    danger: str
    warning: str
    success: str
    live_bg: str
    live_border: str
    selection: str
    highlight: str


DARK = Palette(
    name="dark",
    bg="#0d1117",
    surface="#151b24",
    surface2="#1c2430",
    border="#273242",
    text="#e6edf5",
    muted="#8593a6",
    accent="#3dd6c3",
    accent_text="#04201c",
    accent_soft="#173b3a",
    danger="#ff5c6c",
    warning="#f5b14c",
    success="#3ddc84",
    live_bg="#101e22",
    live_border="#1f4a4a",
    selection="#2f5f9e",
    highlight="#6b5317",
)

LIGHT = Palette(
    name="light",
    bg="#f3f5f9",
    surface="#ffffff",
    surface2="#eef1f6",
    border="#d7dde7",
    text="#18212c",
    muted="#5f6c7e",
    accent="#0e9f8e",
    accent_text="#ffffff",
    accent_soft="#d9f3ef",
    danger="#e5484d",
    warning="#c77d12",
    success="#1f9d55",
    live_bg="#eefaf8",
    live_border="#a8e2d9",
    selection="#b6d4fb",
    highlight="#ffe59a",
)

PALETTES = {"dark": DARK, "light": LIGHT}


def stylesheet(p: Palette) -> str:
    return f"""
    QWidget {{
        background: {p.bg}; color: {p.text};
        font-family: 'Segoe UI Variable Text', 'Segoe UI'; font-size: 10pt;
    }}
    QToolTip {{
        background: {p.surface2}; color: {p.text}; border: 1px solid {p.border};
        padding: 6px; border-radius: 6px;
    }}
    QLabel {{ background: transparent; }}
    QLabel#title {{ font-size: 13pt; font-weight: 700; letter-spacing: 0.5px; }}
    QLabel#muted, QLabel#status, QLabel#hint {{ color: {p.muted}; font-size: 9pt; }}
    QLabel#recClock {{
        color: {p.muted}; font-family: 'Cascadia Mono', 'Consolas';
        font-size: 10pt; font-weight: 600;
    }}
    QLabel#recClock[recording="true"] {{ color: {p.danger}; }}
    QLabel#livePill {{
        background: {p.accent_soft}; color: {p.accent}; border-radius: 9px;
        padding: 2px 9px; font-size: 8pt; font-weight: 700; letter-spacing: 1px;
    }}
    QLabel#livePill[active="false"] {{ background: {p.surface2}; color: {p.muted}; }}
    QLabel#sectionTitle {{
        color: {p.muted}; font-size: 8pt; font-weight: 700; letter-spacing: 1px;
    }}

    QFrame#card, QFrame#drawer, QFrame#controlBar, QFrame#sidebar {{
        background: {p.surface}; border: 1px solid {p.border}; border-radius: 14px;
    }}
    QListWidget#sessionList {{
        background: transparent; border: none; color: {p.text}; outline: none;
    }}
    QListWidget#sessionList::item {{
        padding: 8px 10px; border-radius: 9px; margin: 2px 4px;
    }}
    QListWidget#sessionList::item:hover {{ background: {p.surface2}; }}
    QListWidget#sessionList::item:selected {{
        background: {p.accent_soft}; color: {p.text};
    }}
    QFrame#liveCard {{
        background: {p.live_bg}; border: 1px solid {p.live_border}; border-radius: 14px;
    }}
    QFrame#liveCard[active="false"] {{
        background: {p.surface}; border: 1px solid {p.border};
    }}

    QTextEdit, QPlainTextEdit {{
        background: transparent; border: none; color: {p.text};
        selection-background-color: {p.selection};
    }}
    QTextEdit#transcript {{ font-size: 11.5pt; }}
    QPlainTextEdit#live {{ font-size: 12pt; font-style: italic; color: {p.text}; }}

    QLineEdit, QComboBox, QDoubleSpinBox {{
        background: {p.surface2}; color: {p.text}; border: 1px solid {p.border};
        border-radius: 9px; padding: 6px 10px; min-height: 20px;
        selection-background-color: {p.selection};
    }}
    QFrame#drawer QLineEdit, QFrame#drawer QComboBox, QFrame#drawer QDoubleSpinBox,
    QFrame#controlBar QComboBox {{ background: {p.surface2}; }}
    QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {{
        border: 1px solid {p.accent};
    }}
    QLineEdit#search {{ border-radius: 15px; padding-left: 12px; min-width: 180px; }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QComboBox QAbstractItemView {{
        background: {p.surface}; color: {p.text}; border: 1px solid {p.border};
        selection-background-color: {p.accent_soft}; selection-color: {p.text};
        padding: 4px; outline: none;
    }}
    QComboBox:disabled, QLineEdit:disabled, QDoubleSpinBox:disabled {{
        color: {p.muted};
    }}

    QPushButton, QToolButton {{
        background: {p.surface2}; color: {p.text}; border: 1px solid {p.border};
        border-radius: 9px; padding: 6px 12px;
    }}
    QPushButton:hover, QToolButton:hover {{ border-color: {p.accent}; }}
    QPushButton:pressed, QToolButton:pressed {{ background: {p.accent_soft}; }}
    QPushButton:checked, QToolButton:checked {{
        background: {p.accent_soft}; border-color: {p.accent}; color: {p.accent};
    }}
    QPushButton:disabled, QToolButton:disabled {{ color: {p.muted}; }}
    QToolButton::menu-indicator {{ image: none; width: 0; }}
    QToolButton#icon, QPushButton#icon {{
        background: transparent; border: 1px solid transparent; border-radius: 9px;
        padding: 5px 8px; font-size: 11pt;
    }}
    QToolButton#icon:hover, QPushButton#icon:hover {{
        background: {p.surface2}; border-color: {p.border};
    }}
    QPushButton#icon:checked {{
        background: {p.accent_soft}; border-color: {p.accent}; color: {p.accent};
    }}
    QPushButton#primary {{
        background: {p.accent}; color: {p.accent_text}; border: none; font-weight: 600;
    }}
    QPushButton#record {{
        background: {p.danger}; color: white; border: none; border-radius: 27px;
        font-size: 15pt; font-weight: 700; padding: 0;
        min-width: 54px; max-width: 54px; min-height: 54px; max-height: 54px;
    }}
    QPushButton#record:hover {{ background: {p.danger}; border: 3px solid {p.text}; }}
    QPushButton#record[recording="true"] {{
        background: {p.surface2}; color: {p.danger}; border: 2px solid {p.danger};
    }}
    QPushButton#record:disabled {{ background: {p.surface2}; color: {p.muted}; }}
    QPushButton#newPill {{
        background: {p.accent}; color: {p.accent_text}; border: none;
        border-radius: 14px; padding: 5px 14px; font-weight: 600;
    }}

    QMenu {{
        background: {p.surface}; color: {p.text}; border: 1px solid {p.border};
        border-radius: 10px; padding: 6px;
    }}
    QMenu::item {{ padding: 7px 22px 7px 14px; border-radius: 6px; }}
    QMenu::item:selected {{ background: {p.accent_soft}; }}
    QMenu::separator {{ height: 1px; background: {p.border}; margin: 5px 8px; }}

    QCheckBox {{ background: transparent; spacing: 8px; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px 2px; }}
    QScrollBar::handle:vertical {{
        background: {p.border}; border-radius: 3px; min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {p.muted}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    """


def app_icon(size: int = 256) -> QIcon:
    """Draw the LocalScribe mark: a rounded tile with a stylized waveform."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    rect = QRectF(size * 0.04, size * 0.04, size * 0.92, size * 0.92)
    gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
    gradient.setColorAt(0.0, QColor("#3dd6c3"))
    gradient.setColorAt(1.0, QColor("#4f6cff"))
    path = QPainterPath()
    path.addRoundedRect(rect, size * 0.22, size * 0.22)
    painter.fillPath(path, gradient)

    heights = (0.22, 0.42, 0.62, 0.36, 0.52, 0.28)
    bar_w = size * 0.075
    gap = size * 0.045
    total = len(heights) * bar_w + (len(heights) - 1) * gap
    x = (size - total) / 2
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(255, 255, 255, 235))
    for h in heights:
        bar_h = size * h
        painter.drawRoundedRect(
            QRectF(x, (size - bar_h) / 2, bar_w, bar_h), bar_w / 2, bar_w / 2
        )
        x += bar_w + gap
    painter.end()
    return QIcon(pixmap)
