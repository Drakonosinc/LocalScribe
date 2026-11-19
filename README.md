# LocalScribe

**LocalScribe** is a Windows desktop application (Python 3.12 + PySide6) that transcribes live audio **100% locally**, without uploading recordings to the cloud or requiring API keys.

It captures **system audio via WASAPI loopback**, your **microphone**, or **both sources simultaneously**, displaying a provisional live draft while consolidating each confirmed utterance into a persistent, searchable, and exportable transcript history.

![LocalScribe Screenshot](docs/screenshot.png)

---

## Key Features

- **100% Local Processing:** Runs CPU inference (INT8) with [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper), supporting switchable model profiles (`tiny`, `base`, `small`) and automatic language detection.
- **Audio Source Selector:** Switch directly from the control bar between:
  - **System Audio (WASAPI):** captures meetings (Zoom, Meet, Teams), lectures, browser audio, or videos.
  - **Microphone:** captures your default Windows microphone.
  - **Both (System + Microphone):** mixes both streams concurrently into a unified 16 kHz mono pipeline.
- **Stable History & Protected Live Draft:**
  - Confirmed text stays in an independent timestamped panel with **smart scrolling** (auto-follows at the bottom, never jumps while reading or selecting earlier text).
  - The **Live** draft panel defers visual updates while you select text to copy.
  - Quick actions and shortcuts: *Copy selection*, *Copy last block*, *Copy current draft*, *Copy recent utterance*, search (`Ctrl+F`), and *Jump to latest* (`Ctrl+L`).
- **Custom Vocabulary (`initial_prompt`):** Optional context field to guide Whisper with technical terms, acronyms, or domain keywords (e.g., `Docker, Kubernetes, FastAPI, PyTorch...`).
- **Multi-Format Export:** Export confirmed history to **`.txt`** (plain text with timestamps), **`.md`** (structured Markdown), or **`.srt`** (standard SubRip subtitles).
- **Incremental Session Backup:** Every confirmed block is immediately appended and synced (`fsync`) to `sessions/session_YYYYMMDD_HHMMSS.txt` to prevent transcript loss on unexpected exits.
- **Persistent Settings:** Automatically saves and restores model profile, silence threshold, audio source, **Always on top** state, and custom vocabulary in `config.json`.
- **Built-in Diagnostics:** Toggleable panel with live metrics (inference duration, audio duration, RTF, lag, and backlog) plus rotating logs in `logs/diagnostics.log` that never store audio or transcribed text.

---

## Requirements

- **Operating System:** Windows 10 or 11 (64-bit).
- **Python:** Python **3.12** (64-bit) with the Windows `py` launcher enabled.
- **Internet Connection (first run only):** Required to install dependencies and download the chosen `faster-whisper` model weights. Once cached, LocalScribe runs completely offline.

---

## Installation

### Recommended (Automated Setup)

1. Clone or download this repository.
2. Run **`INSTALL.bat`** once. It verifies Python 3.12, creates the `.venv` virtual environment, and installs all dependencies from `requirements.txt`.

### Manual Installation (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Usage

Double-click **`START.bat`** or run from PowerShell:

```powershell
.\.venv\Scripts\python.exe main.py
```

1. Choose your **audio source** (*System Audio (WASAPI)*, *Microphone*, or *Both*).
2. Pick a **model profile** (*Fast `tiny`*, *Balanced `base`*, or *Accurate `small`*).
3. Optionally adjust the voice detection **Threshold** and **Optional vocabulary**.
4. Click **▶ Start** to begin live transcription.

---

## Project Structure

```text
LocalScribe/
├── INSTALL.bat              # Automated Windows virtual environment installer
├── START.bat                # Windows application launcher
├── main.py                  # Application entry point
├── requirements.txt         # Runtime dependencies
├── ruff.toml                # Linter and formatter configuration
├── docs/
│   └── screenshot.png       # Application UI screenshot
├── localscribe/
│   ├── __init__.py          # Package metadata and version (v0.3.0)
│   ├── audio.py             # WASAPI loopback, microphone, and dual-stream mixer
│   ├── config.py            # Automatic config.json persistence
│   ├── pipeline.py          # Energy/pause segmenter and ASR engine
│   ├── session.py           # Thread orchestration and incremental session backups
│   ├── transcript.py        # Recent transcript state and .txt/.md/.srt exporters
│   └── ui.py                # PySide6 desktop interface
└── tests/
    ├── test_pipeline.py     # Segmentation and recognition queue tests
    ├── test_v02.py          # Concurrency, copy state, and Qt widget regression tests
    └── test_v03.py          # Config persistence, session backups, export, and audio tests
```

---

## Development & Testing

Automated tests run headlessly using Qt's `offscreen` platform without opening real audio devices or downloading models:

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\python.exe -m pytest -v
```

---

## Privacy & Responsible Use

All audio processing runs strictly on your machine and is never transmitted to external APIs. Incremental session backups (`sessions/`), diagnostic logs (`logs/`), and local preferences (`config.json`) remain on your device and are excluded from version control via `.gitignore`. Always respect participant consent and institutional policies when transcribing meetings or classes.
