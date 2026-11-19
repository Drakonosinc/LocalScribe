@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py
) else (
    echo Python virtual environment not found.
    echo Please run INSTALL.bat first.
    pause
)
