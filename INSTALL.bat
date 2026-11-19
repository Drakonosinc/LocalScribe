@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Checking Python installation...
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.12 x64 with the Windows py launcher is required.
    pause
    exit /b 1
)
py -3.12 -m venv .venv
if errorlevel 1 goto error
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error
echo.
echo Installation completed. Run START.bat to launch LocalScribe.
pause
exit /b 0
:error
echo An error occurred during installation.
pause
exit /b 1
