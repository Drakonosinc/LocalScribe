@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Comprobando Python...
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo Es necesario instalar Python 3.12 x64 y habilitar el comando py.
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
echo Instalacion completada. Usa INICIAR.bat para abrir.
pause
exit /b 0
:error
echo Ocurrio un error durante la instalacion.
pause
exit /b 1
