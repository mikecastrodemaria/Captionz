@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\pythonw.exe (
    echo Environnement absent : lance d'abord install.bat
    pause
    exit /b 1
)
start "" .venv\Scripts\pythonw.exe app.py
