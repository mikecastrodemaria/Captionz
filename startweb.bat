@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Environment not found. Run install.bat first.
    pause
    exit /b 1
)
echo Captionz - NiceGUI web interface (press Ctrl+C to stop)
.venv\Scripts\python.exe app.py --ui web %*
pause
