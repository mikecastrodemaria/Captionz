@echo off
setlocal
cd /d "%~dp0"
echo === Captionz installation ===

rem --- Find a complete Python installation (venv + pip + tkinter). Prefer the "py"
rem --- launcher because "python" on PATH may be MSYS2, Anaconda, or the Store stub.
set "PY="
for %%C in ("py -3" "python" "python3") do (
    if not defined PY (
        %%~C -c "import venv, ensurepip, tkinter" >nul 2>&1 && set "PY=%%~C"
    )
)
if not defined PY (
    echo [ERROR] No complete Python 3.10+ installation found ^(venv + pip + tkinter^).
    echo Install Python from https://www.python.org/downloads/ and select "Add to PATH" and "tcl/tk and IDLE".
    pause
    exit /b 1
)
echo Using Python: %PY%

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment .venv...
    if exist .venv rmdir /s /q .venv
    %PY% -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Could not create the virtual environment.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1
".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo [ERROR] Dependency installation failed. & pause & exit /b 1)
".venv\Scripts\python.exe" -c "import tkinter, PIL; print('OK: tkinter + Pillow', PIL.__version__)"
echo.
echo Installation complete. Run start.bat
pause
