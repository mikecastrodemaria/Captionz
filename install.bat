@echo off
setlocal
cd /d "%~dp0"
echo === Captionz : installation ===

rem --- Trouver un Python complet (venv + pip + tkinter). Le lanceur "py" est
rem --- prefere : "python" du PATH peut etre MSYS2 / Anaconda / Store stub.
set "PY="
for %%C in ("py -3" "python" "python3") do (
    if not defined PY (
        %%~C -c "import venv, ensurepip, tkinter" >nul 2>&1 && set "PY=%%~C"
    )
)
if not defined PY (
    echo [ERREUR] Aucun Python 3.10+ complet trouve ^(venv + pip + tkinter^).
    echo Installe Python depuis https://www.python.org/downloads/ en cochant "Add to PATH" et "tcl/tk and IDLE".
    pause
    exit /b 1
)
echo Python utilise : %PY%

if not exist ".venv\Scripts\python.exe" (
    echo Creation de l'environnement virtuel .venv ...
    if exist .venv rmdir /s /q .venv
    %PY% -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo [ERREUR] creation du venv impossible.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1
".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo [ERREUR] installation des dependances. & pause & exit /b 1)
".venv\Scripts\python.exe" -c "import tkinter, PIL; print('OK : tkinter + Pillow', PIL.__version__)"
echo.
echo Installation terminee. Lance start.bat
pause
