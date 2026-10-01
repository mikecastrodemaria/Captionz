#!/usr/bin/env bash
# Captionz installation (Linux / macOS)
set -e
cd "$(dirname "$0")"
echo "=== Captionz installation ==="

# Find a complete Python installation (venv + pip + tkinter). On Windows/Git Bash,
# prefer the "py" launcher because PATH's python3 may be the MSYS2 version.
PY=""
for c in "py -3" python3 python; do
  if $c -c "import venv, ensurepip, tkinter" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "[ERROR] No complete Python 3.10+ installation found (venv + pip + tkinter)." >&2
  echo "  Debian/Ubuntu : sudo apt install python3 python3-venv python3-tk" >&2
  echo "  Fedora        : sudo dnf install python3 python3-tkinter" >&2
  echo "  Arch          : sudo pacman -S python tk" >&2
  echo "  macOS (brew)  : brew install python python-tk" >&2
  exit 1
fi
echo "Using Python: $PY"

# Linux/macOS : .venv/bin ; Git Bash sous Windows : .venv/Scripts
venv_py() { if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo .venv/Scripts/python.exe; fi; }
if [ ! -x "$(venv_py)" ]; then
  echo "Creating virtual environment .venv..."
  rm -rf .venv
  $PY -m venv .venv
fi
VPY=$(venv_py)
if [ ! -x "$VPY" ]; then echo "[ERROR] Could not create the virtual environment." >&2; exit 1; fi
"$VPY" -m pip install --upgrade pip >/dev/null
"$VPY" -m pip install -r requirements.txt
"$VPY" -c "import tkinter, PIL; print('OK: tkinter + Pillow', PIL.__version__)"

echo
echo "Installation complete. Run ./start.sh"
