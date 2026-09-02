#!/usr/bin/env bash
# Captionz : installation (Linux / macOS)
set -e
cd "$(dirname "$0")"
echo "=== Captionz : installation ==="

# Trouver un Python complet (venv + pip + tkinter). Sous Windows/Git Bash le
# lanceur "py" est prefere : python3 du PATH peut etre celui de MSYS2.
PY=""
for c in "py -3" python3 python; do
  if $c -c "import venv, ensurepip, tkinter" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "[ERREUR] Aucun Python 3.10+ complet trouve (venv + pip + tkinter)." >&2
  echo "  Debian/Ubuntu : sudo apt install python3 python3-venv python3-tk" >&2
  echo "  Fedora        : sudo dnf install python3 python3-tkinter" >&2
  echo "  Arch          : sudo pacman -S python tk" >&2
  echo "  macOS (brew)  : brew install python python-tk" >&2
  exit 1
fi
echo "Python utilise : $PY"

# Linux/macOS : .venv/bin ; Git Bash sous Windows : .venv/Scripts
venv_py() { if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo .venv/Scripts/python.exe; fi; }
if [ ! -x "$(venv_py)" ]; then
  echo "Creation de l'environnement virtuel .venv ..."
  rm -rf .venv
  $PY -m venv .venv
fi
VPY=$(venv_py)
if [ ! -x "$VPY" ]; then echo "[ERREUR] creation du venv impossible." >&2; exit 1; fi
"$VPY" -m pip install --upgrade pip >/dev/null
"$VPY" -m pip install -r requirements.txt
"$VPY" -c "import tkinter, PIL; print('OK : tkinter + Pillow', PIL.__version__)"

echo
echo "Installation terminee. Lance ./start.sh"
