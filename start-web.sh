#!/usr/bin/env bash
# Captionz : NiceGUI web UI (Linux / macOS)
cd "$(dirname "$0")"
VPY=.venv/bin/python; [ -x "$VPY" ] || VPY=.venv/Scripts/python.exe
if [ ! -x "$VPY" ]; then
  echo "Environnement absent : lance d'abord ./install.sh" >&2
  exit 1
fi
exec "$VPY" app.py --ui web "$@"
