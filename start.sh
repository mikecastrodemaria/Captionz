#!/usr/bin/env bash
# Captionz startup (Linux / macOS)
cd "$(dirname "$0")"
VPY=.venv/bin/python; [ -x "$VPY" ] || VPY=.venv/Scripts/python.exe
if [ ! -x "$VPY" ]; then
  echo "Environment not found. Run ./install.sh first." >&2
  exit 1
fi
exec "$VPY" app.py "$@"
