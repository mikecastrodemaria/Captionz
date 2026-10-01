#!/usr/bin/env bash
# Captionz NiceGUI web UI (Linux / macOS). Press Ctrl+C to stop.
cd "$(dirname "$0")"
VPY=.venv/bin/python; [ -x "$VPY" ] || VPY=.venv/Scripts/python.exe
if [ ! -x "$VPY" ]; then
  echo "Environment not found. Run ./install.sh first." >&2
  exit 1
fi
exec "$VPY" app.py --ui web "$@"
