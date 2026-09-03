#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PKMAI_PYTHON:-/opt/pkmai-venv/bin/python}"
RUNTIME_DIR="$PROJECT/runtime"
PIDFILE="$RUNTIME_DIR/train.pid"
LOGFILE="$RUNTIME_DIR/train.log"

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'WSL-Python nicht gefunden: %s\n' "$PYTHON_BIN" >&2
  printf 'Erstelle ihn mit: python3 -m venv /opt/pkmai-venv\n' >&2
  exit 1
fi

if [[ -f "$PIDFILE" ]]; then
  pid="$(<"$PIDFILE")"
  if kill -0 "$pid" 2>/dev/null; then
    printf 'Trainer läuft bereits (PID %s). Log: %s\n' "$pid" "$LOGFILE"
    exit 0
  fi
  rm -f "$PIDFILE"
fi

"$PYTHON_BIN" -c 'import stable_retro, stable_baselines3, torch' >/dev/null
mkdir -p "$RUNTIME_DIR"

cd "$PROJECT"
nohup env PYTHONPATH="$PROJECT/src" "$PYTHON_BIN" "$PROJECT/src/train.py" \
  >>"$LOGFILE" 2>&1 < /dev/null &
pid="$!"
printf '%s\n' "$pid" > "$PIDFILE"

sleep 2
if kill -0 "$pid" 2>/dev/null; then
  printf 'Trainer im Hintergrund gestartet (PID %s).\nLog: %s\n' "$pid" "$LOGFILE"
else
  printf 'Trainer ist direkt beendet worden. Prüfe: %s\n' "$LOGFILE" >&2
  rm -f "$PIDFILE"
  exit 1
fi