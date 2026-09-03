#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE="$PROJECT/runtime/train.pid"

if [[ ! -f "$PIDFILE" ]]; then
  printf 'Kein Trainer-PID-File vorhanden.\n'
  exit 0
fi

pid="$(<"$PIDFILE")"
if ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$PIDFILE"
  printf 'Trainer lief nicht mehr; veraltete PID entfernt.\n'
  exit 0
fi

kill -TERM "$pid"
printf 'SIGTERM an Trainer (PID %s) gesendet. Der letzte zyklische Checkpoint bleibt erhalten.\n' "$pid"
rm -f "$PIDFILE"