#!/bin/bash
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT"

for name in train watch web ngrok; do
  pidfile="runtime/${name}.pid"
  if [[ -f "$pidfile" ]]; then
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  fi
done

pkill -f "$PROJECT/src/train.py" || true
pkill -f "$PROJECT/src/watch.py" || true
pkill -f "$PROJECT/src/web_stream.py" || true

echo "PKMAI stopped."
