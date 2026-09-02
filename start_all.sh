#!/bin/bash
set -euo pipefail

PROJECT="$HOME/pokemon_ai_project"
PY="/opt/homebrew/Caskroom/miniforge/base/envs/pokemon-ai/bin/python"

cd "$PROJECT"

if [ -f "$PROJECT/.env" ]; then
  set -a
  source "$PROJECT/.env"
  set +a
fi

echo "=== PKMAI start_all ==="
echo "Projekt: $PROJECT"
echo

start_terminal() {
  local title="$1"
  local command="$2"

  osascript <<EOF
tell application "Terminal"
    activate
    do script "printf '\\\\e]1;${title}\\\\a'; ${command}"
end tell
EOF
}

if pgrep -f "$PROJECT/src/train.py" >/dev/null 2>&1; then
  echo "✓ Trainer läuft bereits"
else
  echo "▶ Starte Trainer"
  start_terminal "PKMAI TRAIN" \
    "cd '$PROJECT' && '$PY' src/train.py"
fi

if pgrep -f "$PROJECT/src/watch.py" >/dev/null 2>&1; then
  echo "✓ Watcher läuft bereits"
else
  echo "▶ Starte Watcher"
  start_terminal "PKMAI WATCHER" \
    "cd '$PROJECT' && '$PY' src/watch.py"
fi

if pgrep -f "$PROJECT/src/web_stream.py" >/dev/null 2>&1; then
  echo "✓ Webserver läuft bereits"
else
  echo "▶ Starte Webserver"
  start_terminal "PKMAI WEB" \
    "cd '$PROJECT' && '$PY' src/web_stream.py"
fi

NGROK_BIN="$(command -v ngrok || true)"

if [ -z "$NGROK_BIN" ]; then
  if [ -x "/opt/homebrew/bin/ngrok" ]; then
    NGROK_BIN="/opt/homebrew/bin/ngrok"
  elif [ -x "/usr/local/bin/ngrok" ]; then
    NGROK_BIN="/usr/local/bin/ngrok"
  fi
fi

if pgrep -f "ngrok http 8000" >/dev/null 2>&1; then
  echo "✓ ngrok läuft bereits"
elif [ -n "$NGROK_BIN" ]; then
  echo "▶ Starte ngrok"
  start_terminal "PKMAI NGROK" \
    "cd '$PROJECT' && '$NGROK_BIN' http 8000"
else
  echo "⚠ ngrok nicht gefunden"
fi

echo
echo "=== Status ==="
echo "Brain: BEHALTEN"
echo "Statistik: BEHALTEN"
echo "Karte: BEHALTEN"
echo "Curriculum: BEHALTEN"
echo "Reset: NEIN"
