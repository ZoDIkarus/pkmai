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

find_bin() {
  # $1 = Name, $2.. = Fallback-Pfade
  local name="$1"; shift
  local found
  found="$(command -v "$name" || true)"
  if [ -n "$found" ]; then echo "$found"; return; fi
  for p in "$@"; do
    if [ -x "$p" ]; then echo "$p"; return; fi
  done
}

# ------------------------------------------------------------------
# TRAINER
# ------------------------------------------------------------------
if pgrep -f "$PROJECT/src/train.py" >/dev/null 2>&1; then
  echo "✓ Trainer läuft bereits"
else
  echo "▶ Starte Trainer"
  start_terminal "PKMAI TRAIN" \
    "cd '$PROJECT' && '$PY' src/train.py"
fi

# ------------------------------------------------------------------
# WATCHER
# ------------------------------------------------------------------
if pgrep -f "$PROJECT/src/watch.py" >/dev/null 2>&1; then
  echo "✓ Watcher läuft bereits"
else
  echo "▶ Starte Watcher"
  start_terminal "PKMAI WATCHER" \
    "cd '$PROJECT' && '$PY' src/watch.py"
fi

# ------------------------------------------------------------------
# WEBSERVER (Dashboard auf :8000)
# ------------------------------------------------------------------
if pgrep -f "$PROJECT/src/web_stream.py" >/dev/null 2>&1; then
  echo "✓ Webserver läuft bereits"
else
  echo "▶ Starte Webserver"
  start_terminal "PKMAI WEB" \
    "cd '$PROJECT' && '$PY' src/web_stream.py"
fi

# ------------------------------------------------------------------
# STATUS-MONITOR (tools/pkmai_status.py, auto-refresh 5s)
# ------------------------------------------------------------------
if pgrep -f "tools/pkmai_status.py" >/dev/null 2>&1; then
  echo "✓ Status-Monitor läuft bereits"
else
  echo "▶ Starte Status-Monitor"
  start_terminal "PKMAI STATUS" \
    "cd '$PROJECT' && '$PY' tools/pkmai_status.py"
fi

# ------------------------------------------------------------------
# CLOUDFLARE TUNNEL (Quick Tunnel -> https://*.trycloudflare.com)
# ------------------------------------------------------------------
CF_BIN="$(find_bin cloudflared /opt/homebrew/bin/cloudflared /usr/local/bin/cloudflared)"

if pgrep -f "cloudflared tunnel" >/dev/null 2>&1; then
  echo "✓ cloudflared läuft bereits"
elif [ -n "$CF_BIN" ]; then
  echo "▶ Starte cloudflared  (URL steht im Fenster: https://<zufall>.trycloudflare.com)"
  start_terminal "PKMAI CLOUDFLARE" \
    "cd '$PROJECT' && '$CF_BIN' tunnel --url http://localhost:8000"
else
  echo "⚠ cloudflared nicht gefunden (brew install cloudflared)"
fi

# ------------------------------------------------------------------
# NGROK TUNNEL (https://*.ngrok-free.app / eigene Domain)
# ------------------------------------------------------------------
NGROK_BIN="$(find_bin ngrok /opt/homebrew/bin/ngrok /usr/local/bin/ngrok)"

if pgrep -f "ngrok http 8000" >/dev/null 2>&1; then
  echo "✓ ngrok läuft bereits"
elif [ -n "$NGROK_BIN" ]; then
  if [ -n "${NGROK_AUTHTOKEN:-}" ]; then
    "$NGROK_BIN" config add-authtoken "$NGROK_AUTHTOKEN" >/dev/null 2>&1 || true
  fi
  echo "▶ Starte ngrok  (URL + Web-Interface: http://localhost:4040)"
  start_terminal "PKMAI NGROK" \
    "cd '$PROJECT' && '$NGROK_BIN' http 8000"
else
  echo "⚠ ngrok nicht gefunden"
fi

echo
echo "=== Status ==="
echo "Brain: BEHALTEN   Statistik: BEHALTEN   Karte: BEHALTEN   Curriculum: BEHALTEN   Reset: NEIN"
echo
echo "Öffentliche URL:"
echo "  cloudflare -> steht im Fenster 'PKMAI CLOUDFLARE' (Zeile mit trycloudflare.com)"
echo "  ngrok      -> steht im Fenster 'PKMAI NGROK' oder auf http://localhost:4040"
