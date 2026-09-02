#!/bin/bash
set -euo pipefail

PROJECT="$HOME/pokemon_ai_project"
PY="/opt/homebrew/Caskroom/miniforge/base/envs/pokemon-ai/bin/python"
NGROK="$(command -v ngrok || true)"

cd "$PROJECT"

# Lokale Secrets laden
if [ -f "$PROJECT/.env" ]; then
    set -a
    source "$PROJECT/.env"
    set +a
fi

echo "=== PKMAI START ==="

# TRAINER
if ! pgrep -f "$PROJECT/src/train.py" >/dev/null; then
    osascript -e "tell application \"Terminal\" to do script \"cd '$PROJECT' && '$PY' src/train.py\""
else
    echo "Trainer läuft bereits."
fi

sleep 1

# WATCHER
if ! pgrep -f "$PROJECT/src/watch.py" >/dev/null; then
    osascript -e "tell application \"Terminal\" to do script \"cd '$PROJECT' && '$PY' src/watch.py\""
else
    echo "Watcher läuft bereits."
fi

sleep 1

# WEB
if ! pgrep -f "$PROJECT/src/web_stream.py" >/dev/null; then
    osascript -e "tell application \"Terminal\" to do script \"cd '$PROJECT' && '$PY' src/web_stream.py\""
else
    echo "Web läuft bereits."
fi

sleep 2

# NGROK
if [ -z "$NGROK" ]; then
    echo "FEHLER: ngrok wurde nicht gefunden."
    echo "Prüfe mit: which ngrok"
else
    if ! pgrep -f "ngrok http 8000" >/dev/null; then

        # Falls Token in .env vorhanden ist, einmal lokal konfigurieren.
        if [ -n "${NGROK_AUTHTOKEN:-}" ]; then
            "$NGROK" config add-authtoken "$NGROK_AUTHTOKEN" >/dev/null 2>&1 || true
        fi

        osascript -e "tell application \"Terminal\" to do script \"cd '$PROJECT' && '$NGROK' http 8000\""
    else
        echo "ngrok läuft bereits."
    fi
fi

echo "=== PKMAI gestartet ==="
