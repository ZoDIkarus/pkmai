#!/bin/bash
set -euo pipefail

PROJECT="$HOME/pokemon_ai_project"
PY="/opt/homebrew/Caskroom/miniforge/base/envs/pokemon-ai/bin/python"

# Cloudflare und Mapper sind bewusst standardmaessig AUS.
NO_CLOUDFLARE=true
NO_MAPPER=true
for arg in "$@"; do
  case "$arg" in
    --no-cloudflare) NO_CLOUDFLARE=true ;;
    --cloudflare)    NO_CLOUDFLARE=false ;;
    --no-mapper)     NO_MAPPER=true ;;
    --mapper)        NO_MAPPER=false ;;
    *) echo "Unbekannte Option: $arg" >&2; exit 2 ;;
  esac
done

cd "$PROJECT"

if [ -f "$PROJECT/.env" ]; then
  set -a
  source "$PROJECT/.env"
  set +a
fi

echo "=== PKMAI start_all ==="
echo "Projekt: $PROJECT"
echo

TMUX_BIN="$(command -v tmux || true)"
[ -x "/opt/homebrew/bin/tmux" ] && TMUX_BIN="/opt/homebrew/bin/tmux"
SESSION="pkmai"

start_window() {
  # Eigenes Terminal-Fenster (fuer den Watcher).
  local title="$1" command="$2"
  osascript <<EOF
tell application "Terminal"
    activate
    do script "printf '\\\\e]0;${title}\\\\a\\\\e]1;${title}\\\\a'; ${command}; exit"
end tell
EOF
}

# ------------------------------------------------------------------
# Sammelfenster: EIN Fenster, mehrere Tabs.
#   * mit tmux  -> echtes ein-Fenster-mit-Tabs (Ctrl-b + Zahl zum Wechseln,
#                  Ctrl-b d zum Loesen ohne zu stoppen)
#   * ohne tmux -> Fallback: je ein eigenes Terminal-Fenster
# ------------------------------------------------------------------
USE_TMUX=false
if [ -n "$TMUX_BIN" ]; then
  USE_TMUX=true
  if ! "$TMUX_BIN" has-session -t "$SESSION" 2>/dev/null; then
    "$TMUX_BIN" new-session -d -s "$SESSION" -n TRAIN -x 220 -y 50
    "$TMUX_BIN" set-option -t "$SESSION" -g mouse on
    "$TMUX_BIN" set-option -t "$SESSION" -g status-style "bg=colour24,fg=white"
  fi
fi

SHARED_OPENED_NONTMUX=false
shared_run() {
  # $1 = tmux-Fenstername / Fenstertitel, $2 = Kommando
  local name="$1" command="$2"
  if [ "$USE_TMUX" = true ]; then
    if [ "$name" = "TRAIN" ]; then
      "$TMUX_BIN" send-keys -t "${SESSION}:TRAIN" "cd '$PROJECT' && $command" C-m
    else
      "$TMUX_BIN" new-window -t "$SESSION" -n "$name" "cd '$PROJECT' && $command; exec \$SHELL"
    fi
  else
    if [ "$SHARED_OPENED_NONTMUX" = false ]; then
      start_window "PKMAI $name" "cd '$PROJECT' && $command"
      SHARED_OPENED_NONTMUX=true
    else
      start_window "PKMAI $name" "cd '$PROJECT' && $command"
    fi
  fi
}

find_bin() {
  local name="$1"; shift
  local found
  found="$(command -v "$name" || true)"
  if [ -n "$found" ]; then echo "$found"; return; fi
  for p in "$@"; do
    if [ -x "$p" ]; then echo "$p"; return; fi
  done
}

# ---------------- TRAINER (Sammelfenster, Tab TRAIN) ----------------
if pgrep -f "[s]rc/train.py" >/dev/null 2>&1; then
  echo "✓ Trainer läuft bereits"
else
  echo "▶ Starte Trainer"
  shared_run "TRAIN" "'$PY' src/train.py"
fi

# ---------------- WEBSERVER (immer sichtbares eigenes Terminal) -----
if pgrep -f "[s]rc/web_stream.py" >/dev/null 2>&1; then
  echo "✓ Webserver läuft bereits"
else
  echo "▶ Starte Webserver"
  start_window "PKMAI WEB" "cd '$PROJECT' && '$PY' -u src/web_stream.py"
fi

# ---------------- STATUS-MONITOR (Sammelfenster, Tab STATUS) --------
if pgrep -f "[t]ools/pkmai_status.py" >/dev/null 2>&1; then
  echo "✓ Status-Monitor läuft bereits"
else
  echo "▶ Starte Status-Monitor"
  shared_run "STATUS" "'$PY' tools/pkmai_status.py"
fi

# ---------------- MAPPER (nur wenn aktiviert) ----------------------
if [[ "$NO_MAPPER" == true ]]; then
  echo "✓ Mapper bleibt ausgeschaltet"
elif pgrep -f "[s]rc/mapper.py" >/dev/null 2>&1; then
  echo "✓ Mapper läuft bereits"
else
  echo "▶ Starte Mapper"
  shared_run "MAPPER" "'$PY' src/mapper.py"
fi

# ---------------- CLOUDFLARE (nur wenn aktiviert) -----------------
CF_BIN="$(find_bin cloudflared /opt/homebrew/bin/cloudflared /usr/local/bin/cloudflared)"
if [[ "$NO_CLOUDFLARE" == true ]]; then
  echo "✓ Cloudflare unverändert (lokaler Neustartmodus)"
elif pgrep -f "cloudflared tunnel" >/dev/null 2>&1; then
  echo "✓ cloudflared läuft bereits"
elif [ -n "$CF_BIN" ]; then
  echo "▶ Starte cloudflared"
  rm -f "$PROJECT/runtime/cloudflare.log"
  shared_run "CLOUDFLARE" "'$CF_BIN' tunnel --url http://localhost:8001 --logfile '$PROJECT/runtime/cloudflare.log' --loglevel info"
else
  echo "⚠ cloudflared nicht gefunden (brew install cloudflared)"
fi

# ---------------- WATCHER (immer eigenes Fenster) -----------------
if pgrep -f "[s]rc/watch.py" >/dev/null 2>&1; then
  echo "✓ Watcher läuft bereits"
else
  echo "▶ Starte Watcher  (eigenes Fenster)"
  start_window "PKMAI WATCHER" "cd '$PROJECT' && '$PY' src/watch.py"
fi

echo "✓ ngrok deaktiviert"

# Sammelfenster sichtbar machen (tmux attach in EINEM Terminal-Fenster).
if [ "$USE_TMUX" = true ]; then
  "$TMUX_BIN" select-window -t "${SESSION}:TRAIN" 2>/dev/null || true
  osascript <<EOF
tell application "Terminal"
    activate
    do script "printf '\\\\e]0;PKMAI SAMMELFENSTER\\\\a'; '${TMUX_BIN}' attach -t ${SESSION}"
end tell
EOF
fi

echo
echo "=== Status ==="
if [ "$USE_TMUX" = true ]; then
  echo "Fenster:  1x WATCHER separat  +  1x Sammelfenster (tmux '$SESSION')"
  echo "  Tab wechseln:  Ctrl-b dann 0/1/2 …   |   Liste: Ctrl-b w   |   loesen (laeuft weiter): Ctrl-b d"
  echo "  Erneut ansehen:  tmux attach -t $SESSION"
else
  echo "Fenster:  1x WATCHER separat  +  je 1 Fenster fuer Trainer/Web/Status"
  echo "  (Fuer EIN Sammelfenster mit Tabs:  brew install tmux  und start_all.sh erneut)"
fi
echo
echo "Zugriff (immer http://, NICHT https://):"
echo "  lokal   -> http://localhost:8001"
echo "  LAN     -> http://192.168.178.63:8001"
echo "  extern  -> http://nwfrdrt6qiykkk7m.myfritz.net:8001   (FRITZ!Box-Portfreigabe 8001 noetig)"
