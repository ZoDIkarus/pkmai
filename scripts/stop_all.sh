#!/bin/bash
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT"

KEEP_CLOUDFLARE=false
for arg in "$@"; do
  case "$arg" in
    --keep-cloudflare) KEEP_CLOUDFLARE=true ;;
    *) echo "Unbekannte Option: $arg" >&2; exit 2 ;;
  esac
done

wait_until_pid_stopped() {
  local pid="$1"
  local attempts="$2"
  local i
  for ((i=0; i<attempts; i++)); do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.1
  done
  return 1
}

graceful_stop() {
  local label="$1"
  local pattern="$2"
  local attempts="$3"
  local pids
  pids="$(pgrep -f "$pattern" 2>/dev/null | sort -rn || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi
  # Moegliche Alt-Duplikate einzeln stoppen. Niedrige/alte PID zuletzt, damit
  # der am laengsten trainierte Learner als letzter sicher speichert.
  local pid
  for pid in $pids; do
    echo "Stoppe $label PID $pid sauber und warte auf Speichern ..."
    kill -INT "$pid" 2>/dev/null || true
    if wait_until_pid_stopped "$pid" "$attempts"; then
      echo "✓ $label PID $pid beendet"
      continue
    fi
    echo "⚠ $label PID $pid reagiert nicht; sende TERM und warte erneut"
    kill -TERM "$pid" 2>/dev/null || true
    wait_until_pid_stopped "$pid" 100 || true
  done
}

# Der Trainer bekommt zuerst Ctrl+C/SIGINT. train.py faengt das Signal ab,
# speichert Candidate + Resume und schliesst erst danach seine Worker.
graceful_stop "Trainer" "[s]rc/train.py" 450
# Auch Watcher/Status duerfen ihren aktuellen Schreibvorgang abschliessen.
graceful_stop "Watcher" "[s]rc/watch.py" 100
graceful_stop "Mapper" "[s]rc/mapper.py" 450
graceful_stop "Webserver" "[s]rc/web_stream.py" 100
graceful_stop "Status-Monitor" "[t]ools/pkmai_status.py" 50

for name in train watch mapper web ngrok; do
  rm -f "runtime/${name}.pid"
done

if [[ "$KEEP_CLOUDFLARE" != true ]]; then
  graceful_stop "Cloudflare" "cloudflared tunnel" 100
fi
graceful_stop "ngrok" "ngrok http 8000" 50
# verwaiste SubprocVecEnv-Worker vom Trainer
pkill -f "multiprocessing-fork" || true
pkill -f "multiprocessing.resource_tracker" || true

# TERM ist asynchron: insbesondere uvicorn kann den Listener noch einen
# Augenblick halten. Erst weitergehen, wenn der alte Webprozess und Port 8001
# wirklich frei sind, sonst kollidiert das neue Dashboard beim Binden.
for _ in {1..50}; do
  if ! pgrep -f "[s]rc/web_stream.py" >/dev/null 2>&1 \
      && ! lsof -nP -iTCP:8001 -sTCP:LISTEN >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done

# Erst die Prozesse beenden, dann deren alte Terminalfenster schliessen.
# Cloudflare gehoert absichtlich nicht zu dieser Liste.
if command -v osascript >/dev/null 2>&1; then
  osascript <<'APPLESCRIPT' >/dev/null 2>&1 || true
tell application "Terminal"
  set serviceTitles to {"PKMAI TRAIN", "PKMAI WATCHER", "PKMAI MAPPER", "PKMAI WEB", "PKMAI STATUS"}
  repeat with i from (count of windows) to 1 by -1
    try
      set windowName to name of window i
      repeat with serviceTitle in serviceTitles
        if windowName contains (serviceTitle as text) then
          -- Der Python-Prozess ist oben bereits nachweislich beendet. Jetzt
          -- die verbliebene Login-Shell sauber verlassen; dadurch erscheint
          -- kein macOS-Dialog zum Erzwingen eines laufenden Prozesses.
          do script "exit" in selected tab of window i
          delay 0.2
          -- Unabhaengig von der Terminal-Einstellung "Fenster nach Exit"
          -- das nun prozessfreie Service-Fenster wirklich schliessen.
          try
            close window i
          end try
          exit repeat
        end if
      end repeat
    end try
  end repeat
end tell
APPLESCRIPT
fi

if [[ "$KEEP_CLOUDFLARE" == true ]]; then
  echo "PKMAI stopped; Cloudflare läuft unverändert weiter."
else
  echo "PKMAI stopped."
fi
