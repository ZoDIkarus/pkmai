#!/bin/bash
# Start the complete live 2x2 stack in visible macOS Terminal windows.
# Nothing is detached: Ctrl-C in a service window stops that service.
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PKMAI_PYTHON:-/opt/homebrew/Caskroom/miniforge/base/envs/pokemon-ai/bin/python}"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: bash scripts/start_all.sh [--dry-run]" >&2
  exit 2
fi

if [[ ! -x "$PY" ]]; then
  echo "Python nicht gefunden oder nicht ausführbar: $PY" >&2
  exit 1
fi

running_pids() {
  pgrep -f "$1" 2>/dev/null | paste -sd, - || true
}

open_service() {
  local title="$1"
  local pattern="$2"
  local command="$3"
  local pids
  pids="$(running_pids "$pattern")"
  if [[ -n "$pids" ]]; then
    echo "✓ $title läuft bereits (PID $pids) — kein Duplikat gestartet"
    return 0
  fi

  if [[ "$DRY_RUN" == true ]]; then
    echo "• würde starten: $title — $command"
    return 0
  fi

  # Give every tab a stable, human-readable title. The process remains in the
  # foreground so the user can stop it directly with Ctrl-C.
  osascript <<APPLESCRIPT >/dev/null
tell application "Terminal"
  activate
  set serviceTab to do script "echo '=== ${title} ==='; cd '${PROJECT}'; if [ -f .env ]; then set -a; source .env; set +a; fi; export PYTHONPATH=src; export PKMAI_TWOBY2_LIVE=1; ${command}"
  set custom title of serviceTab to "${title}"
end tell
APPLESCRIPT
  echo "▶ $title in sichtbarem Terminal gestartet"
}

echo "PKMai 2×2 — sichtbarer Komplettstart"
echo "Bereits laufende Dienste werden nicht doppelt gestartet."

open_service "PKMai - Navigation" "[s]rc/train.py" \
  "'${PY}' -u src/train.py"
open_service "PKMai - Battler" "[s]rc/battle_train.py" \
  "'${PY}' -u src/battle_train.py --workers 9"
open_service "PKMai - Battle Watcher" "[t]ools/battle_mirror_watch.py" \
  "'${PY}' -u tools/battle_mirror_watch.py"
open_service "PKMai - Watcher" "[s]rc/watch.py" \
  "'${PY}' -u src/watch.py"
open_service "PKMai - Web" "[s]rc/web_stream.py" \
  "'${PY}' -u src/web_stream.py"
open_service "PKMai - Status" "[t]ools/pkmai_status.py" \
  "'${PY}' -u tools/pkmai_status.py -n 1"

echo
echo "Fertig: 40 FULL-Navigation · 9 Battle-Fighter · 2 Watcher · Web · Status"
echo "Dashboard: http://localhost:8001"
echo "Jedes Fenster lässt sich separat mit Ctrl-C beenden."
