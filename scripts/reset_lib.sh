#!/bin/bash
# Shared safety helpers for the user-facing reset scripts.
set -euo pipefail

RESET_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESET_PY="${PKMAI_PYTHON:-/opt/homebrew/Caskroom/miniforge/base/envs/pokemon-ai/bin/python}"

reset_dry_run=false
reset_yes=false
for reset_arg in "$@"; do
  case "$reset_arg" in
    --dry-run) reset_dry_run=true ;;
    --yes) reset_yes=true ;;
    *) echo "Unbekannte Option: $reset_arg" >&2; exit 2 ;;
  esac
done

reset_stack_is_running() {
  pgrep -f '[s]rc/train.py|[s]rc/battle_train.py|[t]ools/battle_mirror_watch.py|[s]rc/watch.py|[s]rc/web_stream.py|[t]ools/pkmai_status.py' >/dev/null 2>&1
}

reset_stop_stack_if_needed() {
  if ! reset_stack_is_running; then
    return
  fi
  if [[ "$reset_yes" != true ]]; then
    printf 'PKMai laeuft noch. Jetzt sauber stoppen und Terminalfenster schliessen? [j/N] '
    read -r reset_answer
    [[ "$reset_answer" == "j" || "$reset_answer" == "J" ]] || {
      echo "Abgebrochen. Zuerst: bash scripts/stop_all.sh"
      exit 1
    }
  fi
  bash "$RESET_PROJECT/scripts/stop_all.sh"
}

reset_confirm() {
  local prompt="$1"
  if [[ "$reset_yes" == true ]]; then
    return
  fi
  printf '%s [j/N] ' "$prompt"
  read -r reset_answer
  [[ "$reset_answer" == "j" || "$reset_answer" == "J" ]] || {
    echo "Abgebrochen; nichts wurde veraendert."
    exit 0
  }
}

reset_run_scope() {
  local scope="$1"
  cd "$RESET_PROJECT"
  if [[ "$reset_dry_run" == true ]]; then
    PYTHONPATH=src "$RESET_PY" tools/reset_component.py --scope "$scope"
    return
  fi
  reset_stop_stack_if_needed
  PYTHONPATH=src "$RESET_PY" tools/reset_component.py --scope "$scope" --apply
  echo
  echo "Reset abgeschlossen. Neustart mit: bash scripts/start_all.sh"
}
