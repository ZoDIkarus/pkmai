#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/reset_lib.sh" "$@"

if [[ "$reset_dry_run" == true ]]; then
  reset_run_scope full
  exit 0
fi

echo "ACHTUNG: Navigation Brain, Battle Brain, Maps, Exploration und globale Freischaltungen werden frisch zurueckgesetzt."
echo "Vorher wird ein vollstaendiges, pruefsummen-gesichertes Backup von runtime/ erstellt."
printf 'Wirklich das komplette Brain zuruecksetzen? Tippe exakt KOMPLETT RESET: '
read -r full_answer
if [[ "$full_answer" != "KOMPLETT RESET" ]]; then
  echo "Abgebrochen; nichts wurde veraendert."
  exit 0
fi

reset_stop_stack_if_needed
reset_run_scope full
