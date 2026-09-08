#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/reset_lib.sh" "$@"

if [[ "$reset_dry_run" != true ]]; then
  reset_confirm "Map, Exploration und globale Freischaltungen wirklich zuruecksetzen? Modelle und Savestates bleiben erhalten."
fi
reset_run_scope map-global
