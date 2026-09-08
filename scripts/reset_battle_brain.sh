#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/reset_lib.sh" "$@"

if [[ "$reset_dry_run" != true ]]; then
  reset_confirm "Battle Brain wirklich frisch zuruecksetzen? Alle gelernten v1/v2 Battle-PPOs werden geloescht; Szenarien bleiben erhalten."
fi
reset_run_scope battle-brain
