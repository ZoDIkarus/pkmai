#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/reset_lib.sh" "$@"

if [[ "$reset_dry_run" != true ]]; then
  reset_confirm "Navigation Brain wirklich frisch zuruecksetzen? Champion und Learner werden geloescht; Maps und Savestates bleiben erhalten."
fi
reset_run_scope nav-brain
