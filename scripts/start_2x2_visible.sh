#!/bin/bash
# Backwards-compatible name. The canonical starter is start_all.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/start_all.sh" "$@"
