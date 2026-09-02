#!/bin/bash
exec "$(cd "$(dirname "$0")" && pwd)/scripts/stop_all.sh" "$@"
