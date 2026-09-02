#!/bin/bash
exec "$(cd "$(dirname "$0")" && pwd)/scripts/start_all.sh" "$@"
