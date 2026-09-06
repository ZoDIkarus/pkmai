#!/bin/bash
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PKMAI_PYTHON:-/opt/homebrew/Caskroom/miniforge/base/envs/pokemon-ai/bin/python}"
cd "$PROJECT"
if [ -f "$PROJECT/.env" ]; then
    set -a
    source "$PROJECT/.env"
    set +a
fi
# Detached service sessions survive closing the calling terminal. No reset and
# no tunnel/mapper launch: local dashboard uses the existing port 8001.
"$PY" - <<'PY'
import os
from pathlib import Path
import shlex
import subprocess
import sys

root = Path.cwd()
logs = root / 'runtime' / 'logs'
logs.mkdir(parents=True, exist_ok=True)
processes = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True)
for name, script, args in (
    ('train', 'src/train.py', []),
    ('watch', 'src/watch.py', []),
    ('web', 'src/web_stream.py', []),
    ('status', 'tools/pkmai_status.py', ['-n', '60']),
):
    existing = []
    for line in processes.splitlines():
        try:
            pid, command = line.strip().split(None, 1)
            argv = shlex.split(command)
        except ValueError:
            continue
        if argv and 'python' in Path(argv[0]).name and any(
            arg in (script, str(root / script)) for arg in argv[1:]
        ):
            existing.append(pid)
    if existing:
        print(f'{name}: already running, PID {", ".join(existing)}', flush=True)
        continue
    with (logs / f'{name}.log').open('ab') as log:
        log.write(b'\n--- service start ---\n')
        child = subprocess.Popen(
            [sys.executable, '-u', str(root / script), *args], cwd=root,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (root / 'runtime' / f'{name}.pid').write_text(str(child.pid) + '\n')
    print(f'{name}: PID {child.pid}; log {logs / (name + ".log")}', flush=True)
print('Dashboard: http://localhost:8001')
PY
