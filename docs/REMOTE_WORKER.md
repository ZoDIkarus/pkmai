# PKMAI remote Ray worker

Use this on a separate WireGuard-connected machine. The central brain runs on
the ai-server; workers provide emulator rollouts and never write the brain or
cluster checkpoints.

## Prerequisites

- Checkout the same `sascha` commit as the brain.
- Provide local, untracked files:
  - `local/custom_integrations/PokemonFireRed-Gba/rom.gba`
  - `local/custom_integrations/PokemonFireRed-Gba/rom.sha`
  - `local/cluster_key.txt`
- Ensure the worker host can route directly to the brain's VPN ports.
- On Windows, first prove that a Docker Desktop `network_mode: host` container
  can bind the advertised VPN IP. Docker Desktop usually exposes its internal
  VM IP instead; in that case use the native WSL procedure below rather than
  publishing an unreachable Docker-Desktop address to Ray.

Create a local, untracked `.worker.env` with the WireGuard addresses supplied
by the brain operator:

```text
PKMAI_RAY_ADDRESS=10.10.15.1:6379
PKMAI_WORKER_HOST_IP=<this worker's reachable VPN host IP>
PKMAI_CLUSTER_MASTER_URL=http://10.10.15.1:8765
PKMAI_WORKER_ID=<unique worker name>
PKMAI_WORKER_AGENTS=1
PKMAI_WORKER_CPUS=1.0
```

Build and start one constrained worker:

```bash
docker compose --env-file .worker.env -f compose.remote-worker.yaml build worker
docker compose --env-file .worker.env -f compose.remote-worker.yaml up -d worker
```

The default limit is one CPU and one emulator. The brain's
`PKMAI_CLUSTER_ENV_RUNNERS` determines the actual number of emulator
environments; `PKMAI_WORKER_AGENTS` is telemetry only. For a ten-emulator
worker, set the worker's `PKMAI_WORKER_AGENTS=10` and
`PKMAI_WORKER_CPUS=10.0`, and restart the brain with
`PKMAI_CLUSTER_ENV_RUNNERS=10` plus `PKMAI_CLUSTER_AGENTS_PER_RUNNER=1`.
Confirm registration from the brain host via the authenticated cluster registry
before increasing capacity.

## Windows + WSL fallback for a routable Ray node

Use this procedure only when Docker Desktop cannot bind the Windows/VPN IP.
It runs the worker natively in WSL2 while Windows forwards the fixed Ray ports
from the VPN interface. It was validated with this topology:

```text
Windows VPN address: 192.168.2.88
WSL address:         192.168.87.179
Ray brain:           10.10.15.1:6379
```

Addresses change per host and WSL restarts. Substitute the live addresses;
never copy private keys, ROMs, or `.worker.env` contents into commands or git.

### 1. Install the WSL runtime

In an elevated WSL shell, install the native libraries. Then create a virtual
environment and install the same Python dependencies as the worker image. Use
the CPU PyTorch index so this does not pull a CUDA runtime:

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential cmake pkg-config python3-dev python3-venv \
  libgl1 libglib2.0-0 libsdl2-2.0-0 libsdl2-dev libpng-dev zlib1g-dev iptables

python3 -m venv ~/.venvs/pkmai-worker
~/.venvs/pkmai-worker/bin/python -m pip install --upgrade pip
~/.venvs/pkmai-worker/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
grep -v '^torch$' /mnt/c/zod/pkmai2/requirements.txt >/tmp/pkmai-requirements.txt
~/.venvs/pkmai-worker/bin/pip install -r /tmp/pkmai-requirements.txt
```

Verify the runtime without printing configuration secrets:

```bash
~/.venvs/pkmai-worker/bin/python -c 'import ray, gymnasium, retro, torch; print(ray.__version__, gymnasium.__version__, torch.__version__)'
```

### 2. Forward the fixed Ray ports through Windows

Ray publishes its node manager, object manager, runtime-env agent, Ray Client
service, and workers on fixed ports. Save the following as a `.cmd` file, then
run it from **CMD as Administrator**. It configures forwarding and a firewall
rule scoped to the Windows VPN address:

```cmd
@echo off
setlocal EnableExtensions
set "VPN_IP=192.168.2.88"
set "WSL_IP=192.168.87.179"
for %%P in (10001 10002 10003 10004) do (
  netsh interface portproxy delete v4tov4 listenaddress=%VPN_IP% listenport=%%P
  netsh interface portproxy add v4tov4 listenaddress=%VPN_IP% listenport=%%P connectaddress=%WSL_IP% connectport=%%P protocol=tcp
)
for /L %%P in (11000,1,11100) do (
  netsh interface portproxy delete v4tov4 listenaddress=%VPN_IP% listenport=%%P
  netsh interface portproxy add v4tov4 listenaddress=%VPN_IP% listenport=%%P connectaddress=%WSL_IP% connectport=%%P protocol=tcp
)
netsh advfirewall firewall delete rule name="PKMAI Ray worker inbound"
netsh advfirewall firewall add rule name="PKMAI Ray worker inbound" dir=in action=allow protocol=TCP localip=%VPN_IP% localport=10001-10004,11000-11100 profile=any
netsh interface portproxy show v4tov4
```

`portproxy` needs administrator rights. Re-run this section after `wsl
--shutdown`, because the WSL IP can change. Restrict the firewall rule further
to known brain source addresses when the VPN topology permits it.

### 3. Bind and route the advertised VPN address inside WSL

Run as WSL root before each native worker start. The secondary address lets Ray
bind the address it advertises to the brain. The SNAT rule ensures Ray's GCS
connection can leave WSL through its actual WSL address while reply packets are
delivered back to the socket bound to the advertised VPN address.

```bash
VPN_IP=192.168.2.88
BRAIN_IP=10.10.15.1
WSL_IP="$(ip -4 -o addr show dev eth0 | awk 'NR == 1 {print $4}' | cut -d/ -f1)"
ip addr add "$VPN_IP/32" dev eth0 2>/dev/null || true
iptables -t nat -C POSTROUTING -s "$VPN_IP" -d "$BRAIN_IP" -p tcp --dport 6379 -j SNAT --to-source "$WSL_IP" 2>/dev/null \
  || iptables -t nat -I POSTROUTING -s "$VPN_IP" -d "$BRAIN_IP" -p tcp --dport 6379 -j SNAT --to-source "$WSL_IP"
```

The required tight connectivity check is a socket bound to the advertised
address. It must connect to the brain GCS before starting Ray:

```bash
python3 - <<'PY'
import socket
s = socket.socket(); s.settimeout(5); s.bind(("192.168.2.88", 0))
s.connect(("10.10.15.1", 6379)); print("bound-GCS-connect=ok", s.getsockname())
PY
```

### 4. Launch ten CPU/agent slots natively

Use `bash`, not `sh`, when importing `.worker.env`: the file is compose/bash
configuration and can contain syntax that `/bin/sh` does not accept. Keep the
key file on the local mounted drive and do not print it:

```bash
set -a
source /mnt/c/zod/pkmai2/.worker.env
set +a
export PATH="$HOME/.venvs/pkmai-worker/bin:$PATH"
export RAY_ADDRESS="$PKMAI_RAY_ADDRESS"
export RAY_NODE_IP=192.168.2.88
export PKMAI_CLUSTER_KEY_FILE=/mnt/c/zod/pkmai2/local/cluster_key.txt
export PKMAI_WORKER_ID=windows-wsl-10-runner
export PKMAI_WORKER_AGENTS=10
export PKMAI_WORKER_CPUS=10.0
cd /mnt/c/zod/pkmai2
bash scripts/start_cluster_worker.sh
```

### 5. Verify the complete worker chain

Do not treat a TCP connection or master heartbeat as successful emulator
capacity. Verify all of the following:

1. The WSL `raylet`, `runtime_env_agent`, and `cluster_worker.py` processes
   remain alive.
2. From the brain host, `ray status` shows the additional `10.0` CPU capacity.
3. The authenticated master health endpoint reports the worker online.
4. The brain's configured env-runner demand is at least ten and work is placed
   on the worker rather than merely queued.

If raw TCP connects but `ray status`, `/health`, or SSH hangs, the brain host
or its Docker/Ray stack is unhealthy. Repair or restart the brain first; do not
keep retrying worker startup against a non-responsive GCS.
