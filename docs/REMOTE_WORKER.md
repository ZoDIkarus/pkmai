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
- Ensure the worker host can route directly to the brain's VPN ports. The
  worker uses Docker host networking.

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
