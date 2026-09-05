# PKMAI container deployment

All containers use normal Docker networking. PKMAI's network-facing services bind to host ports on the private VPN interface; do not publish them as `0.0.0.0`.

## Containers and host ports

| Container | Role | Host port(s) |
|---|---|---|
| `pkmai-web` | Dashboard | `8001 -> 8000` |
| `pkmai-cluster-master` | Authenticated worker registry | `8765` |
| `pkmai-cluster-brain` | Ray head / PPO learner | `6379`, `8265`, `10001-10003`, `11000-11100` |
| `pkmai-cluster-worker` | Ray rollout node | `10001-10003`, `11000-11100` on the worker host |
| `pkmai-trainer` | Legacy local SB3 trainer | no published port |

## VPS / Linux brain host

Use `compose.yaml`. Replace the sample VPN host IP if your site uses another address.

```bash
git checkout sascha
git pull --ff-only origin sascha
docker compose --profile trainer build trainer
docker compose --profile web up -d web
docker compose --profile cluster up -d cluster-master cluster-brain
```

Dashboard: `http://10.10.15.1:8001/`

Firewall: allow the Ray and master ports only from trusted VPN subnets. Do not expose them through a public interface.

## VPS / Linux remote worker

Use `compose.remote-worker.yaml` and an untracked `.worker.env`:

```text
PKMAI_RAY_ADDRESS=10.10.15.1:6379
PKMAI_WORKER_HOST_IP=<worker VPN host IP>
PKMAI_CLUSTER_MASTER_URL=http://10.10.15.1:8765
PKMAI_WORKER_ID=<unique worker name>
PKMAI_WORKER_AGENTS=1
```

Place the local ROM integration and `cluster_key.txt` in `local/`, then:

```bash
docker compose --env-file .worker.env -f compose.remote-worker.yaml build worker
docker compose --env-file .worker.env -f compose.remote-worker.yaml up -d worker
```

## Windows worker

Use Docker Desktop with the WSL2 backend and run the Linux-worker commands in an Ubuntu WSL terminal. Configure the Windows/WSL VPN so the worker host can reach the brain host ports above. Keep `.worker.env`, ROM and key local; do not commit them.

## macOS worker

Use Docker Desktop for Mac. Ensure the host VPN routes reach the brain host. Create the same `.worker.env`, provide local ROM/key files, then run the Linux-worker compose commands from Terminal. If the Docker Desktop VM cannot reach the VPN interface, use a Linux/WSL worker instead.

## Operations

```bash
# Status
docker ps --filter name=pkmai-
# Brain logs
docker logs -f pkmai-cluster-brain
# Worker logs
docker logs -f pkmai-cluster-worker
# Stop only the cluster
docker compose --profile cluster down
```

The brain is the sole writer for `runtime/cluster/`; workers never copy or write brain checkpoints.
