# Local Docker deployment

This project is deployed only on the local Windows/Docker Desktop host.

## Services

`compose.yaml` defines:

- `cluster-master` — local rollout coordination;
- `cluster-brain` — dynamic policy learning and best-policy publication;
- `web` — watcher-first dashboard on the same configurable host and container port (`8001` by default);
- `dynamic-watcher` (watcher profile) — non-training emulator stream.

Start the core services. The explicit profiles are required because all application services are profile-gated:

```bash
docker compose --profile cluster --profile web up -d cluster-master cluster-brain web
docker compose --profile watcher up -d dynamic-watcher
```

Start exactly ten explicit trainer containers:

```bash
./scripts/start_local_trainers.sh
```

Do not use Compose scaling for the trainer service with Docker Desktop host networking. The explicit launcher supplies distinct IDs/ranks and avoids shared-hostname registration collisions.

## Dashboard

Set `PKMAI_WEB_HOST` to the intended LAN/VPN interface for Docker's host-side bind and `PKMAI_WEB_PORT` to `8001`. The web application binds inside the container to `PKMAI_WEB_BIND_HOST` (default `0.0.0.0`) and listens on `PKMAI_WEB_PORT`.

- Dashboard: `http://<host>:8001/`
- Observer page: `http://<host>:8001/watcher-observer`
- Compatibility redirect: `http://<host>:8001/watcher` → `/`
- JPEG stream: `http://<host>:8001/watcher.jpg`
- Watcher API: `http://<host>:8001/api/watchers`

The start page renders the watcher list and automatically selects the first watcher as the live preview. The watcher writes status metadata without private model paths. The brain's `dynamic_policy_best.pt` is currently reward-based; it is not yet independently evaluated.

## Required verification

After any worker, brain or watcher deployment:

1. before restarting `cluster-brain`, preserve `runtime/cluster/dynamic_policy.pt` and verify the restarted brain restores it;
2. confirm all ten uniquely named trainer containers are online and registered;
3. confirm the learner serves an increasing policy version;
4. compare several sequential watcher JPEG hashes and inspect selected watcher actions;
5. verify dashboard and JPEG availability through the configured external interface.

A running container or an HTTP 200 alone is not sufficient evidence that the watcher is controlling the emulator.