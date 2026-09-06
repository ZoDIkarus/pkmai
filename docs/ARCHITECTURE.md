# Local dynamic-cluster architecture

PKMAI runs entirely on the local Docker Desktop host. The topology has no remote Ray worker or Stable-Baselines service.

```text
local-trainer-0..9 -> cluster-master -> cluster-brain -> dynamic_policy(_best).pt
                                              |                    |
                                              +--------------------+
                                                                   v
                                                        dynamic-watcher
                                                                   |
                                                     runtime/watcher.jpg + watcher.json
                                                                   |
                                                                  web:8001
```

## Components

- `src/cluster_worker.py` generates rollout batches. Every local container receives an explicit `PKMAI_WORKER_ID`, `PKMAI_WORKER_RANK`, and fleet size.
- `src/cluster_master.py` performs registration, compatibility checks, batch intake, and serves the current policy to trainers.
- `src/dynamic_brain.py` learns from uploaded rollout batches and retains/publishes the highest-reward policy for viewing.
- `src/dynamic_watcher.py` uses its own non-training emulator. It starts every episode at the true initial game state, samples the published best-policy distribution, and publishes a JPEG plus sanitized watcher status.
- `src/web_stream.py` exposes the watcher-first dashboard, `/api/watchers`, `/api/cluster-status`, `/watcher.jpg`, and `/watcher`.

The visible watcher is deliberately separate from the trainer fleet and does not upload rollouts or mutate policy state.