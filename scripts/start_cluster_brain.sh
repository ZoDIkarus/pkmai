#!/usr/bin/env bash
set -euo pipefail

RAY_NODE_IP="${RAY_NODE_IP:?set RAY_NODE_IP to the brain container VPN IP}"
trap 'ray stop --force || true' EXIT INT TERM
ray start --head \
  --node-ip-address="$RAY_NODE_IP" \
  --port="${RAY_HEAD_PORT:-6379}" \
  --node-manager-port="${RAY_NODE_MANAGER_PORT:-10001}" \
  --ray-client-server-port="${RAY_CLIENT_SERVER_PORT:-10004}" \
  --object-manager-port="${RAY_OBJECT_MANAGER_PORT:-10002}" \
  --runtime-env-agent-port="${RAY_RUNTIME_ENV_PORT:-10003}" \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${RAY_DASHBOARD_PORT:-8265}" \
  --min-worker-port="${RAY_MIN_WORKER_PORT:-11000}" \
  --max-worker-port="${RAY_MAX_WORKER_PORT:-11100}" \
  --disable-usage-stats
exec python -u src/cluster_brain.py
