#!/usr/bin/env bash
set -euo pipefail

RAY_NODE_IP="${RAY_NODE_IP:?set RAY_NODE_IP to the brain container VPN IP}"
trap 'ray stop --force || true' EXIT INT TERM
ray start --head \
  --node-ip-address="$RAY_NODE_IP" \
  --port="${RAY_HEAD_PORT:-6379}" \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${RAY_DASHBOARD_PORT:-8265}" \
  --min-worker-port="${RAY_MIN_WORKER_PORT:-11000}" \
  --max-worker-port="${RAY_MAX_WORKER_PORT:-11100}"
exec python -u src/cluster_brain.py
