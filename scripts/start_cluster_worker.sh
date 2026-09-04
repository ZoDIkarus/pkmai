#!/usr/bin/env bash
set -euo pipefail

: "${RAY_ADDRESS:?set RAY_ADDRESS, e.g. 10.10.15.113:6379}"
: "${RAY_NODE_IP:?set RAY_NODE_IP to this worker VPN IP}"
trap 'ray stop --force || true' EXIT INT TERM
ray start --address="$RAY_ADDRESS" --node-ip-address="$RAY_NODE_IP" \
  --resources='{"pkmai_rollout": 1}'
exec python -u src/cluster_worker.py
