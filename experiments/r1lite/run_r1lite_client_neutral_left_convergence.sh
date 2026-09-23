#!/usr/bin/env bash
# Neutral-left A/B client with per-action right-arm convergence gating.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_r1lite_client_neutral_left.sh" \
  --wait-for-convergence \
  --settle-tolerance-deg 2.0 \
  --settle-timeout-s 0.25 \
  --settle-consecutive-samples 2 \
  "$@"
