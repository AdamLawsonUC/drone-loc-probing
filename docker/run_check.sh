#!/usr/bin/env bash
# Run the Phase 0 environment check inside the container on the A100.
#   docker/run_check.sh                          # environment only
#   docker/run_check.sh --models qwen2.5-vl-7b   # also loads and times the model
set -euo pipefail
cd "$(dirname "$0")/.."
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"
exec docker run --rm -it --gpus all \
  -v "$(pwd):/work" \
  -v "${HF_CACHE}:/hf" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  drone-loc-probing \
  python scripts/00_check_env.py "$@"
