#!/usr/bin/env bash
# Run any repo script inside the container, pinned to one GPU (default the A100 at index 1).
#   docker/run.sh scripts/20_extract_activations.py --model qwen2.5-vl-7b --manifest data/manifest_real.csv --limit 8
#   GPU=0 docker/run.sh scripts/30_train_probes.py --acts activations/qwen2.5-vl-7b      # A30, or CPU-only work
set -euo pipefail
cd "$(dirname "$0")/.."
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
GPU="${GPU:-1}"   # wolfgang: 0 = A30 24 GB, 1 = A100 40 GB (check with nvidia-smi -L)
mkdir -p "$HF_CACHE"
TTY=""; [ -t 0 ] && TTY="-it"
exec docker run --rm $TTY --gpus "device=${GPU}" \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$(pwd):/work" \
  -v "${HF_CACHE}:/hf" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  drone-loc-probing \
  python "$@"
