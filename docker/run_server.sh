#!/usr/bin/env bash
# Run the flight server in the container on the A100. Any arguments are passed to 80_flight_server.py.
#   docker/run_server.sh --dry-run
#   docker/run_server.sh --model qwen2.5-vl-7b
#   docker/run_server.sh --model qwen2.5-vl-7b --probes results/<run>/probes --steer altitude
set -euo pipefail
cd "$(dirname "$0")/.."
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
PORT="${PORT:-8765}"
mkdir -p "$HF_CACHE"
exec docker run --rm -it --gpus all \
  -p "${PORT}:8765" \
  -v "$(pwd):/work" \
  -v "${HF_CACHE}:/hf" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  --name drone-loc-server \
  drone-loc-probing \
  python scripts/80_flight_server.py --port 8765 "$@"
