# drone-loc-probing

Does the VLM flying a drone also know where the drone is? Probing Qwen2.5-VL, AeroVLA and an aerial-finetuned
variant for altitude, map-relative position and localisation uncertainty, with causal tests on real DJI Mini 3
flights. Plan: the "Drone Localisation Probing: Experiment Plan" doc.

## Layout

```
configs/        models.yaml, sites.yaml, probes.yaml, flight.yaml
scripts/        numbered scripts; common.py holds run-folder and provenance helpers
data/           manifest_real.csv, manifest.csv (sim slice), tiles/, dem/, frames/   (frames out of git)
flights/        <id>/*.SRT *.txt *.MP4 notes.yaml commands.csv                        (MP4s out of git)
activations/    <model>/<split>/layer_XX_<position>.npy                               (out of git)
results/        <run_id>/ ... every script writes here, never elsewhere
figures/        one file per paper figure
docker/         Dockerfile and run scripts for the A100
```

## Phase 0 quick start (laptop or A100, no Docker)

```
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision   # pick the CUDA tag for your driver
pip install -r requirements.txt
python scripts/00_check_env.py                      # environment only
python scripts/00_check_env.py --models qwen2.5-vl-7b aerovla   # loads each model, hooks it, times a forward pass
```

`00_check_env.py` prints one line per model (loaded, decoder layer count, forward-pass seconds) and writes
`results/<run_id>/env.lock` so the first working environment is pinned.

## Flight server (A100)

Two steps, so the network path is measured before the model is involved:

```
# 1. latency of the path only (no model)
python scripts/80_flight_server.py --dry-run --port 8765
python scripts/client_latency_test.py --server http://<a100-host>:8765 --image tests/sample.jpg --n 30   # from the laptop in the field

# 2. with the model
python scripts/80_flight_server.py --model qwen2.5-vl-7b --port 8765
```

Endpoints: `GET /health`, `POST /act` (multipart `image`, optional `instruction`, optional `telemetry` JSON),
`POST /steer` (`{"delta": 30.0}` shifts the probe readout by that much in its own units; `0` disables).
Every `/act` call is appended to `flights/<session>/commands.csv`.

Docker on the A100:

```
docker build -t drone-loc-probing -f docker/Dockerfile .
docker/run_server.sh --dry-run                      # path test
docker/run_server.sh --model qwen2.5-vl-7b          # real
```

The container mounts the repo at `/work` and the Hugging Face cache at `/hf`, so code edits and downloaded
weights never require a rebuild.

## Manifest columns (real and sim share them)

frame_id, site_or_scene, source (real|sim), split, flight_id, timestamp, x, y, altitude_agl, altitude_baro,
yaw, pitch, lat, lon, tile_id, tile_rotation, u, v, yaw_rel_tile, localisability, sun_azimuth, ground_class

## Rules every script follows

- Seeds on the command line, written to `results/<run_id>/seeds.txt`.
- Nothing written outside a run folder; figures read CSVs only.
- `common.new_run()` creates the run folder with config copy, git hash, seeds and env.lock.
