# Hardware bring-up: DJI Mini 3 app to the flight server on wolfgang

The order below proves one link at a time. Do not skip to a later stage if an earlier one fails.
Stages 1 to 4 need no flying. Keep the files listed under "Keep" for each stage; they are paper artefacts.

## Pieces

```
Mini 3  --video-->  RC-N1 + phone (MSDKRemote app, "Probe flight" screen)
                         |  POST /act: JPEG + telemetry, once a second
                         v
                    wolfgang:8765  scripts/80_flight_server.py (in the container, A100)
                         |  {"action": climb|hold|descend, "vertical_mps": ...}
                         v
                    phone -> virtual stick, vertical axis only, inside the envelope
```

## Network path (pick the first that works)

A. Phone on the university Wi-Fi: use `http://wolfgang.ise.canberra.edu.au:8765` directly.
B. Phone on mobile data with the university VPN app connected: same address.
C. Laptop relay, when the phone cannot reach wolfgang but the laptop can. Put the laptop on the phone's hotspot
   and run `ssh -N -g -L 8765:localhost:8765 adaml@wolfgang.ise.canberra.edu.au`. In the app use
   `http://<laptop address on the hotspot>:8765` (`ip addr` on the laptop shows it).

Quick check for any of them: open `http://<address>:8765/health` in the phone's browser. JSON back means the
path works.

## Stage 1: server alone (wolfgang)

```
cd ~/drone-ego/drone-loc-probing && git pull
docker/run_server.sh --dry-run
# second terminal on wolfgang
curl -s localhost:8765/health
```

Pass: JSON with `"model": "dry-run"`, an `envelope` block and `server_time`.

## Stage 2: phone reaches the server

Server still on `--dry-run`. Phone browser, then the app's "Test connection" button.
Pass: the app prints the envelope, a round trip time and a clock offset.
If it fails in the browser too, it is the network path, not the app. Try the next option above.

## Stage 3: bench loop, no model (drone on, on the table, not flying)

```
docker/run_server.sh --policy pattern --config configs/flight_indoor.yaml --save-frames --session bench01
```

App: Indoor, stick commands OFF, Start. Let it run a minute, Stop.
Pass: one line a second in the app, the action cycling climb, hold, descend, hold; `flights/bench01/commands.csv`
has one row per line with altitude, heading and gimbal pitch filled in; `flights/bench01/frames/` has the images.
Turn the drone by hand and tilt the camera with the gimbal wheel on the controller (the gimbal resists being
pushed by hand), then check the logged heading and pitch follow.
Keep: `flights/bench01/` (it has `session.json` with the config and git hash).

## Stage 4: bench loop with Qwen (latency)

```
docker/run_server.sh --model qwen2.5-vl-7b --config configs/flight_indoor.yaml --save-frames --session bench02
```

Same app settings. Pass: round trip comfortably under 1 s (the `rtt` in each app line; `server_ms` and `frame_ms`
in `commands.csv`). If it is over 1 s, note where the time goes (server or network) before going further.
Keep: `flights/bench02/`. This is the latency number for the paper, measured on each network path you will use.

## Stage 5: indoor flight, pattern policy (first time the server moves the drone)

Before starting: set `ceiling_m` in `configs/flight_indoor.yaml` at least 1 m below the lowest thing above the
drone, propeller guards if you have them, nobody near the drone.

```
docker/run_server.sh --policy pattern --config configs/flight_indoor.yaml --save-frames --session indoor01
```

Take off by hand and hover at about 1.5 m. App: Indoor, stick commands ON, Start.
Pass, in this order:
1. The drone climbs slowly for about 4 s, holds, descends, holds, and never passes the floor or ceiling.
2. Stop: the drone hovers and the sticks on the remote controller work again.
3. Start again, then move a stick or flick the flight mode switch on the remote controller. Write down what
   happens. This is how you will take over in the field, so know it before going outside.
4. Start again, then EMERGENCY LAND: the drone lands by itself.
If virtual stick will not enable indoors, write down the error text; some firmware wants a position fix first.
In that case go straight to the outdoor version of this stage.
Keep: `flights/indoor01/`, the phone CSV, and a line in `flights/indoor01/notes.yaml` on what the stick and mode
switch did.

## Stage 6: indoor altitude hold on the barometer (closed loop without the model)

```
docker/run_server.sh --policy baro --config configs/flight_indoor.yaml --save-frames --session indoor02
```

Start at about 1 m, then at about 2 m. Pass: the drone moves to 1.5 m and stays within about 0.5 m.
This is the reference for what the 1 Hz loop can do when the altitude estimate is perfect.

## Stage 7: outdoors

Same three policies with the default config (`configs/flight.yaml`: 15 to 60 m, 0.5 m/s, target 30 m), app on
Outdoor. Take off by hand and climb to about 20 m before pressing Start.

```
docker/run_server.sh --policy pattern --save-frames --session field01
docker/run_server.sh --policy baro    --save-frames --session field02
docker/run_server.sh --model qwen2.5-vl-7b --save-frames --session field03
```

`field03` is the unsteered Qwen altitude hold from the plan (within 3 m for 2 minutes). It is allowed to fail;
the log says how.

## What a session folder contains

| File | Content |
| --- | --- |
| `flights/<session>/session.json` | policy, full config, git hash, command line, start time |
| `flights/<session>/commands.csv` | one row per frame: server time, frame time, relative altitude, model text, action, vertical speed, probe readout, steering delta, server and frame latency, note, sequence number, heading, gimbal pitch, lat, lon, satellites, frame file, full telemetry JSON |
| `flights/<session>/frames/<seq>.jpg` | the image the model saw (not in git) |
| phone: `probe_<session>_<time>.csv` | the same loop seen from the phone, including the speed actually applied and the round trip time |
| `flights/<session>/notes.yaml` | yours: date, place, indoor or outdoor, wind, pilot, anything odd |

Because every frame is stored with altitude, heading, gimbal pitch and GPS, every session with `--save-frames`
is also labelled probing data.

## Optional: a shared secret

On a shared network, start the server with `FLIGHT_TOKEN=<something> docker/run_server.sh ...` and type the same
value into the app's token field. Without the right token `/act` and `/steer` return 401.

## Collecting ego-pose data and activations with the model in the loop

Add `--save-activations` to a model session and the server also keeps the model's residual stream for every
frame, recorded while it chooses the action:

```
docker/run_server.sh --model qwen2.5-vl-7b --config configs/flight_indoor.yaml --save-frames --save-activations --session indoor_obs01
```

Two kinds of session, both worth having:

- Observe: stick commands OFF in the app, you fly by hand. The model sees every frame and says what it would do,
  but nothing it says reaches the drone. Use this to cover the pose range: heights across the envelope, slow full
  turns, and the gimbal wheel from level to straight down.
- Closed loop: stick commands ON. The model holds the target height. This is the behaviour the causal tests need.

Afterwards:

```
docker/run.sh scripts/61_sessions_to_manifest.py --sessions indoor_obs01 indoor_qwen01 --site lab --out data/manifest_indoor.csv
docker/run.sh scripts/22_session_activations.py --manifest data/manifest_indoor.csv --model qwen2.5-vl-7b --out activations/
docker/run.sh scripts/30_train_probes.py --acts activations/qwen2.5-vl-7b-flight --targets altitude heading pitch --split within_scene --out results/
docker/run.sh scripts/31_baselines.py --manifest data/manifest_indoor.csv --acts activations/ --model qwen2.5-vl-7b-flight --split within_scene --probes_run results/<probes run_id> --out results/
```

With one site the split is within-site (held-out 20 s segments), so this measures whether the signal is there,
not whether it transfers to a new place.
