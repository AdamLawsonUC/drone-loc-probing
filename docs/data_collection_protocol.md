# Data collection protocol: flights where the model flies

Version 2, 19 Nov 2026. For the DJI Mini 3 with the "Probe flight" screen and the flight server on wolfgang.
Version 1 had the pilot flying while the model watched; that is now only a fallback (section 9).

Principle: the model is in control whenever the drone is in the air and nothing is wrong. Qwen sees the camera
picture and a task, and chooses every move: travel, turn, climb, descend, camera tilt. The pilot takes off, hands
over, watches, takes over if needed, and lands. Every frame is stored with the image, the model's activations,
the move it chose, the task it had, and the drone's own height, heading, camera pitch and GPS position.

Status: the server side is tested in software, including the geofence. The app side is type-checked but has not
flown. Sections 3 and 4 are the tests that have to pass before a model flight counts.

## 1. How the pilot stays in charge

| To do this | Do this |
| --- | --- |
| Take over at once | Move either stick on the controller. The run ends and the sticks are yours |
| Stop and hover | Stop button in the app |
| Land now | EMERGENCY LAND in the app |

Limits the app enforces whatever the model says (outdoor config):

- height between 15 and 90 m above the take-off point; the run ends if the drone gets 5 m outside that
- travel at most 2 m/s, and never further out once it is 60 m from where Start was pressed; the run ends at 75 m
- no travel at all without a GPS position
- a lost link means hover; three failed requests end the run

The model does not take off and does not land.

## 2. Before leaving (once per outing)

- [ ] Network path tested where you will stand (uni Wi-Fi, or hotspot with the laptop on VPN)
- [ ] Laptop set not to sleep; phone, laptop and three drone batteries charged
- [ ] The app on the phone is built from the current `android/MSDKRemote`
- [ ] On wolfgang: `cd ~/drone-ego/drone-loc-probing && git pull && df -h . && docker ps`
      (stop any old server with `docker stop drone-loc-server`; a 10 minute flight stores about 0.5 GB)
- [ ] Session names written down (section 5)

Setting up at the site: two PowerShell windows on the laptop.

```
ssh -N -g -L 8765:localhost:8765 adaml@wolfgang.ise.canberra.edu.au      # window 1: the tunnel, leave it open
ssh adaml@wolfgang.ise.canberra.edu.au                                     # window 2: where you run the server
```

In window 2: `cd ~/drone-ego/drone-loc-probing`, then `tmux new -s flight` so the server survives a dropped
connection (`tmux attach -t flight` gets it back). Then `ipconfig` on the laptop and put
`http://<Wi-Fi IPv4 address>:8765` in the app. DJI Fly closed. Drone on flat ground at the take-off point, GPS
locked (10 or more satellites). Take off from the ground: logged height is height above that point.

## 3. Indoor test of the new controls (once, before the first outing)

The model cannot travel indoors (no GPS, so no fence), but turning, camera tilt and stick takeover can be tested.

```
docker/run_server.sh --policy pattern --config configs/flight_indoor_explore.yaml --save-frames --session <date>_lab_axes01
```

App: Indoor preset. Take off, hover at about 1.5 m, tick "Model flies", Start.

| Check | Pass |
| --- | --- |
| Stick takeover, first | nudge a stick: the app says "Pilot took over" and the sticks work. If the run ends by itself with that message when you did NOT touch a stick, stop and tell |
| Turn | the drone turns left, pauses, turns right |
| Climb and descend | as before |
| Camera tilt | the picture tilts down, then up; if the log says the gimbal ignored a rotation, note it |
| Stop | hover, sticks work |

## 4. First outing: tests, in this order

App preset: Outdoor. `Ctrl+C` stops the server between sessions.

| # | Server command (after `docker/run_server.sh`) | Model flies | What to do | Pass |
| --- | --- | --- | --- | --- |
| 1 | `--policy pattern --config configs/flight_explore.yaml --save-frames --session <check name>` | off | Test connection, Start, 10 lines, Stop | lines appear; note the round trip |
| 2 | same server, new Start | off | drone on the ground, motors off, 10 minutes, Stop | about 600 lines: the noise-floor log |
| 3 | `--policy pattern --config configs/flight_fence_test.yaml --save-frames --session <fence name>` | on | Take off, climb to 20 m, point at open ground, Start | it flies forward in steps and stops about 10 m out with `at_fence` or `app_fence` in the lines |
| 4 | `--policy pattern --config configs/flight_explore.yaml --save-frames --session <axes name>` | on | Take off, climb to 20 m, Start, one full cycle (about 80 s) | forward, back, left, right, turn left, turn right, climb, descend, tilt down, tilt up, each followed by a hold; it ends near where it began |
| 5 | model flight, section 6 | on | | section 7 |

In sessions 3 and 4, test the stick takeover once each.

## 5. Session names

One name per server start, never reused: `<date>_<site>_<kind><nn>`, for example `1121_oval_model01`.
Kinds: `check`, `fence`, `axes`, `model`. Sites so far: `oval`, `field`, `campus`, `carpark`. A site is a place
that looks different from the others; two ovals side by side are one site.

## 6. The model flight (the data)

```
docker/run_server.sh --model qwen2.5-vl-7b --config configs/flight_explore.yaml --save-frames --save-activations --session <model name>
```

1. App: Outdoor, "Model flies" OFF, press Start and wait for lines. Stop. (This warms the model up; the first
   reply takes two seconds.)
2. Take off. Climb to about 20 m by hand, over the middle of the area you want it to stay in. The fence is a
   60 m circle around this point, so stand where you can see all of it.
3. Tick "Model flies", Start, confirm.
4. Let it run one full pass of the task schedule, 7 minutes 30 seconds. The tasks, in order: climb high, look
   around, camera down, explore, descend to mid height, look around, camera level, explore, descend low, camera
   down, explore. The current task shows in each line of the app.
5. Take over with a stick, press Land, and after it is on the ground wait 10 seconds. (The height it reports on
   the ground is the barometer drift for the flight.)

Take over early only if something is wrong. Odd but safe behaviour, such as sitting at the fence, circling, or
ignoring the task, is a result: leave it running and write it down.

If the run ends by itself (link lost, fence margin, height margin): the drone hovers and you have control. Fly
back to the middle, press Start again. The server keeps numbering, and the schedule restarts from the top.

## 7. Check before leaving the site (window 2, after `Ctrl+C`)

```
ls flights/<model name>/frames | wc -l
ls flights/<model name>/activations | wc -l
docker/run.sh scripts/61_sessions_to_manifest.py --sessions <model name> --site <site> --out data/check.csv
docker/run.sh -c "import pandas as pd; d=pd.read_csv('flights/<model name>/commands.csv'); print(d.chosen.value_counts()); print(d.note.fillna('').value_counts())"
```

Good looks like: the two counts equal, about 450 for one pass; `model flying` near 100%; `gps` near 100%.
The height, pitch and heading ranges, and the count of each move, show how much of the pose range the model
covered by itself. Narrow ranges are not a failed flight, but tell, because the probes need the range and
the task schedule may need changing.

## 8. The collection plan

Per site: two visits at different times of day, at least three hours apart, one model flight each.

| Site | Type for the paper | Visit 1 | Visit 2 |
| --- | --- | --- | --- |
| oval | repetitive, hard to localise | | |
| campus (buildings) | distinctive built-up | | |
| carpark | repetitive built-up | | |
| field (different surroundings from the oval) | natural or repetitive | | |

The experiment needs eight to ten sites in the end, with at least one natural, one beside water and one on a
slope. The campus gives three or four.

After each outing, once per model session:

```
docker/run.sh scripts/61_sessions_to_manifest.py --sessions <model name> --site <site> --out data/manifest_real.csv --append
```

## 9. Fallback: observe flight

If the model cannot produce usable coverage at a site, the same session can be flown by hand with "Model flies"
OFF: the model still sees every frame, chooses a move and has its activations stored, but nothing it chooses
reaches the drone. The manifest marks those frames as `model_in_control = False`, so they are never mixed up
with frames where the model was flying.

## 10. Notes to write down for every session

```
session: 1121_oval_model01
date: 2026-11-21
local_time_start: "09:40"
site: oval
take_off_point: "centre circle of the oval, on the grass"
weather: "clear"
wind: "light, from the west"
pilot: Adam Lawson
network: "uni Wi-Fi"
round_trip_ms: 180
takeovers: 0          # how many times you took over, and why
what_the_model_did: "climbed to about 60 m, turned twice, drifted north to the fence during explore"
height_on_ground_after_landing_m: 0.3
anything_odd: ""
```

Save it as `flights/<session>/notes.yaml` on wolfgang.

## Limits to know about

- The model is never told its height, heading or camera angle. It gets the picture and the task.
- One move a second, at fixed speeds. It is coarse control by design.
- Images are stored at 640 pixels wide and the model sees them at 448.
- Height is relative to the take-off point. On a slope that is not height above ground.
- Heading is the compass heading. Do a compass calibration if the drone asks for one.
