# Data collection protocol: outdoor flights

Version 1, 19 Nov 2026. For the DJI Mini 3 with the "Probe flight" screen and the flight server on wolfgang.
Follow it from the first outing: flights flown this way are usable as experiment data, including the test flights.

What every frame must end up with: the image the model saw, the model's activations, and the drone's own
reading of height, heading, camera pitch and GPS position. The server stores all of that when it is started with
`--save-frames --save-activations`. Nothing recorded by DJI Fly can be used; DJI Fly must stay closed.

## 0. Before leaving (once per outing)

- [ ] Network path chosen and tested where you will stand: uni Wi-Fi, or phone hotspot with the laptop on VPN
- [ ] Laptop charged, set not to sleep; phone charged; three drone batteries
- [ ] On wolfgang: `cd ~/drone-ego/drone-loc-probing && git pull && df -h . && docker ps`
      (a 10 minute flight stores about 0.5 GB; stop any old `drone-loc-server` with `docker stop drone-loc-server`)
- [ ] Session names written down in advance (section 2), so nothing is typed from memory in the field
- [ ] Paper or phone notes ready for section 7

## 1. Setting up at the site (every outing)

On the laptop, two PowerShell windows:

```
ssh -N -g -L 8765:localhost:8765 adaml@wolfgang.ise.canberra.edu.au      # window 1: the tunnel, leave it open
ssh adaml@wolfgang.ise.canberra.edu.au                                     # window 2: where you run the server
```

In window 2, start the server inside tmux, so it survives a dropped connection:

```
cd ~/drone-ego/drone-loc-probing
tmux new -s flight          # if it says a session exists: tmux attach -t flight
```

If the connection drops, log in again and run `tmux attach -t flight`; the server will still be running.
(If tmux is not installed, run the server directly and keep window 2 open.)

Then:

1. `ipconfig` on the laptop; put `http://<Wi-Fi IPv4 address>:8765` in the app's server field.
2. Drone and controller on, DJI Fly closed, our app open on the Probe flight screen.
3. Put the drone on flat ground at the take-off point. Wait for a GPS lock (the drone's lights, and 10 or more
   satellites). Take off from the ground, not from a table or a slope: logged height is height above this point.
4. Roll the gimbal wheel and check the picture in the app tilts.

## 2. Session names

One name per server start, never reused: `<date>_<site>_<kind><nn>`

| Kind | Meaning | Example |
| --- | --- | --- |
| `check` | link check on the ground | `1121_oval_check01` |
| `noise` | stationary noise-floor log | `1121_oval_noise01` |
| `obs` | observe flight: you fly, the model watches | `1121_oval_obs01` |
| `pattern`, `baro`, `qwen` | closed-loop tests | `1121_oval_qwen01` |

Site names so far: `oval`, `field`, `campus`, `carpark`. A site is a place that looks different from the others;
two ovals side by side are one site. Add `am` or `pm` to your notes, not to the name.

## 3. First outing only: tests, in this order

Run each command in the tmux window. `Ctrl+C` stops the server between sessions. App preset: Outdoor.

| # | Server command (after `docker/run_server.sh`) | Stick commands | What to do | Pass |
| --- | --- | --- | --- | --- |
| 1 | `--policy pattern --save-frames --session <check name>` | off | Test connection, Start, 10 lines, Stop | lines appear; note the round trip time |
| 2 | `--policy pattern --save-frames --session <noise name>` | off | Start; leave the drone on the ground, motors off, for 10 minutes; Stop | about 600 lines |
| 3 | observe flight, section 4 | off | fly the pattern | section 5 checks pass |
| 4 | `--policy pattern --save-frames --session <pattern name>` | on | Take off, climb to 20 m by hand, Start, one full cycle, Stop | climbs, holds, descends, holds |
| 5 | `--policy baro --save-frames --session <baro name>` | on | Start from about 20 m; Stop; Start from about 40 m | settles near 30 m both times |
| 6 | `--model qwen2.5-vl-7b --save-frames --save-activations --session <qwen name>` | on | Start from about 30 m, run 2 minutes, Stop | whatever happens is the result; note it |

Sessions 4 to 6 keep the drone between 15 and 60 m. If the app refuses to start, the drone is outside that band.
Session 2 gives the GPS scatter and barometer drift that every probe error is later compared against.

## 4. The observe flight (the data)

Server:

```
docker/run_server.sh --model qwen2.5-vl-7b --config configs/flight_observe.yaml --save-frames --save-activations --session <obs name>
```

`flight_observe.yaml` sends two frames a second. If the round trip in the field is above about 400 ms, leave the
`--config` part out; that gives one frame a second and is also fine.

App: Outdoor, stick commands OFF. Press Start and wait for lines. Then Take off.

Fly inside a box about 150 m across, slowly (walking to jogging pace), keeping the drone in sight.
One battery, about 9 minutes:

| Block | Camera | Height | What to fly | About |
| --- | --- | --- | --- | --- |
| A1 | straight down | 20 m | two passes across the box, 30 m apart, then one slow full turn on the spot | 90 s |
| A2 | straight down | 50 m | the same | 90 s |
| A3 | straight down | 90 m | the same | 90 s |
| B1 | 45 degrees down | 90 m | the same | 90 s |
| B2 | 45 degrees down | 50 m | the same | 90 s |
| B3 | 45 degrees down | 20 m | the same | 90 s |
| C | 45 degrees down | 20 m to 90 m | one slow steady climb, hovering over the middle of the box | 40 s |
| D | sweep | 50 m | roll the camera slowly from level to straight down and back | 30 s |

Rules that keep the data usable:

- Hold each height steady during a block; the climbs between blocks are useful too, so do not hurry them.
- Make the full turn slow, 30 seconds or more, so every heading is seen.
- Do not stop the app between blocks. One Start for the whole flight.
- After landing, wait 10 seconds before pressing Stop. The height the drone reports on the ground after landing
  is the barometer drift for that flight.
- If the app stops by itself (link lost), land, press Start again and carry on. The server keeps numbering.
- A second battery at the same site: keep the same server running and the same session, press Start again.

## 5. Check before leaving the site (2 minutes, window 2, a second tmux pane or after `Ctrl+C`)

```
ls flights/<obs name>/frames | wc -l
ls flights/<obs name>/activations | wc -l
docker/run.sh scripts/61_sessions_to_manifest.py --sessions <obs name> --site <site> --out data/check.csv
```

Good looks like: the two counts equal and in the hundreds (about 540 at one frame a second, about 1,000 at two);
`kept` close to the count; `alt` from about 20 to about 90 m; `pitch` from about -90 to about -45 or higher;
`gps` near 100%. If height, pitch or GPS is wrong, fly it again now; it is much cheaper than coming back.

## 6. The collection plan

Per site: two visits at different times of day, at least three hours apart (shadows must not give away heading).
Each visit is one observe flight. A site is finished after two good visits.

| Site | Type for the paper | Visit 1 | Visit 2 |
| --- | --- | --- | --- |
| oval | repetitive, hard to localise | | |
| campus (buildings) | distinctive built-up | | |
| carpark | repetitive built-up | | |
| field (different surroundings from the oval) | natural or repetitive | | |

The experiment needs eight to ten sites in the end, with at least one natural, one over or beside water and one on
a slope. The campus gives three or four. The rest come later; the same protocol applies anywhere.

Optional at each site, if a battery is spare: the Qwen closed-loop run from section 3, row 6, for two minutes.

## 7. Notes to write down for every session

```
session: 1121_oval_obs01
date: 2026-11-21
local_time_start: "09:40"
site: oval
take_off_point: "north-west corner of the oval, on the grass"
weather: "clear"
wind: "light, from the west"
pilot: Adam Lawson
network: "uni Wi-Fi" 
round_trip_ms: 180
battery: 1
blocks_flown: [A1, A2, A3, B1, B2, B3, C, D]
height_on_ground_after_landing_m: 0.3
anything_odd: ""
```

Save it afterwards as `flights/<session>/notes.yaml` on wolfgang. The pilot name, weather and take-off point are
needed for the dataset description in the paper.

## 8. After each outing (wolfgang)

```
docker/run.sh scripts/61_sessions_to_manifest.py --sessions <obs name> --site <site> --out data/manifest_real.csv --append
```

Run it once per observe session, with that session's site. Then tell the session names and sites.
Back up `flights/` off wolfgang when convenient; frames and activations are not in git.

## Limits to know about

- Images are stored at 640 pixels wide and the model sees them at 448. That is the dataset; there is no
  full-resolution video unless a separate recording is made.
- Height is relative to the take-off point. On flat ground that is height above ground. On a slope it is not, and
  that site will need the terrain correction later; note the take-off point carefully there.
- Heading is the compass heading. If the app or drone asks for a compass calibration, do it before the flight.
