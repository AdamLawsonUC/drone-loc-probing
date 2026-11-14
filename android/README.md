# Android side: link the DJI MSDK app to the flight server

These files add one screen, "Probe flight (server link)", to the MSDKRemote app in
[AdamLawsonUC/DroneAI](https://github.com/AdamLawsonUC/DroneAI) (or a fork of it, such as SentinelSAR).
The screen sends a camera frame and telemetry to `scripts/80_flight_server.py` once a second and applies the
vertical velocity that comes back. It reuses the app's existing `FrameSnapshotter` and the same DJI SDK calls as
the Gemini screen, and adds no new dependencies or resources.

| File | What it is |
| --- | --- |
| `probe/FlightServerClient.java` | HTTP client for `/health` and `/act`, plus the clock-offset estimate |
| `probe/ProbeFlightRunner.java` | the 1 Hz loop, the envelope checks, the 10 Hz virtual-stick thread, the phone-side CSV log |
| `probe/ProbeFlightActivity.java` | the screen (layout built in code) |
| `msdkremote_probe.patch` | everything above plus the three small edits to existing files, as one patch |

## Install (Windows, PowerShell, in the app repo root)

```
git checkout -b probe-flight
git apply --check C:\path\to\drone-loc-probing\android\msdkremote_probe.patch
git apply C:\path\to\drone-loc-probing\android\msdkremote_probe.patch
```

Then build and run from Android Studio as usual. The patch was made against DroneAI commit `f45baf8`.

If the patch does not apply (a fork that has moved on), do it by hand:

1. Copy the three files in `probe/` to `MSDKRemote/app/src/main/java/com/msdkremote/probe/`.
2. `AndroidManifest.xml`: add `android:usesCleartextTraffic="true"` and
   `tools:replace="android:usesCleartextTraffic"` to `<application>` (the server speaks plain HTTP, and the DJI
   SDK's own manifest sets the attribute to false, so the merge fails without the replace), and declare the activity:
   `<activity android:name=".probe.ProbeFlightActivity" android:exported="false" android:screenOrientation="portrait" />`
3. `MainActivity.java`: add
   `public void StartProbeFlightActivity(View view) { startActivity(new Intent(this, com.msdkremote.probe.ProbeFlightActivity.class)); }`
4. `activity_main.xml`: add a button with `android:onClick="StartProbeFlightActivity"`.

## What the screen does and does not do

- It never takes off, lands, yaws or moves horizontally. You take off and fly into the envelope by hand, then
  press Start. Stop zeroes the command and hands control back to the remote controller.
- "Send stick commands" is off by default. Off is a bench test: the loop runs, the server logs, nothing reaches
  the aircraft.
- Presets: Indoor (0.5 to 2.5 m, 0.2 m/s) and Outdoor (15 to 60 m, 0.5 m/s). The server's envelope must sit
  inside the chosen preset or the run refuses to start; the server's tighter numbers are then the ones used.
- A command is held for two loop periods and then decays to hover, so a dropped link means hover.
- The run ends by itself if the aircraft leaves the envelope by more than the margin, the link to the aircraft
  drops, the aircraft lands, or three requests fail in a row.
- Every round trip is logged on the phone in `Android/data/com.msdkremote/files/probe_flights/probe_<session>_<time>.csv`
  (Android Studio: Device Explorer). The server keeps its own log in `flights/<session>/`.

Status: type-checked against stub versions of the Android and DJI APIs and exercised against the real server
with the same request format; first Android Studio build on 14 Nov 2026 found the manifest merge conflict fixed above. See
`docs/hardware_bringup.md` for the test order.
