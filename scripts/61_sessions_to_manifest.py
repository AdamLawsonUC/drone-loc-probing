#!/usr/bin/env python
"""Turn flight-server sessions into manifest rows: one row per frame the server saved.

A session recorded with `80_flight_server.py --save-frames` already holds, for every frame, the image and the
telemetry the app sent with it (relative altitude, heading, gimbal pitch, GPS). This script reads
flights/<session>/commands.csv and writes the manifest columns the probe scripts use.

Usage:
  python scripts/61_sessions_to_manifest.py --sessions field01 field02 --site majura_paddock --out data/manifest_real.csv
  python scripts/61_sessions_to_manifest.py --sessions field03 --site uc_campus --out data/manifest_real.csv --append
  python scripts/61_sessions_to_manifest.py --sessions bench01 bench02 --site lab --include_ground --out data/manifest_bench.csv

Produces:
  the manifest CSV named by --out (one row per kept frame)
  results/<run_id>/sessions_summary.csv   per session: frames kept and dropped, altitude, heading and pitch ranges

Labels and their limits
  altitude_agl  the aircraft's relative altitude, i.e. height above the take-off point, in 0.1 m steps. It equals
                height above ground only over flat ground level with the take-off point; altitude_method says so.
                The DEM correction for sloped sites is applied later by 60_dji_to_manifest.py.
  yaw           compass heading, converted from the SDK's -180..180 to 0..360 degrees
  pitch         gimbal pitch in degrees, 0 = level, -90 = straight down
  x, y          metres east and north of the session's first GPS fix; empty without a fix (indoors)
  traj_id       the session cut into --segment_s second pieces; the bootstrap resamples these, because
                neighbouring frames are not independent
Frames taken on the ground (flying = false) are dropped unless --include_ground is given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

COLUMNS = ["frame_id", "site_or_scene", "split", "x", "y", "altitude_agl", "yaw", "pitch", "tile_id", "tile_rotation",
           "u", "v", "yaw_rel_tile", "localisability", "sun_azimuth", "source", "traj_id", "frame_index", "image_path",
           "flight_id", "timestamp", "lat", "lon", "sats", "altitude_method", "policy", "action", "steer_delta"]


def load_session(flights: Path, session: str, site: str, include_ground: bool, segment_s: float) -> tuple[pd.DataFrame, dict]:
    d = flights / session
    df = pd.read_csv(d / "commands.csv")
    n_all = len(df)
    tel = df["telemetry_json"].map(lambda s: json.loads(s) if isinstance(s, str) and s else {})
    df["flying"] = tel.map(lambda t: bool(t.get("flying", False)))
    policy = "?"
    if (d / "session.json").exists():
        policy = json.loads((d / "session.json").read_text()).get("policy", "?")

    has_frame = df["frame_file"].map(lambda f: isinstance(f, str) and f != "" and (d / f).exists())
    # A frame file named by more than one row was overwritten by the later row (sessions recorded before the
    # server numbered files itself). Only the last row describes what is on disk.
    overwritten = has_frame & df.duplicated("frame_file", keep="last")
    has_frame &= ~overwritten
    keep = has_frame & df["rel_alt"].notna()
    if not include_ground:
        keep &= df["flying"]
    k = df[keep].copy()
    out = pd.DataFrame(index=k.index)
    t = pd.to_numeric(k["frame_ts"], errors="coerce")
    seq = pd.to_numeric(k["seq"], errors="coerce").fillna(0).astype(int)
    stem = k["frame_file"].map(lambda f: Path(f).stem)
    out["frame_id"] = [f"{session}__{s}" for s in stem]
    out["site_or_scene"] = site
    out["altitude_agl"] = pd.to_numeric(k["rel_alt"], errors="coerce")
    out["yaw"] = pd.to_numeric(k["yaw"], errors="coerce") % 360.0
    out["pitch"] = pd.to_numeric(k["gimbal_pitch"], errors="coerce")
    out["lat"] = pd.to_numeric(k["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(k["lon"], errors="coerce")
    fix = out["lat"].notna() & out["lon"].notna()
    out["x"], out["y"] = np.nan, np.nan
    if fix.any():
        lat0, lon0 = out.loc[fix, "lat"].iloc[0], out.loc[fix, "lon"].iloc[0]
        out.loc[fix, "x"] = (out.loc[fix, "lon"] - lon0) * 111_320.0 * np.cos(np.radians(lat0))
        out.loc[fix, "y"] = (out.loc[fix, "lat"] - lat0) * 110_540.0
    seg = ((t - t.min()) // segment_s).fillna(0).astype(int) if len(t) else pd.Series(dtype=int)
    out["traj_id"] = [f"{session}__seg{s:03d}" for s in seg]
    out["frame_index"] = seq
    out["image_path"] = [str(Path("flights") / session / f) for f in k["frame_file"]]
    out["flight_id"] = session
    out["timestamp"] = t
    out["sats"] = pd.to_numeric(k["sats"], errors="coerce")
    out["altitude_method"] = "rel_takeoff"
    out["policy"] = policy
    out["action"] = k["action"]
    out["steer_delta"] = pd.to_numeric(k["steer_delta"], errors="coerce")
    out["source"] = "real"
    out["split"] = ""
    for c in COLUMNS:
        if c not in out:
            out[c] = np.nan
    summary = {"session": session, "site": site, "policy": policy, "rows": n_all, "kept": len(out),
               "dropped_ground": int((has_frame & ~df["flying"]).sum()) if not include_ground else 0,
               "dropped_no_frame": int((~has_frame & ~overwritten).sum()), "dropped_overwritten": int(overwritten.sum()),
               "alt_min": out.altitude_agl.min(), "alt_max": out.altitude_agl.max(),
               "yaw_spread_deg": float(np.degrees(np.sqrt(-2 * np.log(max(1e-9, np.hypot(np.sin(np.radians(out.yaw)).mean(),
                                                                                         np.cos(np.radians(out.yaw)).mean()))))))
               if len(out) else np.nan,
               "pitch_min": out.pitch.min(), "pitch_max": out.pitch.max(), "gps_fix_frac": float(fix.mean()) if len(out) else 0.0}
    return out[COLUMNS], summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True, help="folder names under flights/")
    ap.add_argument("--site", required=True, help="site name for these sessions (the held-out unit for probes)")
    ap.add_argument("--out", type=Path, default=common.ROOT / "data" / "manifest_real.csv")
    ap.add_argument("--append", action="store_true", help="add to an existing manifest, replacing rows of the same sessions")
    ap.add_argument("--include_ground", action="store_true", help="keep frames taken while not flying (bench tests)")
    ap.add_argument("--segment_s", type=float, default=20.0, help="length of the pieces used as bootstrap clusters")
    ap.add_argument("--flights", type=Path, default=common.ROOT / "flights")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = common.new_run("sessions_manifest", config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                         seed=args.seed)
    parts, summaries = [], []
    for s in args.sessions:
        if not (args.flights / s / "commands.csv").exists():
            print(f"FAIL {s}: no commands.csv under {args.flights / s}")
            continue
        df, summ = load_session(args.flights, s, args.site, args.include_ground, args.segment_s)
        parts.append(df)
        summaries.append(summ)
        print(f"ok  {s:16s} kept {summ['kept']:5d} of {summ['rows']:5d}  (ground {summ['dropped_ground']}, no frame "
              f"{summ['dropped_no_frame']}, overwritten {summ['dropped_overwritten']})  alt {summ['alt_min']} to {summ['alt_max']} m  pitch {summ['pitch_min']} to "
              f"{summ['pitch_max']} deg  gps {summ['gps_fix_frac']:.0%}")
    if not parts or sum(len(p) for p in parts) == 0:
        sys.exit("no frames kept; manifest not written (bench sessions need --include_ground)")
    man = pd.concat(parts, ignore_index=True)
    out = args.out if args.out.is_absolute() else common.ROOT / args.out
    if args.append and out.exists():
        old = pd.read_csv(out)
        old = old[~old.flight_id.isin(args.sessions)]
        man = pd.concat([old, man], ignore_index=True)
    common.ensure_dir(out.parent)
    man.to_csv(out, index=False)
    pd.DataFrame(summaries).to_csv(run / "sessions_summary.csv", index=False)
    print(f"manifest: {len(man)} frames, {man.site_or_scene.nunique()} site(s), {man.flight_id.nunique()} session(s) -> {out}")
    if man.pitch.nunique() <= 1:
        print("note: gimbal pitch never changes in this manifest, so the pitch probe will be skipped")
    print(f"written {run}")


if __name__ == "__main__":
    main()
