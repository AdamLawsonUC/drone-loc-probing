#!/usr/bin/env python
"""Mark the model's flying: was each move the right one for its task, and did each task succeed?

Works from what a session already stores: flights/<session>/commands.csv (the task, the move the model chose,
the drone's height, heading and camera pitch) and session.json (the config, including each task's target).
Nothing here uses the model's activations; 64 joins these marks to the probe readouts.

Usage:
  python scripts/63_score_commands.py --sessions 1121_oval_model01 1121_oval_model02 --out results/

Produces, in results/<run_id>/:
  command_scores.csv   one row per frame where the model was flying a task with a target:
                       frame_id, session, task_id, kind (altitude | gimbal | yaw), target, tol, state, signed_error
                       (target minus state), chosen, correct_move, decision_correct, executed (the move the
                       envelope allowed), note
  task_scores.csv      one row per task attempt: start and end state, final error, fraction of time inside the
                       tolerance, seconds until first inside it, fraction of correct decisions, and for a yaw
                       sweep the degrees actually turned
  summary.csv          per session and task: attempts, mean decision accuracy, mean final error, success rate
The summary is printed.

How a move is marked
  altitude target T, tolerance t   below T - t the right move is climb; above T + t it is descend; inside the band
                                   any move except climb or descend is right
  camera target P, tolerance t     above P + t the right move is tilt_down; below P - t it is tilt_up; inside the
                                   band any move except a tilt is right
  yaw sweep                        any turn is right; the task succeeds if the drone turned through 90% of the sweep
Tasks without a target (explore) are not marked. A task attempt succeeds if it ends inside the tolerance.
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


def correct_move(kind: str, target: float, tol: float, state: float) -> str:
    """The move the task calls for at this state. 'none' means: anything but a move along this axis."""
    if kind == "altitude":
        return "climb" if state < target - tol else "descend" if state > target + tol else "none"
    if kind == "gimbal":
        return "tilt_down" if state > target + tol else "tilt_up" if state < target - tol else "none"
    return "turn"


def is_correct(kind: str, right: str, chosen: str) -> bool:
    if kind == "yaw":
        return chosen in ("turn_left", "turn_right")
    axis = ("climb", "descend") if kind == "altitude" else ("tilt_down", "tilt_up")
    return chosen == right if right != "none" else chosen not in axis


def score_session(flights: Path, session: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = flights / session
    df = pd.read_csv(d / "commands.csv")
    cfg = json.loads((d / "session.json").read_text()).get("config", {})
    targets = {str(t["id"]): t.get("target") for t in (cfg.get("tasks") or [])}
    if "task_id" not in df or "chosen" not in df:
        raise ValueError(f"{session}: no task_id/chosen columns; it was recorded before the task schedule existed")
    tel = df["telemetry_json"].map(lambda s: json.loads(s) if isinstance(s, str) and s else {})
    df["sticks"] = tel.map(lambda t: bool(t.get("sticks", False)))
    df["flying"] = tel.map(lambda t: bool(t.get("flying", False)))
    df["run_t"] = tel.map(lambda t: t.get("run_t", np.nan))
    df["t"] = pd.to_numeric(df["frame_ts"], errors="coerce")
    df["task_id"] = df["task_id"].fillna("").astype(str)
    # a task attempt = an unbroken stretch of one task with the model flying
    live = df["sticks"] & df["flying"]
    change = (df["task_id"] != df["task_id"].shift()) | (live != live.shift()) | (df["run_t"].diff() < 0)
    df["attempt"] = change.cumsum()

    frames, tasks = [], []
    for attempt, g in df[live].groupby("attempt"):
        tid = g["task_id"].iloc[0]
        tg = targets.get(tid)
        if not tg:
            continue
        if "altitude_m" in tg:
            kind, target, tol, state = "altitude", float(tg["altitude_m"]), float(tg.get("tol", 5)), pd.to_numeric(g["rel_alt"], errors="coerce")
        elif "gimbal_pitch_deg" in tg:
            kind, target, tol, state = "gimbal", float(tg["gimbal_pitch_deg"]), float(tg.get("tol", 10)), pd.to_numeric(g["gimbal_pitch"], errors="coerce")
        elif "yaw_sweep_deg" in tg:
            kind, target, tol, state = "yaw", float(tg["yaw_sweep_deg"]), 0.0, pd.to_numeric(g["yaw"], errors="coerce")
        else:
            continue
        ok = state.notna()
        g, state = g[ok], state[ok]
        if len(g) < 2:
            continue
        right = [correct_move(kind, target, tol, s) for s in state]
        chosen = g["chosen"].fillna("hold").astype(str)
        good = np.array([is_correct(kind, r, c) for r, c in zip(right, chosen)])
        stem = g["frame_file"].map(lambda f: Path(f).stem if isinstance(f, str) and f else "")
        f = pd.DataFrame({
            "frame_id": [f"{session}__{s}" if s else "" for s in stem], "session": session, "attempt": attempt,
            "task_id": tid, "kind": kind, "target": target, "tol": tol, "state": state.to_numpy(),
            "signed_error": (target - state.to_numpy()) if kind != "yaw" else np.nan, "chosen": chosen.to_numpy(),
            "correct_move": right, "decision_correct": good, "executed": g["action"].to_numpy(),
            "note": g["note"].fillna("").to_numpy()})
        frames.append(f)

        t = g["t"].to_numpy()
        row = {"session": session, "attempt": attempt, "task_id": tid, "kind": kind, "target": target, "tol": tol,
               "frames": len(g), "seconds": float(t[-1] - t[0]), "decision_accuracy": float(good.mean()),
               "start_state": float(state.iloc[0]), "end_state": float(state.iloc[-1])}
        if kind == "yaw":
            turned = float(np.abs(np.sum(np.degrees(np.angle(np.exp(1j * np.radians(np.diff(state.to_numpy()))))))))
            row.update({"turned_deg": turned, "final_error": max(0.0, target - turned), "success": turned >= 0.9 * target,
                        "frac_in_tol": np.nan, "seconds_to_tol": np.nan})
        else:
            inside = np.abs(state.to_numpy() - target) <= tol
            first = np.flatnonzero(inside)
            row.update({"turned_deg": np.nan, "final_error": float(abs(state.iloc[-1] - target)), "success": bool(inside[-1]),
                        "frac_in_tol": float(inside.mean()),
                        "seconds_to_tol": float(t[first[0]] - t[0]) if len(first) else np.nan})
        tasks.append(row)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), pd.DataFrame(tasks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--flights", type=Path, default=common.ROOT / "flights")
    ap.add_argument("--out", type=Path, default=common.RESULTS, help="kept for the usual interface; runs go to results/<run_id>/")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = common.new_run("command_scores", config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                         seed=args.seed)
    fr, ta = [], []
    for s in args.sessions:
        try:
            f, t = score_session(args.flights, s)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {s}: {str(e)[:140]}")
            continue
        print(f"ok  {s:22s} {len(f):5d} marked frames, {len(t):3d} task attempts")
        fr.append(f)
        ta.append(t)
    if not ta or sum(len(t) for t in ta) == 0:
        sys.exit("nothing to mark: no frames where the model was flying a task with a target")
    frames, tasks = pd.concat(fr, ignore_index=True), pd.concat(ta, ignore_index=True)
    frames.to_csv(run / "command_scores.csv", index=False)
    tasks.to_csv(run / "task_scores.csv", index=False)
    summ = tasks.groupby(["session", "task_id", "kind"]).agg(
        attempts=("success", "size"), decision_accuracy=("decision_accuracy", "mean"),
        final_error=("final_error", "mean"), success_rate=("success", "mean")).reset_index()
    summ.to_csv(run / "summary.csv", index=False)
    print()
    print(summ.round(2).to_string(index=False))
    by_kind = frames.groupby("kind").decision_correct.mean()
    print("\ncorrect moves, all sessions: " + ", ".join(f"{k} {v:.0%}" for k, v in by_kind.items()))
    print("what it chose:               " + ", ".join(f"{k} {v}" for k, v in frames.chosen.value_counts().items()))
    print(f"written {run}")


if __name__ == "__main__":
    main()
