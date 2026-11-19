#!/usr/bin/env python
"""Stack the activations a flight session recorded live into the layout the probe scripts read.

`80_flight_server.py --save-activations` writes one small file per frame, flights/<session>/activations/<seq>.npz,
holding the residual stream of the forward pass that chose the action. This script lines those files up with a
manifest (from 61_sessions_to_manifest.py) and writes one array per layer and readout position.

Usage:
  python scripts/22_session_activations.py --manifest data/manifest_indoor.csv --model qwen2.5-vl-7b --out activations/

Produces, under activations/<model>-flight/<source>/ (source is the manifest's `source` value):
  layer_XX_img_mean.npy       (N, D) mean over the image tokens at the output of decoder layer XX
  layer_XX_last_flight.npy    (N, D) last prompt token under the flight prompt, where the action word is predicted
  vision_tower_img_mean.npy   (N, D) image tokens as they enter decoder layer 0
  frame_ids.txt, index.json   row order, settings, the instruction text and a sha256 per file
Then:  python scripts/30_train_probes.py --acts activations/<model>-flight --split within_scene ...

These differ from script 20's activations in one way: they were recorded while the model was choosing an action
under the flight prompt, not replayed afterwards under a probe prompt. The image tokens come before the prompt, so
img_mean is the same either way for the same image; the last-token readout is not.
Manifest rows without an activation file (frames from a session run without --save-activations) are left out.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model", required=True, help="key of the model that flew, e.g. qwen2.5-vl-7b")
    ap.add_argument("--out", type=Path, default=common.ROOT / "activations")
    ap.add_argument("--flights", type=Path, default=common.ROOT / "flights")
    ap.add_argument("--source", default=None, help="subfolder name; default: the manifest's source column")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    man_path = args.manifest if args.manifest.is_absolute() else common.ROOT / args.manifest
    man = pd.read_csv(man_path)
    source = args.source or str(man["source"].mode().iloc[0])
    run = common.new_run("session_acts", config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                         seed=args.seed)

    files = []
    for _, r in man.iterrows():
        f = args.flights / str(r.flight_id) / "activations" / f"{Path(str(r.image_path)).stem}.npz"   # same number as the frame
        files.append(f if f.exists() else None)
    have = [i for i, f in enumerate(files) if f is not None]
    per_session = man.assign(has=[f is not None for f in files]).groupby("flight_id").has.agg(["sum", "count"])
    for s, row in per_session.iterrows():
        print(f"{'ok ' if row['sum'] else 'none'} {s:16s} {int(row['sum']):5d} of {int(row['count']):5d} frames have activations")
    if not have:
        sys.exit("no activation files found; record the session with --save-activations and the model policy")

    first = np.load(files[have[0]], allow_pickle=False)
    L, D = first["img_mean"].shape
    N = len(have)
    img = np.zeros((L, N, D), np.float16)
    last = np.zeros((L, N, D), np.float16)
    vt = np.zeros((N, D), np.float16)
    has_vt = first["vision_tower"].shape == (D,)
    instructions, n_tok = set(), []
    for k, i in enumerate(have):
        z = np.load(files[i], allow_pickle=False)
        if z["img_mean"].shape != (L, D):
            sys.exit(f"{files[i]}: shape {z['img_mean'].shape} does not match {(L, D)}; sessions from different models?")
        img[:, k], last[:, k] = z["img_mean"], z["last"]
        if has_vt and z["vision_tower"].shape == (D,):
            vt[k] = z["vision_tower"]
        instructions.add(str(z["instruction"]))
        n_tok.append(int(z["n_image_tokens"]))

    model_name = f"{args.model}-flight"
    out = common.ensure_dir(args.out / model_name / source)
    for l in range(L):
        np.save(out / f"layer_{l:02d}_img_mean.npy", img[l])
        np.save(out / f"layer_{l:02d}_last_flight.npy", last[l])
    if has_vt:
        np.save(out / "vision_tower_img_mean.npy", vt)
    (out / "frame_ids.txt").write_text("\n".join(man.frame_id.iloc[have].astype(str)) + "\n")
    bad = int((~np.isfinite(img)).sum() + (~np.isfinite(last)).sum())
    index = {"model": model_name, "base_model": args.model, "random_init": False, "source": source,
             "manifest": str(args.manifest), "manifest_sha256": sha256(man_path), "n": N, "d": D, "n_layers": L,
             "layers": list(range(L)), "positions": ["img_mean", "last"], "recorded": "live, under the flight prompt",
             "instructions": sorted(instructions), "image_tokens_median": int(np.median(n_tok)),
             "sessions": sorted(man.flight_id.iloc[have].astype(str).unique()), "dtype": "float16",
             "non_finite_values": bad, "git_hash": common.git_hash(short=False), "run_id": run.name,
             "sha256": {p.name: sha256(p) for p in sorted(out.glob("*.npy"))}}
    common.dump_json(out / "index.json", index)
    common.dump_json(run / "index.json", index)
    if bad:
        print(f"WARNING: {bad} non-finite values (float16 overflow); the capture needs float32")
    if len(instructions) > 1:
        print(f"note: {len(instructions)} different instructions across these frames; the last-token readout mixes them")
    print(f"{N} frames, {L} layers, D={D} -> {out}")
    print(f"written {run}")


if __name__ == "__main__":
    main()
