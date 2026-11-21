#!/usr/bin/env python
"""Baselines and controls for the probes, on the same frames, folds and probe as 30_train_probes.py.

Usage:
  python scripts/31_baselines.py --manifest data/manifest.csv --acts activations/ \
      --controls pixel vision_tower random_init shuffled --probes_run results/<probes run_id> --out results/

Controls
  pixel         ridge on cheap handcrafted image statistics (edge density, gradient magnitude, Laplacian-pyramid
                energy per scale, colour histograms). Needs the frames on disk; the features are cached in
                activations/pixel/<source>/ so the frames are only needed once.
  vision_tower  the same probe on the image tokens before the language model (vision_tower_img_mean.npy)
  random_init   the same probe on activations/<model>-randinit/<source>/ (run script 20 with --random_init first)
  shuffled      the same probe trained on label-permuted training sets: chance level and its spread

Produces, in results/<run_id>/:
  baselines.csv   the probes.csv columns plus `control`. For shuffled, value is the mean over permutations and
                  ci_lo / ci_hi the 2.5 and 97.5 percentiles over permutations.
  gate_g1.csv     with --probes_run: per target, layer and readout, the probe against pixel and vision tower,
                  and whether the intervals are separated. The verdict for gate G1 is printed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import probe_lib as pl  # noqa: E402


# --------------------------------------------------------------------------------------------- pixel statistics
def pixel_features(path: Path, side: int = 256) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((side, side), Image.BILINEAR)
    rgb = np.asarray(im, dtype=np.float32) / 255.0
    g = rgb.mean(2)
    gx, gy = np.gradient(g)
    mag = np.hypot(gx, gy)
    feats = [mag.mean(), mag.std(), (mag > 0.05).mean(), (mag > 0.10).mean(), g.mean(), g.std()]
    # Laplacian pyramid energy per octave: where the texture energy sits says how big things look
    cur = Image.fromarray((g * 255).astype(np.uint8))
    for _ in range(5):
        small = cur.resize((max(cur.width // 2, 1), max(cur.height // 2, 1)), Image.BILINEAR)
        up = small.resize(cur.size, Image.BILINEAR)
        lap = np.asarray(cur, dtype=np.float32) - np.asarray(up, dtype=np.float32)
        feats += [np.abs(lap).mean() / 255.0, lap.std() / 255.0]
        cur = small
    for c in range(3):
        h, _ = np.histogram(rgb[..., c], bins=8, range=(0, 1))
        feats += list(h / h.sum())
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0
    feats += [hsv[..., 1].mean(), hsv[..., 1].std()]
    return np.asarray(feats, dtype=np.float32)


def get_pixel_features(man: pd.DataFrame, cache_dir: Path) -> np.ndarray:
    ids_file, feat_file = cache_dir / "frame_ids.txt", cache_dir / "pixel_features.npy"
    if feat_file.exists() and ids_file.exists():
        ids = ids_file.read_text().splitlines()
        pos = {f: i for i, f in enumerate(ids)}
        if all(f in pos for f in man.frame_id.astype(str)):
            return np.load(feat_file)[[pos[f] for f in man.frame_id.astype(str)]].astype(np.float64)
    print(f"computing pixel statistics for {len(man)} frames")
    feats = []
    for k, p in enumerate(man.image_path):
        p = Path(p)
        feats.append(pixel_features(p if p.is_absolute() else common.ROOT / p))
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1}/{len(man)}")
    F = np.stack(feats)
    common.ensure_dir(cache_dir)
    np.save(feat_file, F)
    ids_file.write_text("\n".join(man.frame_id.astype(str)) + "\n")
    return F.astype(np.float64)


def main() -> None:
    pcfg = common.load_yaml(common.CONFIGS / "probes.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=common.ROOT / "data" / "manifest.csv")
    ap.add_argument("--acts", type=Path, default=common.ROOT / "activations", help="the activations/ root")
    ap.add_argument("--model", default="qwen2.5-vl-7b")
    ap.add_argument("--source", default=None)
    ap.add_argument("--controls", nargs="+", default=pcfg["controls"], choices=["pixel", "vision_tower", "random_init", "shuffled"])
    ap.add_argument("--targets", nargs="+", default=["altitude", "heading", "pitch", "scene_xy"])
    ap.add_argument("--split", default="scene_heldout", choices=["scene_heldout", "site_heldout", "within_scene", "time_block"])
    ap.add_argument("--probes_run", type=Path, default=None, help="results/<run_id> from script 30, for the G1 table")
    ap.add_argument("--layer_step", type=int, default=1)
    ap.add_argument("--shuffles", type=int, default=10)
    ap.add_argument("--alphas", nargs="*", type=float, default=pcfg["probe"]["alphas"])
    ap.add_argument("--bootstrap_n", type=int, default=pcfg["probe"]["bootstrap_n"])
    ap.add_argument("--out", type=Path, default=common.RESULTS, help="kept for the plan's interface; runs go to results/<run_id>/")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    store = pl.ActStore(args.acts / args.model, args.source)
    man_path = args.manifest if args.manifest.is_absolute() else common.ROOT / args.manifest
    man, rows = store.align(pd.read_csv(man_path))
    specs, notes = pl.build_targets(args.targets, pcfg, man, args.split)
    for n in notes:
        print("note:", n)
    ok = np.all([np.isfinite(t.Y).all(1) for t in specs], axis=0)
    if not ok.all():
        man, rows = man[ok].reset_index(drop=True), rows[ok]
        for t in specs:
            t.Y = t.Y[ok]
    clusters = man["traj_id"].to_numpy()
    cv = pcfg["probe"]["cv_folds"]
    main_specs = [t for t in specs if not t.per_scene]
    xy_specs = [t for t in specs if t.per_scene]
    groups = []
    if main_specs:
        groups.append((main_specs, pl.make_folds(man, args.split, cv), args.split))
    if xy_specs:
        groups.append((xy_specs, pl.make_folds(man, "within_scene", cv, per_scene=True), "within_scene"))

    run = common.new_run("baselines", config={**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                              "acts_dir": str(store.path), "n_frames": len(man)},
                         config_files=[common.CONFIGS / "probes.yaml"], seed=args.seed)
    cols = pl.CSV_COLUMNS + ["control"]
    out_csv = run / "baselines.csv"
    all_rows: list[dict] = []
    base0 = {"model": store.model, "source": store.source, "probe": "ridge", "seed": args.seed}
    layers = sorted(set(store.layers[::args.layer_step]) | {store.layers[-1]})
    t0 = time.time()

    def probe_block(X, base, control):
        for group, folds, split_name in groups:
            oof, _, fold_of, atab = pl.run_probe(X, group, folds, args.alphas, cv, seed=args.seed)
            for t in group:
                for r in pl.score_rows({**base, "split": split_name}, t, oof[t.name], fold_of, clusters, atab,
                                       args.bootstrap_n, args.seed, per_fold=False):
                    all_rows.append({**r, "control": control})
        pd.DataFrame(all_rows, columns=cols).to_csv(out_csv, index=False)

    if "pixel" in args.controls:
        try:
            F = get_pixel_features(man, args.acts / "pixel" / store.source)
            probe_block(F, {**base0, "model": "pixel", "layer": -1, "position": "pixel", "prompt": "any"}, "pixel")
            print("ok  pixel")
        except FileNotFoundError as e:
            print(f"SKIP pixel: frames not on this machine ({str(e)[:80]}). Run this control where data/ lives.")

    if "vision_tower" in args.controls:
        if store.has("vision_tower_img_mean"):
            probe_block(store.load("vision_tower_img_mean", rows),
                        {**base0, "layer": -1, "position": "vision_tower", "prompt": "any"}, "vision_tower")
            print("ok  vision_tower")
        else:
            print("SKIP vision_tower: vision_tower_img_mean.npy not in the activation folder")

    if "random_init" in args.controls:
        try:
            rs = pl.ActStore(args.acts / f"{args.model}-randinit", args.source)
            rpos = {f: i for i, f in enumerate(rs.frame_ids)}
            rrows = np.array([rpos[f] for f in man.frame_id.astype(str)])
            for layer in [l for l in layers if l in rs.layers]:
                for stem, position, prompt in ((f"layer_{layer:02d}_img_mean", "img_mean", "any"),
                                               (f"layer_{layer:02d}_last_neutral", "last", "neutral")):
                    if rs.has(stem):
                        probe_block(rs.load(stem, rrows), {**base0, "model": rs.model, "layer": layer, "position": position,
                                                           "prompt": prompt}, "random_init")
                print(f"  random_init layer {layer:02d}")
            print("ok  random_init")
        except (FileNotFoundError, KeyError) as e:
            print(f"SKIP random_init: {str(e)[:120]}\n     run: docker/run.sh scripts/20_extract_activations.py --model {args.model} "
                  f"--manifest {args.manifest} --positions img_mean last --prompts neutral --random_init --out activations/")

    if "shuffled" in args.controls and args.shuffles > 0:
        for layer in layers:
            stem = f"layer_{layer:02d}_img_mean"
            if not store.has(stem):
                continue
            X = store.load(stem, rows)
            for group, folds, split_name in groups:
                _, shuf, _, _ = pl.run_probe(X, group, folds, args.alphas, cv, n_shuffle=args.shuffles, seed=args.seed)
                for t in group:
                    per = [pl.metrics_for(t.enc, t.Y, shuf[t.name][s], t.unit) for s in range(args.shuffles)
                           if np.isfinite(shuf[t.name][s]).all()]
                    for k in (per[0] if per else {}):
                        v = np.array([m[k] for m in per])
                        all_rows.append({**base0, "target": t.name, "layer": layer, "position": "img_mean", "prompt": "any",
                                         "split": split_name, "fold": "pooled", "metric": k, "value": float(np.nanmean(v)),
                                         "ci_lo": float(np.nanpercentile(v, 2.5)), "ci_hi": float(np.nanpercentile(v, 97.5)),
                                         "alpha": np.nan, "n_test": len(t.Y), "control": "shuffled"})
            pd.DataFrame(all_rows, columns=cols).to_csv(out_csv, index=False)
            print(f"  shuffled layer {layer:02d}")
        print("ok  shuffled")
    common.log_compute(run, "baselines", time.time() - t0)
    base = pd.DataFrame(all_rows, columns=cols)
    base.to_csv(out_csv, index=False)

    # ------------------------------------------------------------------------------------- gate G1 table
    if args.probes_run is not None:
        pr = args.probes_run if args.probes_run.is_absolute() else common.ROOT / args.probes_run
        probes = pd.read_csv(pr / "probes.csv")
        probes = probes[(probes.fold == "pooled") & (probes.probe == "ridge")]
        gate = []
        for t in specs:
            hm = pl.headline_metric(t)
            lower = hm in pl.LOWER_IS_BETTER
            ctrl = {c: base[(base.control == c) & (base.target == t.name) & (base.metric == hm) & (base.layer == -1)]
                    for c in ("pixel", "vision_tower")}
            for _, r in probes[(probes.target == t.name) & (probes.metric == hm)].iterrows():
                row = {"target": t.name, "metric": hm, "layer": int(r.layer), "position": r.position, "prompt": r.prompt,
                       "probe": r.value, "probe_lo": r.ci_lo, "probe_hi": r.ci_hi}
                beats = []
                for c, d in ctrl.items():
                    if d.empty:
                        row.update({c: np.nan, f"{c}_lo": np.nan, f"{c}_hi": np.nan, f"beats_{c}": np.nan})
                        continue
                    b = d.iloc[0]
                    sep = (r.ci_hi < b.ci_lo) if lower else (r.ci_lo > b.ci_hi)
                    row.update({c: b.value, f"{c}_lo": b.ci_lo, f"{c}_hi": b.ci_hi, f"beats_{c}": bool(sep)})
                    beats.append(bool(sep))
                for c in ("shuffled", "random_init"):
                    d = base[(base.control == c) & (base.target == t.name) & (base.metric == hm) & (base.layer == r.layer)
                             & (base.position == r.position)]
                    row[c] = d.value.iloc[0] if not d.empty else np.nan
                row["represented"] = bool(len(beats) == 2 and all(beats))
                gate.append(row)
        gate = pd.DataFrame(gate)
        gate.to_csv(run / "gate_g1.csv", index=False)
        print("\ngate table (pooled held-out folds; 'represented' = beats pixel and vision tower with separated intervals):")
        for t in specs:
            g = gate[gate.target == t.name]
            if g.empty:
                continue
            hm = g.metric.iloc[0]
            best = g.loc[g.probe.idxmin() if hm in pl.LOWER_IS_BETTER else g.probe.idxmax()]
            n_pass = int(g.represented.sum())
            print(f"  {t.name:10s} {hm}: best probe {best.probe:.3f} [{best.probe_lo:.3f}, {best.probe_hi:.3f}] at L{int(best.layer):02d} "
                  f"{best.position}/{best.prompt}; pixel {best.pixel:.3f}; vision tower {best.vision_tower:.3f}; "
                  f"shuffled {best.shuffled:.3f}; passes at {n_pass} of {len(g)} layer-readout combinations")
        alt = gate[gate.target == "altitude"]
        if not alt.empty:
            verdict = "PASSED" if alt.represented.any() else "NOT PASSED"
            print(f"\nGate G1 (altitude beats pixel and vision tower on held-out scenes at some layer): {verdict}")
            (run / "gate_g1.txt").write_text(f"G1 {verdict}\nprobes run: {pr.name}\nbaselines run: {run.name}\n")
    print(f"written {run}")


if __name__ == "__main__":
    main()
