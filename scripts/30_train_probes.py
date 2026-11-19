#!/usr/bin/env python
"""Train linear probes on cached activations, at every stored layer, readout position and prompt condition.

Usage (laptop or wolfgang; CPU only):
  python scripts/30_train_probes.py --acts activations/qwen2.5-vl-7b --targets altitude heading pitch scene_xy \
      --split scene_heldout --out results/
  python scripts/30_train_probes.py --acts activations/qwen2.5-vl-7b --targets altitude --split scene_heldout \
      --layer_step 4 --bootstrap_n 200                                   # quick look, about 10 minutes

Produces, in results/<run_id>/:
  probes.csv        long format: model, target, layer, position, prompt, split, source, fold, probe, metric, value,
                    ci_lo, ci_hi, alpha, n_test, seed. fold = pooled carries the bootstrap interval; the other fold
                    rows are per held-out scene.
  probes/<model>_<target>_L<XX>_<split>.npz
                    weights on raw img_mean activations, fitted on all frames: w, b, alpha, encoding, position.
                    This is the file 80_flight_server.py and the patching scripts load.
  best_layers.csv   best layer per target, position and prompt, by the headline metric
  predictions/L<XX>_<position>_<prompt>_<split>.npz
                    the held-out probe reading for every frame (frame_id, <target>_pred, <target>_true), for
                    comparing what the probe read with what the model then did
  alpha_edges.txt   written only if the alpha search hit the edge of the grid somewhere

Notes
  - Ridge on standardised activations, alpha by grouped 5-fold CV inside the training split (probes.yaml).
  - scene_heldout leaves one scene out in turn and pools the held-out predictions.
  - scene_xy is fitted per scene on trajectory-held-out folds whatever --split says, because scene coordinates
    mean nothing across scenes; its rows are labelled split = within_scene. It measures memorisation only.
  - Intervals are a cluster bootstrap over trajectories.
  - Targets without labels in the manifest are skipped with a message.
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


def feature_sets(store: pl.ActStore, layer: int, positions: list[str], prompts: list[str], targets: list[str]):
    """Yield (file stem, position, prompt, targets that this file serves)."""
    if "img_mean" in positions and store.has(f"layer_{layer:02d}_img_mean"):
        yield f"layer_{layer:02d}_img_mean", "img_mean", "any", targets
    if "last" in positions:
        if "neutral" in prompts and store.has(f"layer_{layer:02d}_last_neutral"):
            yield f"layer_{layer:02d}_last_neutral", "last", "neutral", targets
        if "flight" in prompts and store.has(f"layer_{layer:02d}_last_flight"):
            yield f"layer_{layer:02d}_last_flight", "last", "flight", targets
        if "targeted" in prompts:
            for t in targets:
                if store.has(f"layer_{layer:02d}_last_targeted-{t}"):
                    yield f"layer_{layer:02d}_last_targeted-{t}", "last", "targeted", [t]


def mlp_check(X, t: pl.TargetSpec, folds, seed: int) -> np.ndarray:
    """One-hidden-layer MLP on the same folds, to test whether the linear probe is leaving signal behind."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    pred = np.full(t.Y.shape, np.nan)
    for f in folds:
        sc = StandardScaler().fit(X[f.train])
        ymu, ysd = t.Y[f.train].mean(0), t.Y[f.train].std(0) + 1e-8
        m = MLPRegressor(hidden_layer_sizes=(256,), alpha=1e-2, early_stopping=True, max_iter=300, random_state=seed)
        yt = (t.Y[f.train] - ymu) / ysd
        m.fit(sc.transform(X[f.train]), yt if yt.shape[1] > 1 else yt[:, 0])
        p = m.predict(sc.transform(X[f.test]))
        pred[f.test] = (p.reshape(len(f.test), -1)) * ysd + ymu
    return pred


def main() -> None:
    pcfg = common.load_yaml(common.CONFIGS / "probes.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", type=Path, required=True, help="activations/<model> or activations/<model>/<source>")
    ap.add_argument("--source", default=None, help="subfolder of --acts when there is more than one")
    ap.add_argument("--manifest", type=Path, default=None, help="default: the manifest recorded in index.json")
    ap.add_argument("--targets", nargs="+", default=["altitude", "heading", "pitch", "scene_xy"])
    ap.add_argument("--split", default="scene_heldout", choices=["scene_heldout", "site_heldout", "within_scene"])
    ap.add_argument("--positions", nargs="+", default=["img_mean", "last"])
    ap.add_argument("--prompts", nargs="+", default=["neutral", "targeted", "flight"],
                    help="flight = the last token under the flight prompt, from sessions recorded with --save-activations")
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--layer_step", type=int, default=1)
    ap.add_argument("--alphas", nargs="*", type=float, default=pcfg["probe"]["alphas"])
    ap.add_argument("--bootstrap_n", type=int, default=pcfg["probe"]["bootstrap_n"])
    ap.add_argument("--no_mlp", action="store_true")
    ap.add_argument("--no_weights", action="store_true")
    ap.add_argument("--out", type=Path, default=common.RESULTS, help="kept for the plan's interface; runs go to results/<run_id>/")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    store = pl.ActStore(args.acts, args.source)
    man_path = args.manifest or Path(store.index["manifest"])
    man_path = man_path if man_path.is_absolute() else common.ROOT / man_path
    man, rows = store.align(pd.read_csv(man_path))
    specs, notes = pl.build_targets(args.targets, pcfg, man, args.split)
    for n in notes:
        print("note:", n)
    if not specs:
        sys.exit("no usable targets")
    ok = np.all([np.isfinite(t.Y).all(1) for t in specs], axis=0)
    if not ok.all():
        print(f"note: dropping {int((~ok).sum())} frames with a missing label")
        man, rows = man[ok].reset_index(drop=True), rows[ok]
        for t in specs:
            t.Y = t.Y[ok]
    clusters = man["traj_id"].to_numpy()
    cv = pcfg["probe"]["cv_folds"]

    run = common.new_run("probes", config={**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                           "acts_dir": str(store.path), "activations_run": store.index.get("run_id"),
                                           "manifest_sha256": store.index.get("manifest_sha256"), "n_frames": len(man)},
                         config_files=[common.CONFIGS / "probes.yaml"], seed=args.seed)
    layers = args.layers if args.layers else sorted(set(store.layers[::args.layer_step]) | {store.layers[-1]})
    layers = [l for l in layers if l in store.layers]
    main_specs = [t for t in specs if not t.per_scene]
    xy_specs = [t for t in specs if t.per_scene]
    folds = pl.make_folds(man, args.split, cv) if main_specs else []
    xy_folds = pl.make_folds(man, "within_scene", cv, per_scene=True) if xy_specs else []
    print(f"{store.model}/{store.source}: {len(man)} frames, {man.site_or_scene.nunique()} scenes, D={store.index['d']}, "
          f"layers={layers}, targets={[t.name for t in specs]}, split={args.split} ({len(folds)} folds)")

    base0 = {"model": store.model, "split": args.split, "source": store.source, "probe": "ridge", "seed": args.seed}
    out_csv = run / "probes.csv"
    probes_dir = common.ensure_dir(run / "probes")
    preds_dir = common.ensure_dir(run / "predictions")
    all_rows, edges = [], []
    t0 = time.time()
    for li, layer in enumerate(layers):
        for stem, position, prompt, tnames in feature_sets(store, layer, args.positions, args.prompts, [t.name for t in specs]):
            X = store.load(stem, rows)
            base = {**base0, "layer": layer, "position": position, "prompt": prompt}
            main_atab = []
            for group, gfolds, split_name in ((main_specs, folds, args.split), (xy_specs, xy_folds, "within_scene")):
                group = [t for t in group if t.name in tnames]
                if not group or not gfolds:
                    continue
                oof, _, fold_of, atab = pl.run_probe(X, group, gfolds, args.alphas, cv, seed=args.seed)
                edges += [(layer, position, prompt, f, tn, a) for (f, tn, a, e) in atab if e]
                if split_name == args.split:
                    main_atab = atab
                for t in group:
                    all_rows += pl.score_rows({**base, "split": split_name}, t, oof[t.name], fold_of, clusters, atab,
                                              args.bootstrap_n, args.seed, per_fold=(split_name == args.split))
                # held-out prediction for every frame, in the probe's own units (log metres for altitude, sin and
                # cos for angles): what the probe read on a frame it was not trained on
                np.savez(preds_dir / f"L{layer:02d}_{position}_{prompt}_{split_name}.npz",
                         frame_id=man.frame_id.to_numpy().astype(str), fold=fold_of.astype(str),
                         **{f"{t.name}_pred": oof[t.name].astype(np.float32) for t in group},
                         **{f"{t.name}_true": t.Y.astype(np.float32) for t in group})
            # weights for reuse: img_mean only, fitted on every frame at the typical chosen alpha
            if position == "img_mean" and not args.no_weights and main_specs:
                grp = [t for t in main_specs if t.name in tnames]
                a_by_t = {t.name: [a for (_, tn, a, _) in main_atab if tn == t.name] for t in grp}
                for t in grp:
                    a = float(np.exp(np.median(np.log(a_by_t[t.name])))) if a_by_t.get(t.name) else float(np.median(args.alphas))
                    a = min(args.alphas, key=lambda v: abs(np.log(v) - np.log(a)))
                    fit = pl.ridge_cv(X, t.Y, clusters, None, [a], [list(range(t.Y.shape[1]))], cv, return_weights=True)
                    w = fit.W.T.astype(np.float32)
                    np.savez(probes_dir / f"{store.model}_{t.name}_L{layer:02d}_{args.split}.npz",
                             w=w[0] if w.shape[0] == 1 else w, b=fit.b.astype(np.float32) if len(fit.b) > 1 else np.float32(fit.b[0]),
                             alpha=a, encoding=t.enc, position=position, layer=layer, target=t.name, model=store.model)
        pd.DataFrame(all_rows, columns=pl.CSV_COLUMNS).to_csv(out_csv, index=False)
        el = time.time() - t0
        print(f"  layer {layer:02d} done  ({li + 1}/{len(layers)}, {el / 60:.1f} min, eta {el / (li + 1) * (len(layers) - li - 1) / 60:.0f} min)")
    common.log_compute(run, "probes", time.time() - t0)

    df = pd.DataFrame(all_rows, columns=pl.CSV_COLUMNS)
    pooled = df[df.fold == "pooled"]
    best = []
    for t in specs:
        hm = pl.headline_metric(t)
        d = pooled[(pooled.target == t.name) & (pooled.metric == hm)]
        for (position, prompt), g in d.groupby(["position", "prompt"]):
            r = g.loc[g.value.idxmin()] if hm in pl.LOWER_IS_BETTER else g.loc[g.value.idxmax()]
            best.append({"target": t.name, "position": position, "prompt": prompt, "metric": hm, "best_layer": int(r.layer),
                         "value": r.value, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "alpha": r.alpha, "split": r.split})
    best = pd.DataFrame(best)
    best.to_csv(run / "best_layers.csv", index=False)

    # linearity check: MLP on the best img_mean layer per target
    if pcfg["probe"].get("mlp_check_best_layer") and not args.no_mlp:
        for t in main_specs:
            b = best[(best.target == t.name) & (best.position == "img_mean")]
            if b.empty:
                continue
            layer = int(b.iloc[0].best_layer)
            X = store.load(f"layer_{layer:02d}_img_mean", rows)
            pred = mlp_check(X, t, folds, args.seed)
            fold_of = np.array([""] * len(man), dtype=object)
            for f in folds:
                fold_of[f.test] = f.name
            all_rows += pl.score_rows({**base0, "layer": layer, "position": "img_mean", "prompt": "any", "probe": "mlp"}, t,
                                      pred, fold_of, clusters, [], args.bootstrap_n, args.seed, per_fold=False)
        df = pd.DataFrame(all_rows, columns=pl.CSV_COLUMNS)
        df.to_csv(out_csv, index=False)

    if edges:
        (run / "alpha_edges.txt").write_text(
            "alpha landed on the edge of the grid here; widen probe.alphas in configs/probes.yaml if this is the best layer\n"
            + "\n".join(f"layer {l:02d} {p} {pr} fold {f} target {tn} alpha {a:g}" for (l, p, pr, f, tn, a) in edges) + "\n")
        print(f"note: alpha hit the grid edge in {len(edges)} fits, see alpha_edges.txt")
    print("\nbest layer per target (pooled over held-out folds, 95% interval):")
    for _, r in best.iterrows():
        print(f"  {r.target:10s} {r.position:8s} {r.prompt:8s} L{int(r.best_layer):02d}  {r.metric} = {r.value:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]")
    mlp = df[(df.probe == "mlp") & (df.fold == "pooled")]
    for _, r in mlp.iterrows():
        if r.metric == pl.HEADLINE.get(next(t.enc for t in specs if t.name == r.target), "") or r.metric.startswith("mae"):
            print(f"  mlp check  {r.target:10s} L{int(r.layer):02d}  {r.metric} = {r.value:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]")
    print(f"\nnext: python scripts/31_baselines.py --manifest {man_path} --acts activations/ --probes_run results/{run.name} "
          f"--controls pixel vision_tower random_init shuffled --out results/")
    print(f"written {run}")


if __name__ == "__main__":
    main()
