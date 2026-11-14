"""Probe machinery shared by 30_train_probes.py and 31_baselines.py.

Ridge on standardised features with the penalty chosen by grouped k-fold cross-validation inside the training
split. The Gram matrix of the training set is built once per fold from per-chunk Gram matrices, so every alpha,
every target and every shuffled-label control reuses the same O(D^2) statistics; only a Cholesky solve per alpha
is repeated. Results match sklearn Ridge(alpha) on StandardScaler features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

CSV_COLUMNS = ["model", "target", "layer", "position", "prompt", "split", "source", "fold", "probe", "metric",
               "value", "ci_lo", "ci_hi", "alpha", "n_test", "seed"]


# --------------------------------------------------------------------------------------------- targets
def encode_target(name: str, tcfg: dict, man: pd.DataFrame) -> np.ndarray:
    """Manifest columns -> the probe's regression target, shape (N, k). NaN where the label is missing."""
    enc, col = tcfg["encoding"], tcfg["column"]
    if enc == "log":
        return np.log(man[col].to_numpy(float))[:, None]
    if enc == "sincos":
        a = np.radians(man[col].to_numpy(float))
        return np.stack([np.sin(a), np.cos(a)], 1)
    if enc == "scalar":
        return man[col].to_numpy(float)[:, None]
    if enc == "xy":
        return man[list(col)].to_numpy(float)
    raise ValueError(f"unknown encoding {enc} for target {name}")


def metrics_for(enc: str, y: np.ndarray, p: np.ndarray, unit: str = "") -> dict[str, float]:
    """Metrics in the target's natural units. y and p are in the encoded space, shape (n, k)."""
    if len(y) < 3:
        return {}
    if enc == "log":
        return {"mae_m": float(np.mean(np.abs(np.exp(p[:, 0]) - np.exp(y[:, 0])))), "r2": _r2(y[:, 0], p[:, 0]),
                "spearman": _spearman(y[:, 0], p[:, 0])}
    if enc == "sincos":
        d = np.arctan2(p[:, 0], p[:, 1]) - np.arctan2(y[:, 0], y[:, 1])
        d = np.degrees(np.abs(np.arctan2(np.sin(d), np.cos(d))))
        return {"angular_mae_deg": float(d.mean())}
    if enc == "scalar":
        return {f"mae{unit}": float(np.mean(np.abs(p[:, 0] - y[:, 0]))), "r2": _r2(y[:, 0], p[:, 0]),
                "spearman": _spearman(y[:, 0], p[:, 0])}
    if enc == "xy":
        return {"median_m": float(np.median(np.linalg.norm(p - y, axis=1)))}
    raise ValueError(enc)


LOWER_IS_BETTER = {"mae_m", "angular_mae_deg", "mae_deg", "mae", "median_m"}
HEADLINE = {"log": "mae_m", "sincos": "angular_mae_deg", "scalar": "mae", "xy": "median_m"}


def _r2(y, p) -> float:
    ss = float(np.sum((y - y.mean()) ** 2))
    return float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan")


def _spearman(y, p) -> float:
    if np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(spearmanr(y, p).statistic)


def cluster_bootstrap(enc: str, y: np.ndarray, p: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int,
                      unit: str = "") -> dict[str, tuple[float, float]]:
    """95% interval for every metric, resampling whole trajectories (frames within one are not independent)."""
    if n_boot <= 0 or len(y) < 3:
        return {}
    rng = np.random.default_rng(seed)
    _, inv = np.unique(clusters, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv)
    mat = np.full((len(counts), counts.max()), -1, dtype=np.int64)
    pos = np.concatenate([np.arange(c) for c in counts])
    mat[inv[order], pos] = order
    vals: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = mat[rng.integers(0, len(counts), len(counts))].ravel()
        idx = idx[idx >= 0]
        for k, v in metrics_for(enc, y[idx], p[idx], unit).items():
            vals.setdefault(k, []).append(v)
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in vals.items()}


# --------------------------------------------------------------------------------------------- folds
@dataclass
class Fold:
    name: str
    train: np.ndarray
    test: np.ndarray
    inner_groups: np.ndarray     # group label per training row, for the alpha cross-validation


def make_folds(man: pd.DataFrame, split: str, cv: int = 5, per_scene: bool = False) -> list[Fold]:
    """scene_heldout: leave one scene out, rotating over every scene. The inner alpha search also leaves whole
    scenes out when at least three training scenes remain, otherwise whole trajectories.
    within_scene: grouped k-fold over trajectories, all scenes pooled; or one model per scene with per_scene=True
    (used for scene_xy, whose coordinates mean nothing across scenes)."""
    scene = man["site_or_scene"].to_numpy()
    traj = man["traj_id"].to_numpy()
    idx = np.arange(len(man))
    folds: list[Fold] = []
    if split in ("scene_heldout", "site_heldout") and not per_scene:
        scenes = sorted(pd.unique(scene))
        if len(scenes) < 2:
            raise ValueError("scene_heldout needs at least two scenes in the manifest")
        for s in scenes:
            tr, te = idx[scene != s], idx[scene == s]
            ig = scene[tr] if len(scenes) - 1 >= 3 else traj[tr]
            folds.append(Fold(str(s), tr, te, ig))
    elif per_scene:
        for s in sorted(pd.unique(scene)):
            sub = idx[scene == s]
            k = min(cv, len(pd.unique(traj[sub])))
            if k < 2:
                continue
            for j, (a, b) in enumerate(GroupKFold(k).split(sub, groups=traj[sub])):
                folds.append(Fold(f"{s}/cv{j}", sub[a], sub[b], traj[sub[a]]))
    elif split == "within_scene":
        k = min(cv, len(pd.unique(traj)))
        for j, (a, b) in enumerate(GroupKFold(k).split(idx, groups=traj)):
            folds.append(Fold(f"cv{j}", idx[a], idx[b], traj[a]))
    else:
        raise ValueError(f"unknown split {split}")
    return folds


# --------------------------------------------------------------------------------------------- ridge
@dataclass
class FitResult:
    pred: np.ndarray                      # (n_test, k_total) predictions in the encoded target space
    alphas: list[float]                   # chosen alpha per column group
    W: np.ndarray | None = None           # (D, k_total) weights on raw (unstandardised) features
    b: np.ndarray | None = None           # (k_total,)
    edge: list[bool] = field(default_factory=list)   # alpha landed on the edge of the grid


def ridge_cv(Xtr: np.ndarray, Ytr: np.ndarray, groups: np.ndarray, Xte: np.ndarray | None, alphas: list[float],
             col_groups: list[list[int]], cv: int = 5, return_weights: bool = False) -> FitResult:
    """Ridge with a separate alpha per column group (one group per target), chosen by grouped k-fold CV.

    Standardisation uses the whole training split, including inside the alpha search (the inner folds share the
    outer-train mean and scale; they never see test rows)."""
    Xtr = np.asarray(Xtr, dtype=np.float64)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-8] = 1.0
    Z = (Xtr - mu) / sd
    ymu = Ytr.mean(0)
    Yc = Ytr - ymu
    D = Z.shape[1]
    eye = np.eye(D)
    n_groups = len(np.unique(groups))
    k = min(cv, n_groups)
    if k >= 2 and len(alphas) > 1:
        chunk = np.empty(len(Z), dtype=int)
        for j, (_, b) in enumerate(GroupKFold(k).split(Z, groups=groups)):
            chunk[b] = j
        Gs = [Z[chunk == j].T @ Z[chunk == j] for j in range(k)]
        Bs = [Z[chunk == j].T @ Yc[chunk == j] for j in range(k)]
        G, B = sum(Gs), sum(Bs)
        err = np.zeros((len(alphas), Yc.shape[1]))
        for j in range(k):
            Gtr, Btr = G - Gs[j], B - Bs[j]
            Zv, Yv = Z[chunk == j], Yc[chunk == j]
            for a_i, a in enumerate(alphas):
                W = cho_solve(cho_factor(Gtr + a * eye, overwrite_a=True, check_finite=False), Btr, check_finite=False)
                err[a_i] += ((Zv @ W - Yv) ** 2).sum(0)
        del Gs
        var = Yc.var(0) * len(Yc)
        var[var == 0] = 1.0
        err = err / var
        best = [int(np.argmin(err[:, cols].sum(1))) for cols in col_groups]
    else:
        G, B = Z.T @ Z, Z.T @ Yc
        best = [len(alphas) // 2 for _ in col_groups]
    W = np.zeros((D, Yc.shape[1]))
    for a_i in sorted(set(best)):
        cols = [c for g, cg in zip(best, col_groups) if g == a_i for c in cg]
        W[:, cols] = cho_solve(cho_factor(G + alphas[a_i] * eye, check_finite=False), B[:, cols], check_finite=False)
    pred = None
    if Xte is not None:
        pred = ((np.asarray(Xte, dtype=np.float64) - mu) / sd) @ W + ymu
    res = FitResult(pred=pred, alphas=[float(alphas[i]) for i in best],
                    edge=[len(alphas) > 2 and i in (0, len(alphas) - 1) for i in best])
    if return_weights:
        res.W = W / sd[:, None]
        res.b = ymu - mu @ res.W
    return res


@dataclass
class TargetSpec:
    name: str
    enc: str
    Y: np.ndarray        # (N, k) encoded labels
    unit: str = ""       # suffix for scalar MAE names, e.g. "_deg"
    per_scene: bool = False


def run_probe(X: np.ndarray, targets: list[TargetSpec], folds: list[Fold], alphas: list[float], cv: int,
              n_shuffle: int = 0, seed: int = 0):
    """Out-of-fold predictions for every target over the given folds.

    Returns (oof, shuf, fold_of_row, alpha_table):
      oof[target]        (N, k) predictions, NaN where a row was never in a test fold
      shuf[target]       (n_shuffle, N, k) the same from probes trained on label-permuted training sets
      fold_of_row        (N,) name of the fold in which each row was tested
      alpha_table        list of (fold, target, alpha, on_edge)
    """
    N = len(X)
    cols, col_groups, c = {}, [], 0
    for t in targets:
        k = t.Y.shape[1]
        cols[t.name] = list(range(c, c + k))
        col_groups.append(cols[t.name])
        c += k
    Yall = np.concatenate([t.Y for t in targets], 1)
    k_real = Yall.shape[1]
    oof = {t.name: np.full((N, t.Y.shape[1]), np.nan) for t in targets}
    shuf = {t.name: np.full((n_shuffle, N, t.Y.shape[1]), np.nan) for t in targets}
    fold_of = np.array([""] * N, dtype=object)
    alpha_table = []
    rng = np.random.default_rng(seed)
    for f in folds:
        Ytr = Yall[f.train]
        groups_all = list(col_groups)
        if n_shuffle:
            extra = []
            for s in range(n_shuffle):
                perm = rng.permutation(len(f.train))
                extra.append(Ytr[perm])
                groups_all += [[k_real * (s + 1) + j for j in cg] for cg in col_groups]
            Ytr = np.concatenate([Ytr] + extra, 1)
        res = ridge_cv(X[f.train], Ytr, f.inner_groups, X[f.test], alphas, groups_all, cv)
        fold_of[f.test] = f.name
        for gi, t in enumerate(targets):
            oof[t.name][f.test] = res.pred[:, cols[t.name]]
            alpha_table.append((f.name, t.name, res.alphas[gi], res.edge[gi]))
            for s in range(n_shuffle):
                shuf[t.name][s, f.test] = res.pred[:, [k_real * (s + 1) + j for j in cols[t.name]]]
    return oof, shuf, fold_of, alpha_table


def score_rows(base: dict, t: TargetSpec, pred: np.ndarray, fold_of: np.ndarray, clusters: np.ndarray,
               alpha_table: list, n_boot: int, seed: int, per_fold: bool = True) -> list[dict]:
    """Long-format rows: one pooled row per metric with a cluster-bootstrap interval, plus one row per fold."""
    ok = np.isfinite(pred).all(1) & np.isfinite(t.Y).all(1)
    rows = []
    al = [a for (_, tn, a, _) in alpha_table if tn == t.name]
    alpha = float(np.exp(np.mean(np.log(al)))) if al else float("nan")
    m = metrics_for(t.enc, t.Y[ok], pred[ok], t.unit)
    ci = cluster_bootstrap(t.enc, t.Y[ok], pred[ok], clusters[ok], n_boot, seed, t.unit)
    for k, v in m.items():
        lo, hi = ci.get(k, (np.nan, np.nan))
        rows.append({**base, "target": t.name, "fold": "pooled", "metric": k, "value": v, "ci_lo": lo, "ci_hi": hi,
                     "alpha": alpha, "n_test": int(ok.sum())})
    if per_fold:
        fa = {fn: a for (fn, tn, a, _) in alpha_table if tn == t.name}
        for fn in pd.unique(fold_of[ok]):
            sel = ok & (fold_of == fn)
            for k, v in metrics_for(t.enc, t.Y[sel], pred[sel], t.unit).items():
                rows.append({**base, "target": t.name, "fold": fn, "metric": k, "value": v, "ci_lo": np.nan,
                             "ci_hi": np.nan, "alpha": fa.get(fn, np.nan), "n_test": int(sel.sum())})
    return rows


# --------------------------------------------------------------------------------------------- activation store
class ActStore:
    """Reads activations/<model>/<source>/ written by 20_extract_activations.py."""

    def __init__(self, path: Path, source: str | None = None):
        path = Path(path)
        if not (path / "index.json").exists():
            subs = [d for d in sorted(path.iterdir()) if (d / "index.json").exists()] if path.exists() else []
            if source:
                subs = [d for d in subs if d.name == source]
            subs = [d for d in subs if not d.name.endswith("-smoke")] or subs
            if len(subs) != 1:
                raise FileNotFoundError(f"expected index.json in {path} or in exactly one subfolder "
                                        f"(found {[d.name for d in subs]}); pass --source")
            path = subs[0]
        import json
        self.path = path
        self.index = json.loads((path / "index.json").read_text())
        self.frame_ids = [s for s in (path / "frame_ids.txt").read_text().splitlines() if s]
        self.layers: list[int] = list(self.index["layers"])
        self.model: str = self.index["model"] + ("-randinit" if self.index.get("random_init") else "")
        self.source: str = self.index.get("source", path.name)

    def has(self, name: str) -> bool:
        return (self.path / f"{name}.npy").exists()

    def load(self, name: str, rows: np.ndarray | None = None) -> np.ndarray:
        a = np.load(self.path / f"{name}.npy", mmap_mode="r")
        a = a if rows is None else a[rows]
        return np.asarray(a, dtype=np.float64)

    def align(self, man: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Manifest rows in activation order, and the activation row index for each."""
        pos = {f: i for i, f in enumerate(self.frame_ids)}
        man = man[man.frame_id.astype(str).isin(pos)].copy()
        rows = man.frame_id.astype(str).map(pos).to_numpy()
        order = np.argsort(rows)
        return man.iloc[order].reset_index(drop=True), rows[order]


def build_targets(names: list[str], pcfg: dict, man: pd.DataFrame, split: str) -> tuple[list[TargetSpec], list[str]]:
    """TargetSpecs for the requested targets that have usable labels; returns (specs, messages about the rest)."""
    specs, notes = [], []
    for n in names:
        if n not in pcfg["targets"]:
            notes.append(f"{n}: not in configs/probes.yaml, skipped")
            continue
        tc = pcfg["targets"][n]
        cols = tc["column"] if isinstance(tc["column"], list) else [tc["column"]]
        if any(c not in man.columns for c in cols) or man[cols].isna().all().any():
            notes.append(f"{n}: no labels in this manifest ({', '.join(cols)} empty), skipped")
            continue
        Y = encode_target(n, tc, man)
        if np.nanstd(Y, axis=0).max() == 0:
            notes.append(f"{n}: label is constant in this manifest, skipped")
            continue
        unit = "_deg" if n == "pitch" else ""
        specs.append(TargetSpec(n, tc["encoding"], Y, unit, per_scene=(tc["encoding"] == "xy" and not tc.get("needs_tile"))))
    return specs, notes


def headline_metric(t: TargetSpec) -> str:
    h = HEADLINE[t.enc]
    return h + t.unit if t.enc == "scalar" else h
