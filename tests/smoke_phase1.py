#!/usr/bin/env python
"""Probe pipeline smoke test without a GPU, a drone or any data: fake sites, fake activations, scripts 30 and 31.

  python tests/smoke_phase1.py            # under a minute; prints PASS at the end

It writes a small manifest with the real-flight columns (four fake sites, altitude, heading, pitch), a few
textured frames, and activations in the format script 20 produces, with altitude, heading and pitch planted
linearly at layer 2 on top of a per-site offset and noise. It then checks that script 30 finds layer 2 and saves
weights that read altitude back out, and that script 31 produces every control and the gate table.
The runs it creates in results/ end in -probes and -baselines; delete them afterwards.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def sh(*a):
    print("$", " ".join(str(x) for x in a))
    r = subprocess.run([PY, *map(str, a)], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-2500:])
    if r.returncode:
        print(r.stderr[-3000:])
        sys.exit("FAIL")
    return r.stdout


def main():
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp(prefix="dlp_smoke_"))
    rows = []
    for site in ("site_a", "site_b", "site_c", "site_d"):
        for t in range(25):
            alt, yaw, pitch = rng.uniform(15, 90), rng.uniform(0, 360), rng.choice([-90.0, -45.0])
            for i in range(4):
                a = alt + rng.normal(0, 0.5)
                f = tmp / "frames" / site / f"f{t:02d}_{i}.png"
                f.parent.mkdir(parents=True, exist_ok=True)
                scale = max(2, int(64 / (1 + a / 10)))                  # coarser texture when lower
                tex = rng.integers(0, 255, (max(96 // scale, 1), max(128 // scale, 1), 3), dtype=np.uint8)
                Image.fromarray(tex).resize((128, 96), Image.NEAREST).save(f)
                rows.append({"frame_id": f"{site}__f{t:02d}__{i}", "site_or_scene": site, "split": "", "x": rng.uniform(-200, 200),
                             "y": rng.uniform(-200, 200), "altitude_agl": a, "yaw": yaw, "pitch": pitch, "source": "real",
                             "traj_id": f"{site}__f{t:02d}", "image_path": str(f)})
    man = pd.DataFrame(rows)
    man_path = tmp / "manifest_real.csv"
    man.to_csv(man_path, index=False)

    # fake activations in script 20's format
    N, D, layers = len(man), 96, [0, 1, 2, 3]
    acts = tmp / "activations"
    for model, planted in (("fake-vlm", True), ("fake-vlm-randinit", False)):
        out = acts / model / "real"
        out.mkdir(parents=True)
        dirs = rng.normal(size=(4, D))
        y = np.stack([np.log(man.altitude_agl), np.sin(np.radians(man.yaw)), np.cos(np.radians(man.yaw)), man.pitch], 1)
        y = (y - y.mean(0)) / y.std(0)
        site_off = {s: rng.normal(size=D) for s in man.site_or_scene.unique()}
        off = np.stack([site_off[s] for s in man.site_or_scene])
        for l in layers:
            gain = [0.0, 1.5, 3.0, 1.0][l] if planted else 0.0
            for stem in (f"layer_{l:02d}_img_mean", f"layer_{l:02d}_last_neutral", f"layer_{l:02d}_last_targeted-altitude"):
                np.save(out / f"{stem}.npy", (gain * y @ dirs + off + rng.normal(size=(N, D))).astype(np.float16))
        np.save(out / "vision_tower_img_mean.npy", (0.3 * y @ dirs + off + rng.normal(size=(N, D))).astype(np.float16))
        (out / "frame_ids.txt").write_text("\n".join(man.frame_id) + "\n")
        (out / "index.json").write_text(json.dumps({"model": "fake-vlm", "random_init": not planted, "source": "real",
                                                    "manifest": str(man_path), "n": N, "d": D, "layers": layers}))

    o = sh("scripts/30_train_probes.py", "--acts", acts / "fake-vlm", "--targets", "altitude", "heading", "pitch", "scene_xy",
           "--split", "scene_heldout", "--bootstrap_n", 100)
    run30 = Path(o.strip().splitlines()[-1].split("written ")[-1])
    best = pd.read_csv(run30 / "best_layers.csv")
    assert set(best.target) == {"altitude", "heading", "pitch", "scene_xy"}, set(best.target)
    b = best[(best.target == "altitude") & (best.position == "img_mean")].iloc[0]
    assert b.best_layer == 2, f"expected the planted layer 2, got {b.best_layer}"
    z = np.load(run30 / "probes" / "fake-vlm_altitude_L02_scene_heldout.npz")
    X = np.load(acts / "fake-vlm" / "real" / "layer_02_img_mean.npy").astype(np.float64)
    r = np.corrcoef(X @ z["w"] + z["b"], np.log(man.altitude_agl))[0, 1]
    assert z["w"].shape == (D,) and r > 0.9, f"saved weights do not read out log altitude (r={r:.2f})"

    o = sh("scripts/31_baselines.py", "--manifest", man_path, "--acts", acts, "--model", "fake-vlm", "--probes_run", run30,
           "--controls", "pixel", "vision_tower", "random_init", "shuffled", "--shuffles", 3, "--bootstrap_n", 100)
    run31 = Path(o.strip().splitlines()[-1].split("written ")[-1])
    base = pd.read_csv(run31 / "baselines.csv")
    assert set(base.control) == {"pixel", "vision_tower", "random_init", "shuffled"}, set(base.control)
    assert (run31 / "gate_g1.csv").exists()
    chance = base[(base.control == "shuffled") & (base.target == "heading")].value.mean()
    assert 70 < chance < 110, f"shuffled heading error should sit near 90 deg, got {chance:.0f}"
    shutil.rmtree(tmp)
    print("PASS")


if __name__ == "__main__":
    main()
