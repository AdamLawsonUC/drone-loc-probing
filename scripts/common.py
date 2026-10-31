"""Shared helpers: run folders, provenance, seeds, model registry, decoder-layer lookup.

Every script calls new_run() first so results land in results/<run_id>/ with config.yaml,
git_hash.txt, seeds.txt and env.lock written automatically.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"


def git_hash(short: bool = True) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
        h = out.decode().strip()
        return h[:8] if short else h
    except Exception:
        return "nogit"


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def pip_freeze() -> str:
    try:
        return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL).decode()
    except Exception as e:  # pragma: no cover
        return f"# pip freeze failed: {e}\n"


def new_run(name: str, config: dict | None = None, config_files: list[Path] | None = None, seed: int = 0) -> Path:
    """Create results/<timestamp>-<githash>-<name>/ and write provenance. Returns the folder."""
    run_id = f"{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{git_hash()}-{name}"
    run = RESULTS / run_id
    run.mkdir(parents=True, exist_ok=False)
    (run / "git_hash.txt").write_text(git_hash(short=False) + "\n")
    (run / "seeds.txt").write_text(f"{seed}\n")
    (run / "env.lock").write_text(pip_freeze())
    (run / "command.txt").write_text(" ".join(sys.argv) + "\n")
    if config is not None:
        (run / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    for f in config_files or []:
        shutil.copy(f, run / f.name)
    (run / "compute_log.csv").write_text("timestamp,gpu,event,seconds\n")
    set_seed(seed)
    return run


def log_compute(run: Path, event: str, seconds: float) -> None:
    gpu = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    with (run / "compute_log.csv").open("a") as f:
        f.write(f"{_dt.datetime.now().isoformat(timespec='seconds')},{gpu},{event},{seconds:.3f}\n")


def load_yaml(path: Path | str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def model_entry(key: str) -> dict:
    reg = load_yaml(CONFIGS / "models.yaml")
    if key not in reg:
        raise KeyError(f"{key!r} not in configs/models.yaml (have: {', '.join(reg)})")
    return reg[key]


def find_decoder_layers(model):
    """Return the ModuleList of language-model decoder layers, whatever the HF wrapper calls it.

    Works for Qwen2.5-VL (model.model.layers or model.model.language_model.layers depending on the
    transformers version) and for OpenVLA / Prismatic (language_model.model.layers).
    """
    import torch.nn as nn

    cfg = getattr(model, "config", None)
    n = None
    for c in (getattr(cfg, "text_config", None), cfg):
        n = getattr(c, "num_hidden_layers", None) if c is not None else None
        if n:
            break
    candidates = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and name.endswith("layers") and len(mod) > 4:
            candidates.append((name, mod))
    if not candidates:
        raise RuntimeError("no decoder layer list found")
    if n:
        for name, mod in candidates:
            if len(mod) == n:
                return name, mod
    # fall back to the longest list
    return max(candidates, key=lambda t: len(t[1]))


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def env_flag(name: str, default: str = "") -> str:
    return os.environ.get(name, default)
