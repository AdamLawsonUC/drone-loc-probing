#!/usr/bin/env python
"""Phase 0 environment check.

Usage:
  python scripts/00_check_env.py                              # environment and GPU only
  python scripts/00_check_env.py --models qwen2.5-vl-7b aerovla   # also load, hook and time each model

Prints one line per check and one line per model (loaded, decoder layers, forward-pass seconds), and writes
results/<run_id>/env.lock plus check_env.json. Gate G0 is: every requested model prints "ok".
"""
from __future__ import annotations

import argparse
import importlib
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def check_import(name: str) -> tuple[bool, str]:
    try:
        m = importlib.import_module(name)
        return True, getattr(m, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        return False, str(e).splitlines()[0][:80]


def gpu_report() -> dict:
    rep = {"cuda": False}
    try:
        import torch
        rep["torch"] = torch.__version__
        rep["cuda"] = torch.cuda.is_available()
        if rep["cuda"]:
            rep["device"] = torch.cuda.get_device_name(0)
            rep["memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
            rep["cuda_build"] = torch.version.cuda
            rep["bf16"] = torch.cuda.is_bf16_supported()
    except Exception as e:  # noqa: BLE001
        rep["torch_error"] = str(e)[:120]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], stderr=subprocess.DEVNULL
        )
        rep["driver"] = out.decode().strip().splitlines()[0]
    except Exception:
        rep["driver"] = "nvidia-smi unavailable"
    return rep


def dummy_image(side: int = 448):
    from PIL import Image
    import numpy as np
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (side, side, 3), dtype="uint8"))


def load_and_time(key: str) -> dict:
    """Load one model, find its decoder layers, register a no-op hook, time a forward pass."""
    import torch

    entry = common.model_entry(key)
    dtype = getattr(torch, entry.get("dtype", "bfloat16"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    fam = entry["family"]
    if fam == "qwen2_5_vl":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(entry["hf_id"], torch_dtype=dtype, device_map=device)
        processor = AutoProcessor.from_pretrained(entry["hf_id"])
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the scene."}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[dummy_image()], return_tensors="pt").to(model.device)
    elif fam == "openvla":
        from transformers import AutoModelForVision2Seq, AutoProcessor
        model = AutoModelForVision2Seq.from_pretrained(
            entry["hf_id"], torch_dtype=dtype, trust_remote_code=entry.get("trust_remote_code", True)
        ).to(device)
        processor = AutoProcessor.from_pretrained(entry["hf_id"], trust_remote_code=True)
        inputs = processor("In: What action should the robot take to hold altitude?\nOut:", dummy_image(224)).to(
            model.device, dtype=dtype
        )
    else:
        raise ValueError(f"unknown family {fam}")
    load_s = time.time() - t0
    model.eval()

    name, layers = common.find_decoder_layers(model)
    seen = []
    h = layers[len(layers) // 2].register_forward_hook(lambda m, i, o: seen.append(tuple((o[0] if isinstance(o, tuple) else o).shape)))

    with torch.inference_mode():
        _ = model(**inputs)  # warm-up
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        _ = model(**inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        fwd_s = time.time() - t1
    h.remove()
    rep = {
        "model": key,
        "status": "ok" if seen else "hook_failed",
        "layer_list": name,
        "n_layers": len(layers),
        "hidden": seen[0][-1] if seen else None,
        "load_s": round(load_s, 1),
        "forward_s": round(fwd_s, 3),
        "device": device,
    }
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help="keys from configs/models.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = common.new_run("check_env", config={"models": args.models}, seed=args.seed)
    report = {"python": platform.python_version(), "platform": platform.platform(), "gpu": gpu_report(), "imports": {}, "models": []}

    print(f"python {report['python']}  {report['platform']}")
    g = report["gpu"]
    print(f"torch {g.get('torch', '?')}  cuda={g.get('cuda')}  device={g.get('device', '-')}  mem={g.get('memory_gb', '-')} GB  driver={g.get('driver')}  bf16={g.get('bf16', '-')}")
    for pkg in ["transformers", "accelerate", "nnsight", "sklearn", "fastapi", "uvicorn", "PIL", "yaml", "pandas"]:
        ok, ver = check_import(pkg)
        report["imports"][pkg] = {"ok": ok, "version": ver}
        print(f"{'ok ' if ok else 'MISSING'} {pkg:12s} {ver}")

    for key in args.models:
        try:
            rep = load_and_time(key)
        except Exception as e:  # noqa: BLE001
            rep = {"model": key, "status": "error", "error": str(e)[:200]}
        report["models"].append(rep)
        if rep["status"] == "ok":
            print(f"ok  {key:14s} layers={rep['n_layers']} ({rep['layer_list']}) hidden={rep['hidden']} load={rep['load_s']}s forward={rep['forward_s']}s")
        else:
            print(f"FAIL {key:14s} {rep.get('error', rep['status'])}")

    common.dump_json(run / "check_env.json", report)
    print(f"written {run}")


if __name__ == "__main__":
    main()
