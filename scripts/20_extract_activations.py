#!/usr/bin/env python
"""Cache residual-stream activations for every frame in a manifest. One frozen model, a few forward passes per frame.

Usage (on wolfgang, in the container):
  docker/run.sh scripts/20_extract_activations.py --model qwen2.5-vl-7b --manifest data/manifest.csv \
      --positions img_mean last --prompts neutral targeted --out activations/ --limit 8      # smoke test first
  docker/run.sh scripts/20_extract_activations.py --model qwen2.5-vl-7b --manifest data/manifest.csv \
      --positions img_mean last --prompts neutral targeted --out activations/
  docker/run.sh scripts/20_extract_activations.py --model qwen2.5-vl-7b --manifest data/manifest.csv \
      --positions img_mean last --prompts neutral --random_init --out activations/           # control 2

Produces, under activations/<model>/<source>/ (source is the manifest's `source` value, e.g. real):
  layer_XX_img_mean.npy               (N, D) mean over the drone-view image tokens at the output of decoder layer XX
  layer_XX_last_neutral.npy           (N, D) last prompt token, neutral prompt ("Describe the scene.")
  layer_XX_last_targeted-<target>.npy (N, D) last prompt token, targeted prompt for that target
  vision_tower_img_mean.npy           (N, D) mean over the image tokens as they enter decoder layer 0, i.e. the
                                      vision tower plus projector output, before any language-model layer
  frame_ids.txt, index.json           row order, settings and a sha256 per file
With --random_init the folder is activations/<model>-randinit/<source>/ and the decoder layers are re-initialised
(vision tower and embeddings kept).

Design notes
  - Layer XX is the raw output of decoder layer XX (a forward hook), the same tensor 80_flight_server.py steers.
  - Image tokens come before the prompt text and attention is causal, so img_mean does not depend on the prompt.
    It is stored once, from the neutral pass.
  - Neutral prompt: the text is the user turn; "last" is the final token of the generation prompt.
    Targeted prompt: the user turn is the image alone and the text is given as the start of the assistant's
    answer, so "last" is the final token of e.g. "The drone's altitude in metres is".
  - Resumable: rerun the same command with --resume after an interruption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
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


def reinit_decoder(layers, seed: int) -> None:
    """Control 2: random language-model layers, trained vision tower. Matrices N(0, 0.02), norms 1, biases 0."""
    import torch
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in layers.named_parameters():
            if p.dim() >= 2:
                p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32).mul_(0.02).to(p.device, p.dtype))
            elif "bias" in name:
                p.zero_()
            else:
                p.fill_(1.0)


class QwenExtractor:
    def __init__(self, key: str, random_init: bool, seed: int):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        entry = common.model_entry(key)
        if entry["family"] != "qwen2_5_vl":
            raise NotImplementedError(f"{key}: family {entry['family']} is not wired into this script yet "
                                      f"(AeroVLA / OpenVLA extraction is added in Phase 3)")
        self.torch = torch
        self.entry = entry
        dtype = getattr(torch, entry.get("dtype", "bfloat16"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(entry["hf_id"], torch_dtype=dtype, device_map=self.device)
        self.processor = AutoProcessor.from_pretrained(entry["hf_id"])
        self.model.eval()
        self.layer_name, self.layers = common.find_decoder_layers(self.model)
        if random_init:
            reinit_decoder(self.layers, seed)
        self.image_token_id = getattr(self.model.config, "image_token_id", None)
        if self.image_token_id is None:
            self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        self._mask = None
        self._img, self._last, self._vt = {}, {}, None
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(self._make_hook(i))
        self.layers[0].register_forward_pre_hook(self._pre_hook, with_kwargs=True)

    def _make_hook(self, i: int):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            h = hs[0]
            self._img[i] = h[self._mask].float().mean(0).cpu()
            self._last[i] = h[-1].float().cpu()
        return hook

    def _pre_hook(self, module, args, kwargs):
        hs = args[0] if args else kwargs.get("hidden_states")
        if hs is not None:
            self._vt = hs[0][self._mask].float().mean(0).cpu()

    def build_text(self, prompt: str, mode: str) -> str:
        if mode == "user":
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        msgs = [{"role": "user", "content": [{"type": "image"}]}]
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prompt

    def forward(self, image, text: str):
        """Returns (img_mean [L, D], last [L, D], vision_tower [D], n_image_tokens)."""
        torch = self.torch
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        self._mask = inputs["input_ids"][0] == self.image_token_id
        n_img = int(self._mask.sum())
        if n_img == 0:
            raise RuntimeError("no image tokens found in the prompt; check image_token_id")
        self._img.clear(); self._last.clear(); self._vt = None
        with torch.inference_mode():
            self.model(**inputs)
        L = len(self.layers)
        img = torch.stack([self._img[i] for i in range(L)]).numpy()
        last = torch.stack([self._last[i] for i in range(L)]).numpy()
        vt = self._vt.numpy() if self._vt is not None else None
        return img, last, vt, n_img


def main() -> None:
    pcfg = common.load_yaml(common.CONFIGS / "probes.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="key from configs/models.yaml")
    ap.add_argument("--manifest", type=Path, default=common.ROOT / "data" / "manifest.csv")
    ap.add_argument("--positions", nargs="+", default=["img_mean", "last"], choices=["img_mean", "last"])
    ap.add_argument("--prompts", nargs="+", default=["neutral", "targeted"], choices=["neutral", "targeted"])
    ap.add_argument("--targets", nargs="*", default=None,
                    help="targets that get a targeted prompt; default: every target in probes.yaml that has one and needs no tile")
    ap.add_argument("--out", type=Path, default=common.ROOT / "activations")
    ap.add_argument("--source", default=None, help="subfolder name; default: the manifest's source column")
    ap.add_argument("--image_side", type=int, default=448, help="longest image side in pixels before the processor")
    ap.add_argument("--layer_step", type=int, default=1, help="store every n-th layer (the last layer is always kept)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--limit", type=int, default=None, help="first n frames only, for a smoke test")
    ap.add_argument("--random_init", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image

    man = pd.read_csv(args.manifest)
    if args.limit:
        man = man.iloc[:args.limit].reset_index(drop=True)
    source = args.source or str(man["source"].mode().iloc[0])
    model_dir = args.model + ("-randinit" if args.random_init else "")
    out = common.ensure_dir(args.out / model_dir / (source + ("-smoke" if args.limit else "")))

    # forward passes: (label, text, mode)
    passes = []
    if "neutral" in args.prompts:
        passes.append(("neutral", pcfg["prompts"]["neutral"], "user"))
    if "targeted" in args.prompts and "last" in args.positions:
        tp = pcfg["prompts"]["targeted"]
        tgts = args.targets or [t for t in tp if not pcfg["targets"][t].get("needs_tile")]
        passes += [(f"targeted-{t}", tp[t], "prefill") for t in tgts]
    if not passes:
        sys.exit("nothing to do: --prompts targeted needs --positions last")

    run = common.new_run("extract", config={**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                            "passes": passes, "out_dir": str(out)},
                         config_files=[common.CONFIGS / "probes.yaml", common.CONFIGS / "models.yaml"], seed=args.seed)
    t_load = time.time()
    ex = QwenExtractor(args.model, args.random_init, args.seed)
    common.log_compute(run, "load_model", time.time() - t_load)
    L = len(ex.layers)
    keep_layers = sorted(set(range(0, L, args.layer_step)) | {L - 1})
    N = len(man)
    print(f"model {args.model} layers={L} ({ex.layer_name}) keep={len(keep_layers)} frames={N} passes={[p[0] for p in passes]} -> {out}")

    def load_image(row):
        p = Path(row.image_path)
        im = Image.open(p if p.is_absolute() else common.ROOT / p).convert("RGB")
        im.thumbnail((args.image_side, args.image_side))
        return im

    # one probe pass to learn D, then open the memmaps
    texts = {label: ex.build_text(text, mode) for label, text, mode in passes}
    img0, _, _, n_img0 = ex.forward(load_image(man.iloc[0]), texts[passes[0][0]])
    D = img0.shape[1]
    progress_file = out / "progress.json"
    start = 0
    if args.resume and progress_file.exists():
        prog = json.loads(progress_file.read_text())
        if prog.get("n") == N and prog.get("d") == D and prog.get("layers") == keep_layers:
            start = int(prog["done"])
            print(f"resuming at frame {start}")
        else:
            print("progress.json does not match this run; starting again")
    mode = "r+" if start > 0 else "w+"

    def mm(name: str):
        f = out / name
        if mode == "r+" and f.exists():
            return np.lib.format.open_memmap(f, mode="r+")
        return np.lib.format.open_memmap(f, mode="w+", dtype=args.dtype, shape=(N, D))

    files: dict[str, np.memmap] = {}
    store_img = "img_mean" in args.positions
    store_last = "last" in args.positions
    if store_img:
        files["vision_tower_img_mean"] = mm("vision_tower_img_mean.npy")
        for l in keep_layers:
            files[f"layer_{l:02d}_img_mean"] = mm(f"layer_{l:02d}_img_mean.npy")
    if store_last:
        for label, _, _ in passes:
            for l in keep_layers:
                files[f"layer_{l:02d}_last_{label}"] = mm(f"layer_{l:02d}_last_{label}.npy")
    (out / "frame_ids.txt").write_text("\n".join(man.frame_id.astype(str)) + "\n")

    fmax = float(np.finfo(args.dtype).max)
    clipped = 0

    def put(name: str, i: int, v: np.ndarray) -> None:
        nonlocal clipped
        clipped += int((np.abs(v) > fmax).sum())
        files[name][i] = np.clip(v, -fmax, fmax)

    t0 = time.time()
    n_img_tokens = []
    for i in range(start, N):
        image = load_image(man.iloc[i])
        for k, (label, _, _) in enumerate(passes):
            img, last, vt, n_img = ex.forward(image, texts[label])
            if k == 0:
                n_img_tokens.append(n_img)
                if store_img:
                    if vt is not None:
                        put("vision_tower_img_mean", i, vt)
                    for l in keep_layers:
                        put(f"layer_{l:02d}_img_mean", i, img[l])
            if store_last:
                for l in keep_layers:
                    put(f"layer_{l:02d}_last_{label}", i, last[l])
        if (i + 1) % 200 == 0 or i + 1 == N:
            for f in files.values():
                f.flush()
            progress_file.write_text(json.dumps({"n": N, "d": D, "layers": keep_layers, "done": i + 1}))
            el = time.time() - t0
            rate = (i + 1 - start) / el
            print(f"  {i + 1}/{N}  {rate:.2f} frames/s  eta {(N - i - 1) / max(rate, 1e-9) / 60:.0f} min")
    common.log_compute(run, "extract", time.time() - t0)
    for f in files.values():
        f.flush()
    del files
    if clipped:
        print(f"WARNING: {clipped} values exceeded {args.dtype} range and were clipped; rerun with --dtype float32")

    index = {
        "model": args.model, "hf_id": ex.entry["hf_id"], "random_init": args.random_init, "source": source,
        "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest), "n": N, "d": D,
        "n_layers": L, "layers": keep_layers, "layer_list": ex.layer_name, "positions": args.positions,
        "passes": [{"label": a, "text": b, "mode": c} for a, b, c in passes],
        "image_side": args.image_side, "image_tokens_median": int(np.median(n_img_tokens)) if n_img_tokens else n_img0,
        "dtype": args.dtype, "clipped_values": clipped, "git_hash": common.git_hash(short=False), "run_id": run.name,
        "sha256": {p.name: sha256(p) for p in sorted(out.glob("*.npy"))},
    }
    common.dump_json(out / "index.json", index)
    common.dump_json(run / "index.json", index)
    print(f"activations in {out}")
    print(f"written {run}")


if __name__ == "__main__":
    main()
