#!/usr/bin/env python
"""Flight server: receives a frame from the ground-station app, returns a text action and a vertical velocity.

Usage:
  python scripts/80_flight_server.py --dry-run --port 8765                 # no model; measures the network path
  python scripts/80_flight_server.py --model qwen2.5-vl-7b --port 8765     # real policy
  python scripts/80_flight_server.py --model qwen2.5-vl-7b --probes results/<run>/probes/ --steer altitude

Endpoints
  GET  /health                       -> {"status", "model", "steer_delta", "uptime_s"}
  POST /act                          multipart: image (jpeg/png), instruction (optional), telemetry (JSON string,
                                     optional: {"rel_alt": m, "yaw": deg, "ts": s})
                                     -> {"action", "vertical_mps", "raw_text", "probe_readout", "steer_delta",
                                         "server_ms", "frame_ms", "session"}
  POST /steer                        json {"delta": float}   shifts the probe readout by delta; 0 disables

Every /act call is appended to flights/<session>/commands.csv. The envelope in configs/flight.yaml is enforced
here as well as in the app: no climb above the ceiling, no descend below the floor, hold on stale frames.

Steering: with --probes and --steer <target>, the ridge direction w from <probes>/<model>_<target>_L<layer>_*.npz
is added to the residual stream at the chosen layer as delta * w / (w.w), which shifts a linear probe's readout
by exactly delta. The probe readout (w.x + b on the mean image token) is returned alongside the action.

Note: no `from __future__ import annotations` here; FastAPI needs the real UploadFile type at runtime.
"""
import argparse
import csv
import datetime as dt
import io
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ACTIONS = ("climb", "hold", "descend")


# ----------------------------------------------------------------------------------------------- policies
class DryRunPolicy:
    name = "dry-run"

    def act(self, image: Image.Image, instruction: str) -> tuple[str, Optional[float]]:
        return "hold", None


class SteeringHook:
    """Adds delta * w / (w.w) to the residual stream output of one decoder layer."""

    def __init__(self, w: np.ndarray, positions: str = "all"):
        self.w = w.astype(np.float32)
        self.scale = float(self.w @ self.w)
        self.delta = 0.0
        self.positions = positions
        self._tensor = None

    def __call__(self, module, inputs, output):
        if self.delta == 0.0:
            return output
        import torch
        hs = output[0] if isinstance(output, tuple) else output
        if self._tensor is None or self._tensor.device != hs.device or self._tensor.dtype != hs.dtype:
            self._tensor = torch.tensor(self.w / self.scale, device=hs.device, dtype=hs.dtype)
        hs = hs + self.delta * self._tensor
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs


class QwenPolicy:
    def __init__(self, key: str, cfg: dict, probes_dir: Optional[Path], steer_target: Optional[str]):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        entry = common.model_entry(key)
        self.name = key
        self.cfg = cfg
        dtype = getattr(torch, entry.get("dtype", "bfloat16"))
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(entry["hf_id"], torch_dtype=dtype, device_map="cuda")
        self.processor = AutoProcessor.from_pretrained(entry["hf_id"])
        self.model.eval()
        self.layer_name, self.layers = common.find_decoder_layers(self.model)

        self.hook: Optional[SteeringHook] = None
        self.probe_w = None
        self.probe_b = 0.0
        self.probe_layer = None
        self._readout = []
        if probes_dir and steer_target:
            self._load_probe(probes_dir, key, steer_target, cfg["steering"].get("layer"))

    def _load_probe(self, probes_dir: Path, key: str, target: str, layer: Optional[int]) -> None:
        pattern = f"{key}_{target}_L{layer:02d}_*.npz" if layer is not None else f"{key}_{target}_L*_*.npz"
        files = sorted(Path(probes_dir).glob(pattern))
        if not files:
            raise FileNotFoundError(f"no probe matching {pattern} in {probes_dir}")
        f = files[0]
        z = np.load(f)
        self.probe_w = z["w"].astype(np.float32)
        self.probe_b = float(z["b"]) if "b" in z else 0.0
        self.probe_layer = int(re.search(r"_L(\d+)_", f.name).group(1))
        self.hook = SteeringHook(self.probe_w, self.cfg["steering"].get("positions", "all"))
        self.layers[self.probe_layer].register_forward_hook(self.hook)
        self.layers[self.probe_layer].register_forward_hook(self._capture)
        print(f"steering ready: {f.name} at layer {self.probe_layer} ({self.layer_name})")

    def _capture(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] > 1:  # prefill pass only
            x = hs[0].float().mean(0).detach().cpu().numpy()
            self._readout.append(float(self.probe_w @ x + self.probe_b))
        return output

    def set_delta(self, delta: float) -> None:
        if self.hook is not None:
            self.hook.delta = float(delta)

    def act(self, image: Image.Image, instruction: str) -> tuple[str, Optional[float]]:
        import torch
        side = self.cfg["policy"]["image_max_side_px"]
        image.thumbnail((side, side))
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        self._readout.clear()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.cfg["policy"]["max_new_tokens"], do_sample=False)
        raw = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        readout = self._readout[0] if self._readout else None
        return raw, readout


# ----------------------------------------------------------------------------------------------- server
class Server:
    def __init__(self, policy, cfg: dict, session: str):
        self.policy = policy
        self.cfg = cfg
        self.session = session
        self.t_start = time.time()
        self.last_cmd_time = 0.0
        self.lock = threading.Lock()
        self.log_path = common.ensure_dir(common.ROOT / cfg["logging"]["session_dir"] / session) / "commands.csv"
        if not self.log_path.exists():
            self.log_path.write_text("timestamp,frame_ts,rel_alt,raw_text,action,vertical_mps,probe_readout,steer_delta,server_ms,frame_ms,note\n")

    def parse_action(self, raw: str) -> str:
        m = re.search(r"\b(climb|hold|descend)\b", raw.lower())
        return m.group(1) if m else "hold"

    def apply_envelope(self, action: str, telemetry: dict) -> tuple[str, str]:
        env = self.cfg["envelope"]
        note = ""
        alt = telemetry.get("rel_alt")
        ts = telemetry.get("ts")
        if ts is not None and (time.time() - float(ts)) > env["hold_on_stale_s"]:
            return "hold", "stale_frame"
        if alt is not None:
            if action == "climb" and alt >= env["ceiling_m"]:
                action, note = "hold", "at_ceiling"
            elif action == "descend" and alt <= env["floor_m"]:
                action, note = "hold", "at_floor"
        if action != "hold":
            min_gap = 1.0 / env["command_rate_hz"]
            if time.time() - self.last_cmd_time < min_gap:
                action, note = "hold", "rate_limited"
            else:
                self.last_cmd_time = time.time()
        return action, note

    def handle(self, image_bytes: bytes, instruction: Optional[str], telemetry: dict) -> dict:
        t0 = time.time()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        instr = instruction or self.cfg["policy"]["prompt"].format(**self.cfg["policy"])
        with self.lock:
            raw, readout = self.policy.act(img, instr)
            action = self.parse_action(raw)
            action, note = self.apply_envelope(action, telemetry)
        v = self.cfg["envelope"]["max_vertical_mps"]
        vertical = {"climb": v, "hold": 0.0, "descend": -v}[action]
        server_ms = (time.time() - t0) * 1000
        frame_ms = (time.time() - float(telemetry["ts"])) * 1000 if telemetry.get("ts") else None
        delta = getattr(getattr(self.policy, "hook", None), "delta", 0.0)
        row = [dt.datetime.now().isoformat(timespec="milliseconds"), telemetry.get("ts"), telemetry.get("rel_alt"),
               raw.replace("\n", " ").replace(",", ";"), action, vertical, readout, delta, round(server_ms, 1),
               None if frame_ms is None else round(frame_ms, 1), note]
        with self.log_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)
        return {"action": action, "vertical_mps": vertical, "raw_text": raw, "probe_readout": readout,
                "steer_delta": delta, "server_ms": round(server_ms, 1), "frame_ms": frame_ms, "session": self.session,
                "note": note}


def build_app(server: Server):
    from fastapi import FastAPI, File, Form, UploadFile
    app = FastAPI(title="drone-loc-probing flight server")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": server.policy.name,
                "steer_delta": getattr(getattr(server.policy, "hook", None), "delta", 0.0),
                "uptime_s": round(time.time() - server.t_start, 1), "session": server.session}

    @app.post("/act")
    async def act(image: UploadFile = File(...), instruction: Optional[str] = Form(None), telemetry: Optional[str] = Form(None)):
        tel = json.loads(telemetry) if telemetry else {}
        return server.handle(await image.read(), instruction, tel)

    @app.post("/steer")
    def steer(body: dict):
        delta = float(body.get("delta", 0.0))
        if not hasattr(server.policy, "set_delta") or getattr(server.policy, "hook", None) is None:
            return {"ok": False, "reason": "no steering hook loaded (start with --probes and --steer)"}
        server.policy.set_delta(delta)
        return {"ok": True, "steer_delta": delta}

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="key from configs/models.yaml")
    ap.add_argument("--dry-run", action="store_true", help="no model; returns hold, for latency tests")
    ap.add_argument("--probes", type=Path, default=None, help="folder of probe .npz files")
    ap.add_argument("--steer", default=None, help="probe target to steer, e.g. altitude")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--session", default=None, help="flights/<session>; default is a timestamp")
    ap.add_argument("--config", type=Path, default=common.CONFIGS / "flight.yaml")
    args = ap.parse_args()

    cfg = common.load_yaml(args.config)
    session = args.session or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.dry_run or not args.model:
        policy = DryRunPolicy()
    else:
        policy = QwenPolicy(args.model, cfg, args.probes, args.steer)
    server = Server(policy, cfg, session)
    print(f"session {session}  policy {policy.name}  log {server.log_path}")

    import uvicorn
    uvicorn.run(build_app(server), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
