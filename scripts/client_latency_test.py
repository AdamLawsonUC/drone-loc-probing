#!/usr/bin/env python
"""Round-trip latency test for the flight server. Run from the laptop in the field over the same path the
phone will use (hotspot + VPN), first against --dry-run, then against the real model.

Usage:
  python scripts/client_latency_test.py --server http://<a100-host>:8765 --image tests/sample.jpg --n 30
  python scripts/client_latency_test.py --server http://<a100-host>:8765 --image tests/sample.jpg --n 30 --rel-alt 30

Prints mean, median and p95 round trip, server-side time, and the action distribution. Writes a CSV next to
the image unless --out is given.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import requests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--rel-alt", type=float, default=None, help="telemetry rel_alt to send (tests the envelope)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    h = requests.get(f"{args.server}/health", timeout=10).json()
    print("health:", h)
    img = args.image.read_bytes()
    rows, rtts, srv = [], [], []
    for i in range(args.n):
        tel = {"ts": time.time()}
        if args.rel_alt is not None:
            tel["rel_alt"] = args.rel_alt
        t0 = time.time()
        r = requests.post(f"{args.server}/act", files={"image": (args.image.name, img, "image/jpeg")},
                          data={"telemetry": json.dumps(tel)}, timeout=60)
        rtt = (time.time() - t0) * 1000
        j = r.json()
        rtts.append(rtt)
        srv.append(j.get("server_ms", 0.0))
        rows.append({"i": i, "rtt_ms": round(rtt, 1), **j})
        print(f"{i:3d} rtt={rtt:7.1f} ms  server={j.get('server_ms'):7.1f} ms  action={j.get('action')}  note={j.get('note', '')}")

    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
    print(f"\nround trip: mean {statistics.mean(rtts):.1f}  median {statistics.median(rtts):.1f}  p95 {q(rtts, 0.95):.1f} ms")
    print(f"server:     mean {statistics.mean(srv):.1f}  median {statistics.median(srv):.1f}  p95 {q(srv, 0.95):.1f} ms")
    print(f"network:    median {statistics.median(rtts) - statistics.median(srv):.1f} ms")
    acts = {}
    for r in rows:
        acts[r["action"]] = acts.get(r["action"], 0) + 1
    print("actions:", acts)

    out = args.out or args.image.with_suffix(".latency.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("written", out)


if __name__ == "__main__":
    main()
