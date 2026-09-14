#!/usr/bin/env python3
"""Compare every config run so far.

  ./compare.py                 # all configs
  ./compare.py --tasks         # add a per-task matrix
  ./compare.py --json out.json # machine-readable dump (used by the report)
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CATS = ["agentic", "coding", "tools", "reasoning", "security"]
G, Y, R, DIM, B, RST = "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[1m", "\033[0m"


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "results", "*", "*", "*", "summary.json"))):
        model, variant, mode = p.split(os.sep)[-4:-1]
        if model.startswith("_"):
            continue
        s = json.load(open(p))
        srv_p = os.path.join(HERE, "results", model, variant, "server.json")
        srv = json.load(open(srv_p)) if os.path.exists(srv_p) else {}
        n_tasks = len(s["tasks"])
        rows.append({
            "config": f"{model}/{variant}/{mode}", "model": model, "variant": variant, "mode": mode,
            "overall": s["overall"], "by_category": s["by_category"], "tasks": s["tasks"],
            "complete": all(t["n"] >= 3 for t in s["tasks"]) and n_tasks >= 25,
            "n_tasks": n_tasks,
            "errors": sum(t["errors"] for t in s["tasks"]),
            "truncated": sum(t["truncated"] for t in s["tasks"]),
            "decode_tok_s": (srv.get("speed") or {}).get("decode_tok_s"),
            "kv_tokens": (srv.get("metrics") or {}).get("kv_tokens"),
            "weights_gb": (srv.get("metrics") or {}).get("weights_gb"),
            "peak_vram_mib": srv.get("peak_vram_mib"),
            "sampling": s["config"]["sampling"],
        })
    return rows


def colour(v):
    return G if v >= 80 else (Y if v >= 55 else R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = sorted(load(), key=lambda r: -r["overall"])
    if not rows:
        raise SystemExit("no results yet")

    hdr = f"  {'config':34} {'OVERALL':>8} " + " ".join(f"{c[:9]:>9}" for c in CATS) + \
          f" {'tok/s':>6} {'KV tok':>7} {'VRAM':>6} {'err':>4} {'trunc':>5}"
    print(B + hdr + RST)
    for r in rows:
        cells = []
        for c in CATS:
            v = r["by_category"].get(c)
            cells.append(f"{colour(v)}{v:8.1f}%{RST}" if v is not None else f"{'-':>9}")
        flag = "" if r["complete"] else f" {DIM}(partial: {r['n_tasks']} tasks){RST}"
        print(f"  {r['config']:34} {colour(r['overall'])}{r['overall']:7.1f}%{RST} " + " ".join(cells) +
              f" {r['decode_tok_s'] or '-':>6} {r['kv_tokens'] or '-':>7} "
              f"{(str(round(r['peak_vram_mib']/1024, 1)) + 'G') if r['peak_vram_mib'] else '-':>6} "
              f"{r['errors']:>4} {r['truncated']:>5}{flag}")

    if a.tasks:
        ids = sorted({t["id"] for r in rows for t in r["tasks"]},
                     key=lambda i: (next(t["category"] for r in rows for t in r["tasks"] if t["id"] == i), i))
        short = [r["config"].replace("/default", "").replace("/nothink", "/nt").replace("/think", "/th")
                 for r in rows]
        print("\n" + B + f"  {'task':20}" + "".join(f"{s[:14]:>15}" for s in short) + RST)
        for i in ids:
            line = f"  {i:20}"
            for r in rows:
                t = next((t for t in r["tasks"] if t["id"] == i), None)
                if not t:
                    line += f"{'-':>15}"
                    continue
                m = t["mean"] * 100
                spread = (t["max"] - t["min"]) * 100
                line += f"{colour(m)}{m:6.0f}%{RST}{DIM} ±{spread/2:3.0f}{RST}    "
            print(line)

    if a.json:
        json.dump(rows, open(a.json, "w"), indent=2)
        print(f"\n{DIM}wrote {a.json}{RST}")


if __name__ == "__main__":
    main()
