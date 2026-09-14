#!/usr/bin/env python3
"""Live score table from per-sample files, including runs still in progress.

  ./live.py            # markdown table (all configs)
  ./live.py --tasks    # add per-task means for configs with data
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CATS = ["agentic", "coding", "tools", "reasoning", "security"]
sys.path.insert(0, HERE)
from agentic import AGENT_TASKS  # noqa: E402
from suite import TASKS  # noqa: E402

ALL = {t["id"]: t for t in TASKS + AGENT_TASKS}
TOTAL_JOBS = len(ALL) * 3


def main():
    rows = []
    for mdir in sorted(glob.glob(os.path.join(HERE, "results", "*", "*", "*"))):
        model, variant, mode = mdir.split(os.sep)[-3:]
        if model.startswith("_") or not os.path.isdir(os.path.join(mdir, "samples")):
            continue
        per_task = {}
        for p in glob.glob(os.path.join(mdir, "samples", "*.json")):
            r = json.load(open(p))
            per_task.setdefault(r["task"], []).append(r)
        if not per_task:
            continue
        cats = {}
        errors = trunc = 0
        for tid, recs in per_task.items():
            t = ALL.get(tid)
            if not t:
                continue
            mean = sum(r["score"] for r in recs) / len(recs)
            c = cats.setdefault(t["category"], [0.0, 0.0])
            c[0] += mean * t.get("weight", 1.0)
            c[1] += t.get("weight", 1.0)
            errors += sum("error" in (r.get("response") or {}) for r in recs)
            trunc += sum((r.get("note") or "").startswith("TRUNCATED") for r in recs)
        by_cat = {c: 100 * s / w for c, (s, w) in cats.items()}
        n = sum(len(v) for v in per_task.values())
        srv = os.path.join(HERE, "results", model, variant, "server.json")
        speed = (json.load(open(srv)).get("speed") or {}).get("decode_tok_s") if os.path.exists(srv) else None
        rows.append(dict(model=model, variant=variant, mode=mode, by_cat=by_cat, n=n, errors=errors,
                         trunc=trunc, speed=speed, per_task=per_task,
                         overall=sum(by_cat.values()) / len(by_cat)))
    if not rows:
        print("no samples yet")
        return
    print("| config | done | overall* | " + " | ".join(CATS) + " | tok/s | err | trunc |")
    print("|---|---|---|" + "---|" * len(CATS) + "---|---|---|")
    for r in sorted(rows, key=lambda r: -r["overall"]):
        cells = [f"{r['by_cat'][c]:.0f}" if c in r["by_cat"] else "–" for c in CATS]
        name = f"{r['model']}" + ("" if r["variant"] == "default" else f" [{r['variant']}]") + f" · {r['mode']}"
        print(f"| {name} | {r['n']}/{TOTAL_JOBS} | **{r['overall']:.1f}** | " + " | ".join(cells) +
              f" | {r['speed'] or '–'} | {r['errors']} | {r['trunc']} |")
    print("\n*overall = mean of the categories with data so far; partial runs are not comparable yet.")
    if "--tasks" in sys.argv:
        for r in rows:
            print(f"\n{r['model']} [{r['variant']}] {r['mode']}")
            for tid in sorted(r["per_task"], key=lambda i: (ALL[i]["category"], i)):
                recs = r["per_task"][tid]
                sc = [x["score"] for x in sorted(recs, key=lambda x: x["sample"])]
                print(f"  {tid:18} {sum(sc)/len(sc)*100:5.0f}%  {' '.join(f'{v*100:.0f}' for v in sc):14} "
                      f"{recs[0]['note'][:80]}")


if __name__ == "__main__":
    main()
