#!/usr/bin/env python3
"""Score the hand-written Claude answers with the SAME scorers used for models.

Writes results/claude-opus-5/ so `./run.py --compare` picks it up alongside
the served-model runs.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suite import TASKS            # noqa: E402
from claude_answers import ANSWERS  # noqa: E402
from run import save_artifact       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "claude-opus-5")
os.makedirs(OUT, exist_ok=True)

G, Y, R, DIM, RST = "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[0m"

rows, tot_w, tot_s = [], 0.0, 0.0
for t in TASKS:
    ans = ANSWERS.get(t["id"])
    if ans is None:
        print(f"  {t['id']:22} {DIM}no answer written{RST}")
        continue
    resp = {"content": ans.get("content", ""),
            "reasoning_content": ans.get("reasoning_content", ""),
            "tool_calls": ans.get("tool_calls", []),
            "finish_reason": "stop",
            "elapsed": 0.0, "completion_tokens": None}
    score, note = t["scorer"](resp, t)
    art = save_artifact(OUT, t, resp)
    w = t.get("weight", 1.0)
    if score is not None:
        tot_w += w
        tot_s += score * w
    col = R if score < 0.5 else (Y if score < 0.85 else G)
    print(f"  {t['id']:22} {DIM}{t['category']:9}{RST} {col}{score*100:5.0f}%{RST}"
          f"  {DIM}{note[:70]}{RST}")
    if art:
        print(f"{' '*24}{DIM}artifact: {art}{RST}")
    rows.append({"id": t["id"], "category": t["category"], "weight": w,
                 "score": score, "note": note, "artifact": art})

by_cat = {}
for r in rows:
    c = by_cat.setdefault(r["category"], [0.0, 0.0])
    c[0] += r["score"] * r["weight"]
    c[1] += r["weight"]

print(f"\n  {'-'*66}")
for c, (s, w) in sorted(by_cat.items()):
    print(f"  {c:12} {s/w*100:5.1f}%   {DIM}(weight {w:.1f}){RST}")
overall = tot_s / tot_w * 100 if tot_w else 0
print(f"  {'OVERALL':12} {overall:5.1f}%\n")

json.dump({"model": "claude-opus-5 (hand-written reference)", "base": "n/a",
           "thinking": None, "when": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
           "overall": overall,
           "by_category": {c: s / w * 100 for c, (s, w) in by_cat.items()},
           "tasks": rows},
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print(f"{DIM}  wrote {OUT}/summary.json{RST}")
