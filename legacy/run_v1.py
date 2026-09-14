#!/usr/bin/env python3
"""Run the benchmark suite against an OpenAI-compatible endpoint.

  ./run.py --model qwen3.5-9b
  ./run.py --model qwen3.5-9b --base http://127.0.0.1:30000/v1 --tag baseline
  ./run.py --model X --only kepler_orbit,sec_logs
  ./run.py --compare                      # table across every run so far

Results land in results/<tag>/ : one .json per task, plus any artifacts
(the Kepler HTML is written out so it can actually be opened and looked at).
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suite import TASKS, build_needle_prompt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

G, Y, R, DIM, RST = "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[0m"


def post(base, payload, timeout):
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer local"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run_task(base, model, task, timeout, thinking):
    prompt = task["prompt"]
    if task.get("needle"):
        prompt = build_needle_prompt()

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": task.get("max_tokens", 1200),
        "temperature": 0.2,          # low: we are measuring capability, not variety
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }
    if task.get("tools"):
        payload["tools"] = task["tools"]

    t0 = time.time()
    try:
        d = post(base, payload, timeout)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:400].decode(errors='replace')}",
                "elapsed": time.time() - t0}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "elapsed": time.time() - t0}
    el = time.time() - t0

    ch = d["choices"][0]
    msg = ch.get("message", {}) or {}
    usage = d.get("usage", {}) or {}
    return {
        "content": msg.get("content") or "",
        "reasoning_content": msg.get("reasoning_content") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "finish_reason": ch.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "elapsed": el,
        "tok_per_s": (usage.get("completion_tokens") or 0) / el if el > 0 else 0,
    }


def save_artifact(outdir, task, resp):
    """Write generated code out so it can be opened and judged by eye."""
    if task.get("artifact") != "html":
        return None
    body = resp.get("content") or ""
    m = re.search(r"<!DOCTYPE html.*?</html>", body, re.S | re.I)
    html = m.group(0) if m else None
    if not html:
        m = re.search(r"```(?:html)?\s*(.*?)```", body, re.S)
        if m and "<canvas" in m.group(1).lower():
            html = m.group(1)
    if not html:
        return None
    p = os.path.join(outdir, f"{task['id']}.html")
    with open(p, "w") as f:
        f.write(html)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="served model id")
    ap.add_argument("--base", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--tag", help="results subdirectory (default: model id)")
    ap.add_argument("--only", help="comma-separated task ids")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--thinking", action="store_true",
                    help="enable_thinking=true (costs context; off by default)")
    ap.add_argument("--compare", action="store_true", help="table across all runs")
    a = ap.parse_args()

    if a.compare:
        return compare()
    if not a.model:
        ap.error("--model is required (or use --compare)")

    tag = a.tag or a.model.replace("/", "_")
    outdir = os.path.join(RESULTS, tag)
    os.makedirs(outdir, exist_ok=True)

    tasks = TASKS
    if a.only:
        want = {x.strip() for x in a.only.split(",")}
        tasks = [t for t in TASKS if t["id"] in want]
        if not tasks:
            sys.exit(f"no tasks matched {sorted(want)}")

    print(f"{DIM}endpoint {a.base}  model {a.model}  thinking={a.thinking}{RST}")
    print(f"{DIM}{len(tasks)} tasks -> {outdir}{RST}\n")

    rows, tot_w, tot_s = [], 0.0, 0.0
    for t in tasks:
        print(f"  {t['id']:22} {DIM}{t['category']:9}{RST} ", end="", flush=True)
        resp = run_task(a.base, a.model, t, a.timeout, a.thinking)

        if "error" in resp:
            score, note = 0.0, resp["error"][:90]
        else:
            try:
                score, note = t["scorer"](resp, t)
            except Exception as e:
                score, note = None, f"scorer crashed: {e}"

        art = save_artifact(outdir, t, resp) if "error" not in resp else None
        w = t.get("weight", 1.0)
        if score is not None:
            tot_w += w
            tot_s += score * w

        col = R if (score or 0) < 0.5 else (Y if (score or 0) < 0.85 else G)
        disp = "REVIEW" if score is None else f"{score*100:5.0f}%"
        extra = ""
        if "error" not in resp:
            extra = f"{DIM}{resp['elapsed']:5.1f}s {resp.get('completion_tokens') or 0:>5}tok{RST}"
        print(f"{col}{disp}{RST}  {extra}  {DIM}{note[:70]}{RST}")
        if art:
            print(f"{' '*24}{DIM}artifact: {art}{RST}")

        rows.append({"id": t["id"], "category": t["category"], "weight": w,
                     "score": score, "note": note, "artifact": art,
                     "response": resp})

    by_cat = {}
    for r in rows:
        if r["score"] is None:
            continue
        c = by_cat.setdefault(r["category"], [0.0, 0.0])
        c[0] += r["score"] * r["weight"]
        c[1] += r["weight"]

    print(f"\n  {'-'*66}")
    for c, (s, w) in sorted(by_cat.items()):
        print(f"  {c:12} {s/w*100:5.1f}%   {DIM}(weight {w:.1f}){RST}")
    overall = tot_s / tot_w * 100 if tot_w else 0
    print(f"  {'OVERALL':12} {overall:5.1f}%\n")

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump({"model": a.model, "base": a.base, "thinking": a.thinking,
                   "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "overall": overall,
                   "by_category": {c: s / w * 100 for c, (s, w) in by_cat.items()},
                   "tasks": [{k: v for k, v in r.items() if k != "response"}
                             for r in rows]}, f, indent=2)
    with open(os.path.join(outdir, "raw.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"{DIM}  wrote {outdir}/summary.json{RST}")
    if any(r["artifact"] for r in rows):
        print(f"{DIM}  open the .html artifacts to judge the Kepler render by eye{RST}")


def compare():
    if not os.path.isdir(RESULTS):
        sys.exit("no results yet")
    runs = []
    for tag in sorted(os.listdir(RESULTS)):
        p = os.path.join(RESULTS, tag, "summary.json")
        if os.path.exists(p):
            runs.append((tag, json.load(open(p))))
    if not runs:
        sys.exit("no summary.json files yet")

    cats = sorted({c for _, s in runs for c in s.get("by_category", {})})
    print(f"\n  {'run':22} " + " ".join(f"{c[:9]:>10}" for c in cats) + f" {'OVERALL':>9}")
    print(f"  {'-'*(22 + 11*len(cats) + 10)}")
    for tag, s in sorted(runs, key=lambda r: -r[1].get("overall", 0)):
        cells = []
        for c in cats:
            v = s.get("by_category", {}).get(c)
            cells.append(f"{v:9.1f}%" if v is not None else f"{'-':>10}")
        print(f"  {tag[:22]:22} " + " ".join(cells) + f" {s.get('overall', 0):8.1f}%")
    print()


if __name__ == "__main__":
    main()
