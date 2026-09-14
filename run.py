#!/usr/bin/env python3
"""Run the v2 suite for one or more configs.

  ./run.py gemma4-12b                          # default variant, both modes, 3 samples
  ./run.py granite42-8b --modes nothink --samples 1 --only tool_basic,code_lru
  ./run.py mellum2-12b --variant kvfp8
  ./run.py qwen3.5-9b --attach                 # use a server already running on :30000

Starts the model's server (serve-model.sh profile + variant flags), measures
speed/memory, runs every (task, sample) with 2 concurrent requests, and stops
the server safely. Resumable: finished samples are skipped on rerun.

Results: results/<model>/<variant>/<mode>/
           samples/<task>__s<k>.json   one file per sample (response + score + details)
           summary.json                per-task mean/min/max, per-category, overall
         results/<model>/<variant>/server.json   startup metrics, speed, peak VRAM
"""
import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import traceback
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agentic import AGENT_TASKS, run_episode  # noqa: E402
from configs import MODES, THINK_EXTRA_TOKENS, resolve  # noqa: E402
from kepler import extract_html  # noqa: E402
from server import Server, chat, stream_timing  # noqa: E402
from suite import TASKS  # noqa: E402

RESULTS = os.path.join(HERE, "results")
ALL_TASKS = TASKS + AGENT_TASKS
G, Y, R, DIM, RST = "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[0m"
PRINT = threading.Lock()


def log(msg):
    with PRINT:
        print(msg, flush=True)


def seed_for(task_id, sample):
    return int(hashlib.sha256(f"{task_id}:{sample}".encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------
CTX_ERR = re.compile(r"(maximum context|context length|exceeds|too long|max_tokens)", re.I)


def request(base, cfg, messages, max_tokens, seed, tools=None, timeout=1800):
    """One chat call with the config's sampling. If the prompt + budget would
    exceed the context window, shrink max_tokens and retry rather than fail."""
    extra = dict(cfg.get("request", {}))
    ctk = {"enable_thinking": cfg["thinking"], **extra.pop("chat_template_kwargs", {})}
    payload = {"model": cfg["model"], "messages": messages, "max_tokens": max_tokens,
               "seed": seed, "chat_template_kwargs": ctk, **cfg["sampling"], **extra}
    if tools:
        payload["tools"] = tools
    for attempt in range(4):
        try:
            return chat(base, payload, timeout=timeout)
        except urllib.error.HTTPError as e:
            body = e.read()[:600].decode(errors="replace")
            if e.code == 400 and CTX_ERR.search(body) and payload["max_tokens"] > 512:
                payload["max_tokens"] = max(256, payload["max_tokens"] // 2)
                continue
            raise RuntimeError(f"HTTP {e.code}: {body}") from None
    raise RuntimeError("could not fit request into the context window")


def run_single(base, cfg, task, sample):
    budget = task.get("max_tokens", 2000) + (THINK_EXTRA_TOKENS if cfg["thinking"] else 0)
    if cfg["thinking"] and task.get("think_max_tokens"):
        budget = task["think_max_tokens"]
    msgs = [{"role": "user", "content": task["prompt"]}]
    t0 = time.time()
    d = request(base, cfg, msgs, budget, seed_for(task["id"], sample), tools=task.get("tools"))
    el = time.time() - t0
    ch = d["choices"][0]
    m = ch.get("message") or {}
    u = d.get("usage") or {}
    resp = {"content": m.get("content") or "", "reasoning_content": m.get("reasoning_content") or "",
            "tool_calls": m.get("tool_calls") or [], "finish_reason": ch.get("finish_reason"),
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "elapsed": round(el, 2)}
    if task.get("followup_tools") and resp["tool_calls"]:
        # Multi-turn tool task: answer each call with a canned result and let the
        # model continue, recording the calls made on every turn.
        turns = [resp["tool_calls"]]
        for turn in range(3):
            msgs.append({"role": "assistant", "content": resp["content"] if turn == 0 else m.get("content") or "",
                         "tool_calls": [{"id": c.get("id") or f"c{turn}_{i}", "type": "function",
                                         "function": c["function"]} for i, c in enumerate(turns[-1])]})
            for i, c in enumerate(turns[-1]):
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except Exception:
                    args = {}
                msgs.append({"role": "tool", "tool_call_id": c.get("id") or f"c{turn}_{i}",
                             "content": task["followup_tools"](c["function"]["name"], args)})
            d = request(base, cfg, msgs, budget, seed_for(task["id"], sample) + turn + 1, tools=task.get("tools"))
            m = d["choices"][0].get("message") or {}
            if not m.get("tool_calls"):
                resp["final_content"] = m.get("content") or ""
                break
            turns.append(m["tool_calls"])
        resp["tool_turns"] = turns
        resp["elapsed"] = round(time.time() - t0, 2)
    score, note, details = task["scorer"](resp, task)
    if resp["finish_reason"] == "length":
        note = "TRUNCATED; " + note
    return resp, score, note, details


def run_agentic(base, cfg, task, sample):
    turn_tokens = task.get("turn_tokens", 4096) + (4096 if cfg["thinking"] else 0)
    counter = {"n": 0}

    def chat_fn(messages, tools, max_tokens):
        counter["n"] += 1
        return request(base, cfg, messages, max_tokens, seed_for(task["id"], sample) + counter["n"], tools=tools)

    score, note, details = run_episode(chat_fn, task, turn_tokens=turn_tokens,
                                       ctx_budget=cfg["context"] - turn_tokens - 512)
    resp = {"elapsed": details["stats"].get("elapsed"),
            "completion_tokens": details["stats"]["completion_tokens"],
            "finish_reason": details["stats"]["end"]}
    return resp, score, note, details


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--variant", default="default")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--only", help="comma-separated task ids")
    ap.add_argument("--skip", help="comma-separated task ids to skip")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--attach", action="store_true", help="use an already-running server on :30000")
    ap.add_argument("--rescore", action="store_true", help="re-grade stored single-turn responses, no model calls")
    a = ap.parse_args()

    tasks = ALL_TASKS
    if a.only:
        want = set(a.only.split(","))
        tasks = [t for t in tasks if t["id"] in want]
    if a.skip:
        tasks = [t for t in tasks if t["id"] not in set(a.skip.split(","))]
    modes = a.modes.split(",")

    if a.rescore:
        for mode in modes:
            rescore(resolve(a.model, a.variant, mode), tasks)
        return

    cfg0 = resolve(a.model, a.variant, modes[0])
    vdir = os.path.join(RESULTS, a.model, a.variant)
    os.makedirs(vdir, exist_ok=True)

    srv = None
    base = "http://127.0.0.1:30000/v1"
    server_info = {}
    try:
        if not a.attach:
            srv = Server(a.model, extra=cfg0["server_extra"], env=cfg0["env"],
                         log=os.path.join(vdir, "server.log"))
            log(f"{DIM}starting {a.model} [{a.variant}] {' '.join(cfg0['server_extra'])}{RST}")
            server_info["metrics"] = srv.start()
            base = srv.base
            log(f"{DIM}  {server_info['metrics']}{RST}")
            server_info["speed"] = stream_timing(
                base, a.model, "Write a 300-word explanation of how TCP congestion control works.",
                max_tokens=512, extra={"chat_template_kwargs": {"enable_thinking": False}})
            log(f"{DIM}  speed {server_info['speed']}{RST}")

        for mode in modes:
            cfg = resolve(a.model, a.variant, mode)
            run_mode(base, cfg, tasks, a.samples, a.concurrency)
    finally:
        if srv:
            srv.stop()
            server_info["peak_vram_mib"] = srv.gpu.peak_mib
            server_info["peak_temp_c"] = srv.gpu.peak_temp
            prev = {}
            p = os.path.join(vdir, "server.json")
            if os.path.exists(p):
                prev = json.load(open(p))
            prev.update({k: v for k, v in server_info.items() if v})
            prev["server_extra"] = cfg0["server_extra"]
            json.dump(prev, open(p, "w"), indent=2)


def run_mode(base, cfg, tasks, samples, concurrency):
    mdir = os.path.join(RESULTS, cfg["model"], cfg["variant"], cfg["mode"])
    sdir = os.path.join(mdir, "samples")
    os.makedirs(sdir, exist_ok=True)
    jobs = [(t, s) for t in tasks for s in range(samples)
            if not os.path.exists(os.path.join(sdir, f"{t['id']}__s{s}.json"))]
    log(f"\n{cfg['model']} [{cfg['variant']}] mode={cfg['mode']}  sampling={cfg['sampling']}  "
        f"{len(jobs)} jobs to run ({len(tasks) * samples - len(jobs)} cached)")

    def work(t, s):
        t0 = time.time()
        try:
            fn = run_agentic if t["category"] == "agentic" else run_single
            resp, score, note, details = fn(base, cfg, t, s)
        except Exception as e:
            resp, score, note, details = ({"error": f"{type(e).__name__}: {e}"}, 0.0,
                                          f"ERROR {type(e).__name__}: {str(e)[:120]}",
                                          {"traceback": traceback.format_exc()[-3000:]})
        rec = {"task": t["id"], "category": t["category"], "weight": t.get("weight", 1.0),
               "sample": s, "score": score, "note": note, "config": cfg,
               "response": resp, "details": details, "wall_s": round(time.time() - t0, 1)}
        if t.get("artifact") == "html" and resp.get("content"):
            html = extract_html(resp["content"])
            if html:
                p = os.path.join(mdir, "artifacts", f"{t['id']}__s{s}.html")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "w").write(html)
                rec["artifact"] = p
        tmp = os.path.join(sdir, f".{t['id']}__s{s}.json.tmp")
        json.dump(rec, open(tmp, "w"), indent=1, default=str)
        os.replace(tmp, os.path.join(sdir, f"{t['id']}__s{s}.json"))
        col = R if score < 0.5 else (Y if score < 0.85 else G)
        err = "error" in resp
        log(f"  {t['id']:18} s{s} {col}{score*100:5.0f}%{RST} {DIM}{rec['wall_s']:6.1f}s "
            f"{resp.get('completion_tokens') or 0:>6}tok{RST}  {(R if err else DIM)}{note[:95]}{RST}")

    # Agentic episodes first (longest), so the pool stays balanced.
    jobs.sort(key=lambda j: (j[0]["category"] != "agentic", j[0]["id"], j[1]))
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for f in as_completed([ex.submit(work, t, s) for t, s in jobs]):
            f.result()
    summarize(mdir, cfg, tasks)


def summarize(mdir, cfg, tasks):
    sdir = os.path.join(mdir, "samples")
    per_task = {}
    for fn in os.listdir(sdir):
        if not fn.endswith(".json"):
            continue
        rec = json.load(open(os.path.join(sdir, fn)))
        per_task.setdefault(rec["task"], []).append(rec)
    known = {t["id"]: t for t in ALL_TASKS}
    rows = []
    for tid, recs in sorted(per_task.items()):
        if tid not in known:
            continue
        sc = [r["score"] for r in recs]
        rows.append({
            "id": tid, "category": known[tid]["category"], "weight": known[tid].get("weight", 1.0),
            "n": len(sc), "mean": sum(sc) / len(sc), "min": min(sc), "max": max(sc),
            "errors": sum("error" in (r.get("response") or {}) for r in recs),
            "truncated": sum((r.get("note") or "").startswith("TRUNCATED")
                             or (r.get("response") or {}).get("finish_reason") == "length" for r in recs),
            "mean_tokens": sum((r.get("response") or {}).get("completion_tokens") or 0 for r in recs) / len(recs),
            "mean_wall_s": sum(r.get("wall_s") or 0 for r in recs) / len(recs),
            "notes": [r["note"] for r in sorted(recs, key=lambda r: r["sample"])],
        })
    cats = {}
    for r in rows:
        c = cats.setdefault(r["category"], [0.0, 0.0])
        c[0] += r["mean"] * r["weight"]
        c[1] += r["weight"]
    by_cat = {c: 100 * s / w for c, (s, w) in cats.items()}
    # Overall = mean of category scores, so a category's task count doesn't set its influence.
    overall = sum(by_cat.values()) / len(by_cat) if by_cat else 0.0
    summary = {"config": cfg, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
               "overall": overall, "by_category": by_cat, "tasks": rows}
    json.dump(summary, open(os.path.join(mdir, "summary.json"), "w"), indent=2)
    log(f"\n  {'-' * 60}")
    for c in sorted(by_cat):
        log(f"  {c:10} {by_cat[c]:5.1f}%")
    log(f"  {'OVERALL':10} {overall:5.1f}%   {DIM}({mdir}/summary.json){RST}")


def rescore(cfg, tasks):
    """Re-grade stored single-turn responses with the current scorers."""
    mdir = os.path.join(RESULTS, cfg["model"], cfg["variant"], cfg["mode"])
    sdir = os.path.join(mdir, "samples")
    if not os.path.isdir(sdir):
        return
    known = {t["id"]: t for t in tasks}
    n = 0
    for fn in sorted(os.listdir(sdir)):
        rec = json.load(open(os.path.join(sdir, fn)))
        t = known.get(rec["task"])
        if not t or t["category"] == "agentic" or "error" in rec["response"]:
            continue
        rec["score"], rec["note"], rec["details"] = t["scorer"](rec["response"], t)
        json.dump(rec, open(os.path.join(sdir, fn), "w"), indent=1, default=str)
        n += 1
    log(f"rescored {n} samples in {mdir}")
    summarize(mdir, cfg, ALL_TASKS)


if __name__ == "__main__":
    main()
