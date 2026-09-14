#!/usr/bin/env python3
"""Benchmark local models THROUGH omp, the way Tate uses them.

Every task is a real `omp -p` session: omp's own system prompt, tool set,
agent loop and context handling, driving the model served by SGLang. omp and
every command the model runs execute inside the bench-omp container
(--network none, read-only root, no capabilities, non-root); its only route out
is a unix socket bridged to the SGLang port.

  ./omp_run.py granite42-8b                       # base config, 3 samples
  ./omp_run.py gemma4-12b --variant ctx64k
  ./omp_run.py mellum2-12b --variant think --only kepler_orbit --samples 1
  ./omp_run.py --table                            # live score table (markdown)
  ./omp_run.py --list-variants

Results: results_omp/<model>/<variant>/
           samples/<task>__s<k>.json      score, note, episode stats, final reply
           events/<task>__s<k>.jsonl.gz   full omp event stream
           work/<task>__s<k>/             final workspace (for inspection)
           server.json                    startup metrics, speed, peak VRAM
"""
import argparse
import re
import glob
import gzip
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from configs import MODELS  # noqa: E402
from model_bridge import ModelBridge  # noqa: E402
from omp_tasks import TASKS  # noqa: E402
from server import Server, health, stream_timing, wait_until_safe  # noqa: E402

RESULTS = os.path.join(HERE, "results_omp")
IMAGE = "bench-omp:1"
PRINT = threading.Lock()
CATS = ["agentic", "coding", "tools", "reasoning", "security"]
G, Y, R, DIM, RST = "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[0m"

# ---------------------------------------------------------------------------
# omp configuration variants. Each changes ONE thing from "base".
#   context:     target context window (capped by the server's real KV pool)
#   max_tokens:  omp's per-reply reservation (carved out of the window)
#   thinking:    chat_template enable_thinking
#   sampling:    overrides on the vendor sampling
#   tools:       omp --tools list (None = omp default, all 11)
#   server:      extra SGLang flags
# ---------------------------------------------------------------------------
BASE = dict(context=32768, max_tokens=4096, thinking=False, sampling={}, tools=None, server=[], compaction=None)
OMP_PROMPT_FLOOR = 19000   # omp system prompt + 11 tool schemas, measured ~18.7k tokens


def tuned_compaction(window, max_tokens):
    """omp's defaults (reserve 16384, keep 20000 recent) put the compaction
    threshold BELOW omp's own ~18.7k prompt floor in a 28-32k window: the pilot
    compacted 40 times in one episode until the model lost its files. Tuned:
    reserve just the reply plus slack, keep a quarter of the real working space."""
    working = max(2000, window - OMP_PROMPT_FLOOR - max_tokens)
    return {"reserveTokens": max_tokens + 1024, "keepRecentTokens": max(1500, working // 4)}
LEAN_TOOLS = "read,write,edit,bash,grep,glob"
OMP_VARIANTS = {
    "base": {},
    "think": {"thinking": True},
    "max2048": {"max_tokens": 2048},          # Tate's current omp setting
    "max8192": {"max_tokens": 8192},
    "ctx64k": {"context": 65536, "server": ["--context-length", "65536", "--kv-cache-dtype", "fp8_e4m3"]},
    "temp0.2": {"sampling": {"temperature": 0.2}},
    "lean-tools": {"tools": LEAN_TOOLS},
    "think-ctx64k": {"thinking": True, "max_tokens": 8192, "context": 65536,
                     "server": ["--context-length", "65536", "--kv-cache-dtype", "fp8_e4m3"]},
}
# Per-model server adjustments needed for a variant to be valid.
MODEL_SERVER_FIXUPS = {
    # Mellum's KV is tiny (140k tokens at bf16): no fp8 needed for 64k.
    ("mellum2-12b", "ctx64k"): ["--context-length", "65536"],
    ("mellum2-12b-awq", "ctx64k"): ["--context-length", "65536"],
    ("mellum2-12b", "think-ctx64k"): ["--context-length", "65536"],
    ("mellum2-12b-awq", "think-ctx64k"): ["--context-length", "65536"],
    # Qwen3.5 is hybrid mamba; its server profile caps context separately.
    ("qwen3.5-9b", "ctx64k"): ["--context-length", "65536", "--kv-cache-dtype", "fp8_e4m3"],
}

SWA_MODELS = {"gemma4-12b", "gemma4-12b-nvfp4", "gemma4-12b-awq", "mellum2-12b", "mellum2-12b-awq"}

TASK_MAX_TIME = {"kepler_orbit": 1500, "agent_feature": 1500, "agent_bugfix": 1200, "reason_longctx": 1200}
DEFAULT_MAX_TIME = 900


def log(msg):
    with PRINT:
        print(msg, flush=True)


def ladder_variant(variant):
    """ctx<N>k[-kvfp8] -> a context-ladder rung, generated on demand.
    Memory fraction comes from the model's ctx_probe result when present."""
    m = re.fullmatch(r"ctx(\d+)k(-kvfp8)?", variant)
    if not m:
        return None
    ctx = int(m.group(1)) * 1024
    server = ["--context-length", str(ctx)]
    if m.group(2):
        server += ["--kv-cache-dtype", "fp8_e4m3"]
    return {"context": ctx, "server": server, "ladder": True}


KNOB_KEYS = ("thinking", "max_tokens", "sampling", "tools")


def resolve(model, variant):
    v = dict(BASE)
    head, *knobs = variant.split("+")
    for k in knobs:
        if k not in OMP_VARIANTS or any(x not in KNOB_KEYS for x in OMP_VARIANTS[k]):
            raise SystemExit(f"'+{k}' is not a knob variant (knobs: "
                             f"{[n for n, d in OMP_VARIANTS.items() if d and all(x in KNOB_KEYS for x in d)]})")
    tuned = head.endswith("-ct")
    base_variant = head[:-3] if tuned else head
    lv = ladder_variant(base_variant)
    if lv is None and base_variant not in OMP_VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; see --list-variants or use ctx<N>k[-kvfp8]")
    v.update(lv or OMP_VARIANTS[base_variant])
    for k in knobs:
        for key, val in OMP_VARIANTS[k].items():
            v[key] = {**v[key], **val} if key == "sampling" else val
    if lv:
        probe = os.path.join(RESULTS, "_ctx", f"{model}.json")
        if os.path.exists(probe):
            pr = json.load(open(probe))
            kv = "fp8_e4m3" if "--kv-cache-dtype" in v["server"] else "auto"
            t = next((t for t in pr.get("trials", []) if t.get("kv") == kv and t.get("mem")), None)
            # the probe ran single-request; the ladder may run 2 concurrent episodes
            if t:
                # The probe proved 0.91 with ONE request; under omp, SGLang lazily loads more
                # Triton kernels (tool-call grammar bitmask) after serving starts, and at 0.91
                # only 0.13 GiB was left -> CUDA error -> scheduler died (Mellum, 2026-09-14).
                # ...but only cap where the probe showed thin headroom: Gemma kept 1.1-1.3 GiB
                # free even at 0.91, so capping it only shrank its pool from ~118k to 100k.
                headroom = (t.get("metrics") or {}).get("headroom_gb") or 0
                mem = t["mem"] if headroom >= 0.9 else min(t["mem"], 0.89)
                v["server"] = v["server"] + ["--mem-fraction-static", str(mem)]
                if model in SWA_MODELS:
                    # Size the sliding-window pool for the concurrency the rung will really
                    # use: window 1024 + one 2048 prefill chunk per running request, plus
                    # margin. Sizing Gemma's for 2 requests at its 118k rung (where episodes
                    # run one at a time anyway) spent so much memory on sliding-window KV
                    # -- ~20x a full-attention token -- that the full pool shrank to 75.7k.
                    full = (t.get("metrics") or {}).get("full_tokens") or 100000
                    per_request = 4608
                    requests = 2 if full >= 2.4 * v["context"] else 1
                    ratio = min(0.5, max(0.02, requests * per_request / full))
                    v["server"] += ["--swa-full-tokens-ratio", str(round(ratio, 3))]
    m = MODELS[model]
    mode = "think" if v["thinking"] else "nothink"
    sampling = dict(m["sampling"][mode])
    sampling.update(v["sampling"])
    server = v["server"] if v.get("ladder") else MODEL_SERVER_FIXUPS.get((model, base_variant), v["server"])
    return {"model": model, "variant": variant, "repo": m["repo"], "quant": m["quant"],
            "context_target": v["context"], "max_tokens": v["max_tokens"], "thinking": v["thinking"],
            "sampling": sampling, "tools": v["tools"], "server_extra": list(server),
            "compaction": "tuned" if tuned else "default"}


def write_omp_config(cfg, context_window, cfg_dir):
    os.makedirs(cfg_dir, exist_ok=True)
    s = cfg["sampling"]
    extra = "\n".join(f"            {k}: {json.dumps(v)}" for k, v in s.items())
    open(os.path.join(cfg_dir, "models.yml"), "w").write(f"""providers:
  local:
    baseUrl: http://127.0.0.1:30000/v1
    api: openai-completions
    apiKey: local
    models:
      - id: {cfg['model']}
        name: {cfg['model']} (bench {cfg['variant']})
        reasoning: true
        input: [text]
        contextWindow: {context_window}
        maxTokens: {cfg['max_tokens']}
        cost: {{input: 0, output: 0, cacheRead: 0, cacheWrite: 0}}
        compat:
          supportsDeveloperRole: false
          maxTokensField: max_tokens
          supportsReasoningEffort: false
          extraBody:
{extra}
            chat_template_kwargs: {{enable_thinking: {str(cfg['thinking']).lower()}}}
""")
    conf = f"modelRoles:\n  default: local/{cfg['model']}\nsetupVersion: 2\n"
    if cfg.get("compaction") == "tuned":
        c = tuned_compaction(context_window, cfg["max_tokens"])
        cfg["compaction_settings"] = c
        conf += "compaction:\n" + "".join(f"  {k}: {v}\n" for k, v in c.items())
    open(os.path.join(cfg_dir, "config.yml"), "w").write(conf)
    for f in os.listdir(cfg_dir):
        os.chmod(os.path.join(cfg_dir, f), 0o644)
    os.chmod(cfg_dir, 0o755)


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------
def run_episode(task, sample, cfg, vdir, cfg_dir, sock_dir):
    key = f"{task['id']}__s{sample}"
    ws = os.path.join(vdir, "work", key)
    if os.path.exists(ws):
        _rm(ws)
    os.makedirs(ws)
    os.chmod(ws, 0o777)
    for rel, body in task["files"].items():
        p = os.path.join(ws, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(body)
    for root, dirs, files in os.walk(ws):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)
        for f in files:
            os.chmod(os.path.join(root, f), 0o666)

    max_time = TASK_MAX_TIME.get(task["id"], DEFAULT_MAX_TIME)
    if cfg["context_target"] > 40000:
        # Some slack for slower long-context prefill, but capped at 1.5x: omp reuses the
        # cached prefix between turns, so a big multiplier mostly let a flailing model run
        # on (Mellum at 92k: 162 turns, 153 failed tool calls, 72 minutes, one episode).
        max_time = int(max_time * min(1.5, cfg["context_target"] / 32768))
    # Docker names allow only [a-zA-Z0-9_.-]; "+" in knob variants ("ctx92k-ct+think") made
    # every docker run fail instantly and the empty workspaces were graded as zeros.
    cname = re.sub(r"[^a-zA-Z0-9_.-]", "_", f"omp_{cfg['model']}_{cfg['variant']}_{key}")
    cmd = ["docker", "run", "--rm", "--name", cname,
           "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
           "--pids-limit", "512", "--memory", "3g", "--memory-swap", "3g", "--cpus", "4",
           "--tmpfs", "/tmp:size=512m,exec", "--tmpfs", "/home/runner:size=1024m,exec,uid=10001,gid=10001",
           "-e", "HOME=/home/runner", "-v", f"{sock_dir}:/sock", "-v", f"{cfg_dir}:/cfg:ro",
           "-v", f"{ws}:/work", IMAGE,
           "omp", "-p", "--mode", "json", "--auto-approve", "--no-session", "--no-title",
           "--max-time", str(max_time)]
    if cfg["tools"]:
        cmd += ["--tools", cfg["tools"]]
    cmd.append(task["prompt"])
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=max_time + 120)
        stdout, stderr, code = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr, code = "harness timeout", 124
    wall = time.time() - t0
    if code == 125 or (code != 0 and not stdout.strip() and wall < 30):
        # docker itself refused to run (bad name, missing image, daemon error):
        # a harness failure -- never grade an episode that never started.
        raise RuntimeError(f"container failed to start (exit {code}): {(stderr or '')[-300:]}")
    ep = parse_events(stdout)
    ep.update(exit_code=code, wall_s=round(wall, 1), stderr_tail=(stderr or "")[-1500:])
    os.makedirs(os.path.join(vdir, "events"), exist_ok=True)
    with gzip.open(os.path.join(vdir, "events", key + ".jsonl.gz"), "wt") as f:
        f.write(stdout)
    return ws, ep


def server_healthy(base, timeout=10):
    import urllib.request
    try:
        with urllib.request.urlopen(base.rsplit("/v1", 1)[0] + "/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def parse_events(stdout):
    ep = {"turns": 0, "tool_calls": 0, "tool_errors": 0, "tool_names": {}, "output_tokens": 0,
          "max_input_tokens": 0, "stop_reasons": {}, "final": "", "end": None, "errors": []}
    last_assistant_text = ""
    for line in stdout.splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = d.get("type")
        if t == "turn_end":
            msg = d.get("message") or {}
            ep["turns"] += 1
            u = msg.get("usage") or {}
            ep["output_tokens"] += u.get("output") or 0
            ep["max_input_tokens"] = max(ep["max_input_tokens"], u.get("input") or 0)
            sr = msg.get("stopReason")
            ep["stop_reasons"][sr] = ep["stop_reasons"].get(sr, 0) + 1
            if msg.get("errorMessage"):
                ep["errors"].append(str(msg["errorMessage"])[:300])
            texts = [c.get("text", "") for c in msg.get("content") or [] if c.get("type") == "text"]
            if "".join(texts).strip():
                last_assistant_text = "".join(texts)
        elif t == "tool_execution_end":
            ep["tool_calls"] += 1
            name = d.get("toolName")
            ep["tool_names"][name] = ep["tool_names"].get(name, 0) + 1
            if d.get("isError"):
                ep["tool_errors"] += 1
        elif t == "auto_compaction_end":
            ep["compactions"] = ep.get("compactions", 0) + 1
        elif t == "agent_end":
            ep["end"] = "done"
        elif t in ("error", "agent_error"):
            ep["errors"].append(json.dumps(d)[:300])
    ep["final"] = last_assistant_text
    ep["ctx_error"] = any(("exceeds" in x or "maximum context" in x or "Input length" in x) for x in ep["errors"])
    if ep["end"] is None:
        ep["end"] = "incomplete"
    elif ep["stop_reasons"].get("length"):
        ep["end"] = "done (hit max_tokens)"
    return ep


def _rm(path):
    subprocess.run(["docker", "run", "--rm", "--network", "none", "--user", "0:0", "-v",
                    f"{os.path.dirname(path)}:/p", "--entrypoint", "sh", IMAGE, "-c",
                    f"rm -rf /p/{os.path.basename(path)}"], capture_output=True)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--only")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--tasks", action="store_true", help="with --table: per-task detail")
    ap.add_argument("--list-variants", action="store_true")
    ap.add_argument("--regrade", action="store_true", help="re-grade saved workspaces, no model calls")
    a = ap.parse_args()
    if a.table:
        return table(a.tasks)
    if a.list_variants:
        for k, v in OMP_VARIANTS.items():
            print(f"{k:14} {v}")
        return
    if not a.model:
        ap.error("model required")
    tasks = TASKS if not a.only else [t for t in TASKS if t["id"] in set(a.only.split(","))]
    cfg = resolve(a.model, a.variant)
    vdir = os.path.join(RESULTS, a.model, a.variant)
    os.makedirs(os.path.join(vdir, "samples"), exist_ok=True)
    if a.regrade:
        return regrade(vdir, cfg, tasks)

    jobs = [(t, s) for t in tasks for s in range(a.samples)
            if not os.path.exists(os.path.join(vdir, "samples", f"{t['id']}__s{s}.json"))]
    if not jobs:
        log(f"{a.model} [{a.variant}] nothing to do")
        return
    srv = Server(a.model, extra=cfg["server_extra"], log=os.path.join(vdir, "server.log"))
    bridge = ModelBridge(os.path.join("/tmp/claude-1000", f"ompsock_{a.model}_{a.variant}"))
    info = {"config": cfg}
    try:
        log(f"{DIM}starting {a.model} [{a.variant}] {' '.join(cfg['server_extra'])}{RST}")
        info["metrics"] = srv.start()
        kv = info["metrics"].get("kv_tokens") or cfg["context_target"]
        ctx = min(cfg["context_target"], info["metrics"].get("context_len") or cfg["context_target"], kv)
        info["context_window"] = ctx
        info["speed"] = stream_timing(srv.base, a.model, "Write a 300-word explanation of TCP congestion control.",
                                      max_tokens=512, extra={"chat_template_kwargs": {"enable_thinking": False}})
        log(f"{DIM}  {info['metrics']}  omp contextWindow={ctx}  speed={info['speed']}{RST}")
        cfg_dir = os.path.join(vdir, "omp-config")
        write_omp_config(cfg, ctx, cfg_dir)
        bridge.start()
        log(f"\n{a.model} [{a.variant}]  {len(jobs)} episodes  (thinking={cfg['thinking']} "
            f"max_tokens={cfg['max_tokens']} ctx={ctx} tools={cfg['tools'] or 'all'} sampling={cfg['sampling']})")

        server_dead = threading.Event()

        def work(t, s):
            wait_until_safe(log)
            if server_dead.is_set() or srv.proc.poll() is not None or not server_healthy(srv.base):
                # The model server is gone (2026-09-14: a CUDA error killed Mellum's
                # scheduler mid-rung and ~20 episodes were graded against a dead server).
                server_dead.set()
                log(f"  {t['id']:18} s{s} {R}SKIPPED: model server is not healthy{RST}")
                return
            try:
                ws, ep = run_episode(t, s, cfg, vdir, cfg_dir, bridge.sock_dir)
                conn = [x for x in ep.get("errors", []) if re.search(
                    r"socket connection was closed|ECONNREFUSED|ECONNRESET|connection refused|fetch failed", x, re.I)]
                if conn and (srv.proc.poll() is not None or not server_healthy(srv.base)):
                    server_dead.set()
                    raise RuntimeError(f"model server died during the episode: {conn[0][:120]}")
                score, note, det = t["grade"](ws, ep["final"], ep)
            except Exception as e:
                # Not a model result: log it and leave no sample, so a rerun retries the
                # episode and the campaign sees the count short (-> runner failure).
                log(f"  {t['id']:18} s{s} {R}HARNESS ERROR{RST} {type(e).__name__}: {str(e)[:200]}")
                return
            ep["health_after"] = health()
            rec = {"task": t["id"], "category": t["category"], "weight": t["weight"], "sample": s,
                   "score": score, "note": note, "config": cfg, "context_window": ctx,
                   "episode": {k: v for k, v in ep.items() if k != "final"}, "final": ep.get("final", "")[-6000:],
                   "details": det}
            json.dump(rec, open(os.path.join(vdir, "samples", f"{t['id']}__s{s}.json"), "w"), indent=1, default=str)
            col = R if score < 0.5 else (Y if score < 0.85 else G)
            log(f"  {t['id']:18} s{s} {col}{score*100:5.0f}%{RST} {DIM}{ep.get('wall_s', 0):6.0f}s "
                f"{ep.get('turns', 0):3}t {ep.get('tool_calls', 0):3}c {ep.get('tool_errors', 0)}err "
                f"in<={ep.get('max_input_tokens', 0)} cmp={ep.get('compactions', 0)} {ep.get('end')}"
                f"{' CTX-ERR' if any('exceeds' in x or 'maximum context' in x for x in ep.get('errors', [])) else ''}{RST}  {note[:80]}")

        jobs.sort(key=lambda j: (j[0]["category"] != "agentic", j[0]["id"], j[1]))
        conc = a.concurrency
        if kv < 2 * ctx and conc > 1:
            conc = 1
            log(f"{DIM}  KV pool {kv} < 2 x context {ctx}: running episodes one at a time{RST}")
        with ThreadPoolExecutor(max_workers=conc) as ex:
            for f in as_completed([ex.submit(work, t, s) for t, s in jobs]):
                f.result()
    finally:
        bridge.stop()
        srv.stop()
        info["peak_vram_mib"] = srv.gpu.peak_mib
        info["peak_temp_c"] = srv.gpu.peak_temp
        json.dump(info, open(os.path.join(vdir, "server.json"), "w"), indent=2, default=str)
    table(False, only=(a.model, a.variant))


def regrade(vdir, cfg, tasks):
    for t in tasks:
        for p in sorted(glob.glob(os.path.join(vdir, "samples", f"{t['id']}__s*.json"))):
            rec = json.load(open(p))
            ws = os.path.join(vdir, "work", os.path.basename(p)[:-5])
            ep = rec["episode"]
            ep["final"] = rec.get("final", "")
            rec["score"], rec["note"], rec["details"] = t["grade"](ws, ep["final"], ep)
            json.dump(rec, open(p, "w"), indent=1, default=str)
            log(f"  regraded {os.path.basename(p)}: {rec['score']*100:.0f}% {rec['note'][:80]}")


def table(per_task=False, only=None):
    by_id = {t["id"]: t for t in TASKS}
    rows = []
    for vdir in sorted(glob.glob(os.path.join(RESULTS, "*", "*"))):
        model, variant = vdir.split(os.sep)[-2:]
        if model.startswith("_"):          # archived pilots, probes
            continue
        if only and (model, variant) != only:
            continue
        recs = [json.load(open(p)) for p in glob.glob(os.path.join(vdir, "samples", "*.json"))]
        if not recs:
            continue
        per = {}
        for r in recs:
            per.setdefault(r["task"], []).append(r)
        cats = {}
        for tid, rs in per.items():
            t = by_id.get(tid)
            if not t:
                continue
            c = cats.setdefault(t["category"], [0.0, 0.0])
            c[0] += sum(x["score"] for x in rs) / len(rs) * t["weight"]
            c[1] += t["weight"]
        by_cat = {c: 100 * s / w for c, (s, w) in cats.items()}
        srv = os.path.join(vdir, "server.json")
        sj = json.load(open(srv)) if os.path.exists(srv) else {}
        eps = [r.get("episode") or {} for r in recs]
        rows.append(dict(model=model, variant=variant, by_cat=by_cat, n=len(recs), per=per,
                         overall=sum(by_cat.values()) / len(by_cat) if by_cat else 0,
                         tok_s=(sj.get("speed") or {}).get("decode_tok_s"), ctx=sj.get("context_window"),
                         tool_err=sum(e.get("tool_errors", 0) for e in eps),
                         tool_calls=sum(e.get("tool_calls", 0) for e in eps),
                         incomplete=sum(1 for e in eps if e.get("end") not in ("done", "done (hit max_tokens)")),
                         wall=sum(e.get("wall_s", 0) for e in eps) / max(1, len(eps))))
    if not rows:
        print("no omp results yet")
        return
    total = len(TASKS) * 3
    print("| model · variant | done | overall* | " + " | ".join(CATS) + " | tok/s | ctx | tool err | unfinished | avg s/ep |")
    print("|---|---|---|" + "---|" * len(CATS) + "---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: -r["overall"]):
        cells = [f"{r['by_cat'][c]:.0f}" if c in r["by_cat"] else "–" for c in CATS]
        err = f"{r['tool_err']}/{r['tool_calls']}"
        print(f"| {r['model']} · {r['variant']} | {r['n']}/{total} | **{r['overall']:.1f}** | " + " | ".join(cells)
              + f" | {r['tok_s'] or '–'} | {r['ctx'] or '–'} | {err} | {r['incomplete']} | {r['wall']:.0f} |")
    print("\n*overall = mean of categories with data; rows below full sample count are provisional.")
    if per_task:
        for r in rows:
            print(f"\n{r['model']} · {r['variant']}")
            for tid in sorted(r["per"], key=lambda i: (by_id[i]["category"], i)):
                rs = sorted(r["per"][tid], key=lambda x: x["sample"])
                sc = [x["score"] for x in rs]
                print(f"  {tid:18} {sum(sc)/len(sc)*100:5.0f}%  {' '.join(f'{v*100:.0f}' for v in sc):12} {rs[-1]['note'][:90]}")


if __name__ == "__main__":
    main()
