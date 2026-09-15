#!/usr/bin/env python3
"""Coarse-to-fine omp campaign (Tate's chosen scope, 2026-09-13).

  Phase 1  coarse context ladder per checkpoint, 2 samples, tuned compaction:
           max -> midpoint -> 32k, then 24k, 20k ... until UNUSABLE
  Phase 2  refine: add a midpoint rung wherever adjacent rungs differ > 15 points
  Phase 3  best rung per checkpoint topped up to 3 samples
  Phase 4  knob sweeps for the top 3 checkpoints at their best rung, 2 samples:
           +think, +max2048, +max8192, +temp0.2, +lean-tools, and DEFAULT compaction

UNUSABLE = at least half the episodes end in a context error / never finish, or
the overall score is under 20. KV precision per rung: bf16 when the probe showed
its pool fits the rung (better quality), fp8 otherwise.

Resumable: every (model, variant) run is itself resumable, and decisions are
recomputed from results on disk. Run detached:
  setsid nohup .venv/bin/python -u omp_campaign.py > results_omp/_campaign.log 2>&1 &
"""
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from omp_tasks import TASKS  # noqa: E402

RES = os.path.join(HERE, "results_omp")
PY = os.path.join(HERE, ".venv", "bin", "python")
CHECKPOINTS = ["granite42-8b", "mellum2-12b", "gemma4-12b", "qwen3.5-9b",
               "gemma4-12b-nvfp4", "gemma4-12b-awq", "mellum2-12b-awq", "granite42-8b-fp8"]
OMP_MIN = 20 * 1024        # below this omp's own prompt + a reply cannot fit
# Tate, 2026-09-14: "if it is below a 48k context window then we can drop it" -- under 48k,
# omp spends too much of each episode compacting to be practical. No rungs below this.
CTX_FLOOR = 48 * 1024
# A model whose largest rung scores under this is not competitive; don't spend more rungs on it.
COMPETITIVE = 50
# Tate, 2026-09-14: skip the 3rd-sample top-up for these (best rung stays at 2 samples).
SKIP_TOPUP = {"qwen3.5-9b"}
KNOBS = ["think", "max2048", "max8192", "temp0.2", "lean-tools"]
STATE = os.path.join(RES, "_campaign_state.json")


def log(msg):
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def probe(model):
    p = os.path.join(RES, "_ctx", f"{model}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def wait_for_probe(model):
    while True:
        pr = probe(model)
        if pr and len(pr.get("trials", [])) >= 2 and ("max_context" in pr or all(
                t.get("proofs") or t.get("mem") is None for t in pr["trials"])):
            return pr
        if not probe_running():
            return pr
        time.sleep(60)


def probe_running():
    """True only if a python process is actually executing ctx_probe.py.
    `pgrep -f ctx_probe.py` matched any shell or monitor whose command line merely
    contained the string -- including the stall watchdog -- and the campaign
    waited on a probe that had already finished (2026-09-13)."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if argv and b"python" in os.path.basename(argv[0]) and any(os.path.basename(a) == b"ctx_probe.py" for a in argv[1:3]):
            return True
    return False


def pools(pr):
    out = {}
    for t in pr.get("trials", []):
        if t.get("max_context"):
            out[t["kv"]] = t["max_context"]
    return out


def rung_variant(pr, ctx):
    k = ctx // 1024
    bf16 = pools(pr).get("auto", 0)
    return f"ctx{k}k-ct" if bf16 >= k * 1024 else f"ctx{k}k-kvfp8-ct"


def summarize(model, variant):
    vdir = os.path.join(RES, model, variant)
    recs = [json.load(open(p)) for p in glob.glob(os.path.join(vdir, "samples", "*.json"))]
    if not recs:
        return None
    by_id = {t["id"]: t for t in TASKS}
    cats = {}
    per = {}
    for r in recs:
        per.setdefault(r["task"], []).append(r["score"])
    for tid, sc in per.items():
        t = by_id[tid]
        c = cats.setdefault(t["category"], [0.0, 0.0])
        c[0] += sum(sc) / len(sc) * t["weight"]
        c[1] += t["weight"]
    by_cat = {c: 100 * s / w for c, (s, w) in cats.items()}
    eps = [r.get("episode") or {} for r in recs]
    bad = sum(1 for e in eps if e.get("ctx_error") or e.get("end") not in ("done", "done (hit max_tokens)"))
    return {"overall": sum(by_cat.values()) / len(by_cat), "by_cat": by_cat, "n": len(recs),
            "bad_fraction": bad / len(recs)}


class RunnerFailure(Exception):
    pass


def expected_samples(samples):
    return len(TASKS) * samples


def run(model, variant, samples):
    """Run one config. A runner that crashes or leaves samples missing is a
    HARNESS failure, not a result: clean up, retry once, then halt the campaign.
    (2026-09-14: a V8 segfault orphaned a server and the campaign recorded every
    later run as an empty 'unusable' rung, then declared itself complete.)"""
    from server import stop_orphaned_bench_servers
    for attempt in (1, 2):
        log(f"RUN {model} {variant} samples={samples}" + (" (retry)" if attempt == 2 else ""))
        code = subprocess.run([PY, "-u", os.path.join(HERE, "omp_run.py"), model, "--variant", variant,
                               "--samples", str(samples)],
                              stdout=open(os.path.join(RES, f"_run_{model}.log"), "a"), stderr=subprocess.STDOUT).returncode
        s = summarize(model, variant)
        n = s["n"] if s else 0
        log(f"DONE {model} {variant} exit={code} samples={n}/{expected_samples(samples)} -> {json.dumps(s) if s else 'no results'}")
        if code == 0 and n >= expected_samples(samples):
            return s
        stopped = stop_orphaned_bench_servers()
        subprocess.run("docker ps --format '{{.Names}}' | grep '^omp_' | xargs -r docker rm -f",
                       shell=True, capture_output=True)
        log(f"RUNNER FAILURE {model} {variant}: exit={code}, {n}/{expected_samples(samples)} samples; "
            f"cleaned up server sessions {stopped}")
    raise RunnerFailure(f"{model} {variant} failed twice")


def unusable(s):
    return s is None or s["bad_fraction"] >= 0.5 or s["overall"] < 20


def ladder(model, pr):
    mx = max(pools(pr).values(), default=0)
    if mx < OMP_MIN:
        log(f"SKIP {model}: max context {mx} is below omp's minimum")
        return {}
    top = (mx // 1024) * 1024
    if top < CTX_FLOOR:
        log(f"SKIP {model}: max context {top} is below the {CTX_FLOOR} floor")
        return {}
    rungs = [top]
    if top > 40 * 1024:
        # same midpoint formula as before, so completed rungs (e.g. Gemma 72k) are reused
        rungs.append(((top + 32 * 1024) // 2 // 4096) * 4096)
    rungs.append(CTX_FLOOR)
    rungs = sorted({r for r in rungs if CTX_FLOOR <= r <= top}, reverse=True)
    results = {}
    for i, ctx in enumerate(rungs):
        v = rung_variant(pr, ctx)
        s = run(model, v, 2)
        results[v] = {"ctx": ctx, **(s or {})}
        if unusable(s):
            log(f"CLIFF {model}: unusable at {ctx} ({s})")
            break
        if i == 0 and s and s["overall"] < COMPETITIVE:
            log(f"NOT COMPETITIVE {model}: {s['overall']:.1f} at its largest context {ctx}; no more rungs")
            break
        if s and s["overall"] < 20 and s["bad_fraction"] < 0.5:
            # Failing with room to spare and few context errors = a capability or
            # tool-fit failure; a smaller window cannot help (Mellum at 92k, 2026-09-14,
            # thinking on or off). Stop descending.
            log(f"UNFIT {model}: {s['overall']:.1f} overall at its largest usable context {ctx} "
                f"without context pressure; skipping lower rungs")
            break
    # refine big drops
    ordered = sorted(results.items(), key=lambda kv: -kv[1]["ctx"])
    for (va, a), (vb, b) in zip(ordered, ordered[1:]):
        if "overall" in a and "overall" in b and abs(a["overall"] - b["overall"]) > 15:
            mid = ((a["ctx"] + b["ctx"]) // 2 // 4096) * 4096
            if mid not in (a["ctx"], b["ctx"]) and mid >= CTX_FLOOR:
                v = rung_variant(pr, mid)
                s = run(model, v, 2)
                results[v] = {"ctx": mid, **(s or {})}
    return results


def main():
    os.makedirs(RES, exist_ok=True)
    # One GPU: never start a benchmark server while the context probe still owns it.
    while probe_running():
        log("waiting for ctx_probe.py to finish (GPU busy)")
        time.sleep(300)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {"ladders": {}, "best": {}}
    for model in CHECKPOINTS:
        if probe(model) is None:
            # No probe result yet (e.g. a checkpoint fixed after the probe pass):
            # probe it now, while the campaign owns the GPU.
            log(f"PROBE {model}")
            subprocess.run(["/home/tate/qwen35/.venv/bin/python", "-u", os.path.join(HERE, "ctx_probe.py"), model],
                           stdout=open(os.path.join(RES, "_ctx", "probe.log"), "a"), stderr=subprocess.STDOUT)
        pr = wait_for_probe(model)
        if not pr or not pools(pr):
            log(f"SKIP {model}: no proven context (probe: {pr and pr.get('trials')})")
            continue
        log(f"LADDER {model}: proven max contexts {pools(pr)}")
        state["ladders"][model] = ladder(model, pr)
        usable = {v: r for v, r in state["ladders"][model].items() if "overall" in r and not unusable(r)}
        if usable:
            best = max(usable.items(), key=lambda kv: (round(kv[1]["overall"]), kv[1]["ctx"]))[0]
            state["best"][model] = best
            if model in SKIP_TOPUP:
                log(f"SKIP TOPUP {model}: best rung {best} stays at 2 samples (Tate's call)")
            else:
                s = run(model, best, 3)
                state["ladders"][model][best].update(s or {})
        json.dump(state, open(STATE, "w"), indent=2)

    ranked = sorted(state["best"].items(),
                    key=lambda kv: -(state["ladders"][kv[0]][kv[1]].get("overall") or 0))[:3]
    log(f"TOP3 {[(m, v, round(state['ladders'][m][v].get('overall', 0), 1)) for m, v in ranked]}")
    json.dump(state, open(STATE, "w"), indent=2)
    if os.environ.get("RUN_KNOB_SWEEPS") != "1":
        # Tate, 2026-09-14: stop before the knob sweeps and let him decide.
        log("READY FOR KNOB SWEEPS -- stopping here as requested (rerun with RUN_KNOB_SWEEPS=1 to continue)")
        return
    for model, best in ranked:
        for knob in KNOBS:
            run(model, f"{best}+{knob}", 2)
        run(model, best[:-3], 2)          # same rung, omp DEFAULT compaction (Tate's current setup)
    json.dump(state, open(STATE, "w"), indent=2)
    log("CAMPAIGN COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except RunnerFailure as e:
        log(f"HALT: {e} -- campaign stopped; fix the harness and rerun (completed work is kept)")
        sys.exit(2)
