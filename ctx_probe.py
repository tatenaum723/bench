#!/usr/bin/env python3
"""Find the maximum context each model can actually serve on this 12 GB card.

For each KV precision (fp8, bf16) the server starts single-user at the model's
native trained context with a high memory fraction (stepping down if CUDA-graph
capture OOMs). The usable maximum is min(native, allocated full-attention KV
pool). That number is then PROVEN with a real request filled to ~95% of it,
with a fact planted mid-document that must come back, and prefill time is
measured. If the proof fails, the context is cut 15% and retried.

  ./ctx_probe.py gemma4-12b
  ./ctx_probe.py --all

Writes results_omp/_ctx/<model>.json
"""
import glob
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from server import Server, chat, wait_until_safe  # noqa: E402

OUT = os.path.join(HERE, "results_omp", "_ctx")
NATIVE = {"gemma4-12b": 262144, "gemma4-12b-nvfp4": 262144, "gemma4-12b-awq": 131072,
          "granite42-8b": 131072, "granite42-8b-fp8": 131072,
          "mellum2-12b": 131072, "mellum2-12b-awq": 131072, "qwen3.5-9b": 262144}
SWA_MODELS = {"gemma4-12b", "gemma4-12b-nvfp4", "gemma4-12b-awq", "mellum2-12b", "mellum2-12b-awq"}
TOKENIZERS = {"gemma4-12b": "gemma4-12b", "gemma4-12b-nvfp4": "gemma4-12b", "gemma4-12b-awq": "gemma4-12b",
              "granite42-8b": "granite42-8b", "granite42-8b-fp8": "granite42-8b",
              "mellum2-12b": "mellum2-12b", "mellum2-12b-awq": "mellum2-12b", "qwen3.5-9b": "Qwen3.5-9B-AWQ-4bit"}


STDLIB = "/usr/lib/python3.12"


def filler_doc(n_tokens, tok, seed=11):
    """Real, varied text of the kind omp reads: Python standard-library source,
    concatenated file by file in a fixed order, with a passphrase planted as a
    code comment at 50% depth.

    v1 used random combinations of ~20 words. Mellum degenerates on repetitive
    input (seen earlier the same day), so that filler measured repetition
    sensitivity, not context capacity: it failed at 123k and 104k, then passed
    at 89k."""
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(STDLIB, "**", "*.py"), recursive=True))
    files = [f for f in files if "/test" not in f and "site-packages" not in f]
    chunks, count = [], 0
    for f in files:
        try:
            body = open(f, errors="replace").read()
        except OSError:
            continue
        piece = f"# ===== file: {os.path.relpath(f, STDLIB)} =====\n{body}\n"
        n = len(tok(piece)["input_ids"])
        if count + n > n_tokens:
            if count > n_tokens * 0.9:
                break
            continue
        chunks.append(piece)
        count += n
    secret_value = f"{rng.choice(['AMBER', 'COBALT', 'FALCON'])}-{rng.randint(1000, 9999)}"
    note = f"# NOTE for operators: the vault recovery phrase is {secret_value}\n"
    chunks.insert(len(chunks) // 2, note)
    return "".join(chunks), secret_value


def try_start(model, extra, mem_steps):
    for mem in mem_steps:
        srv = Server(model, extra=[*extra, "--mem-fraction-static", str(mem)],
                     log=os.path.join(OUT, f"{model}_server.log"))
        try:
            m = srv.start(timeout=1500)
            return srv, m, mem
        except Exception as e:
            print(f"    start failed at mem={mem}: {str(e).splitlines()[0][:160]}", flush=True)
    return None, None, None


def prove(srv, model, ctx, tok):
    doc, secret = filler_doc(int(ctx * 0.95) - 200, tok)
    prompt = ("Below is source code from a repository.\n\n" + doc +
              "\n\nSomewhere in the code above, a comment gives the vault recovery phrase. "
              "What is it? Reply with the phrase only.")
    n_prompt = len(tok(prompt)["input_ids"])
    t0 = time.time()
    try:
        d = chat(srv.base, {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 32,
                            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}},
                 # scale with the prompt: slowest prefill seen is ~700 tok/s, allow ~150
                 timeout=int(min(2400, max(600, n_prompt / 150))))
    except Exception as e:
        return {"ok": False, "prompt_tokens": n_prompt, "error": f"{type(e).__name__}: {str(e)[:300]}",
                "seconds": round(time.time() - t0, 1)}
    el = time.time() - t0
    out = d["choices"][0]["message"].get("content") or ""
    return {"ok": secret in out, "retrieved": secret in out, "answer": out[:80], "secret": secret,
            "prompt_tokens": d.get("usage", {}).get("prompt_tokens", n_prompt), "seconds": round(el, 1),
            "prefill_tok_s": round((d.get("usage", {}).get("prompt_tokens") or n_prompt) / el, 1),
            "alive": srv.proc.poll() is None}


def probe(model):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"/home/tate/qwen35/{TOKENIZERS[model]}")
    native = NATIVE[model]
    res = {"model": model, "native": native, "trials": []}
    for kv in ("fp8_e4m3", "auto"):
        extra = ["--kv-cache-dtype", kv, "--context-length", str(native), "--max-running-requests", "1"]
        if model in SWA_MODELS:
            # Single user: the sliding-window pool only needs window + one prefill chunk.
            extra += ["--swa-full-tokens-ratio", "0.02"]
        print(f"== {model} kv={kv}", flush=True)
        wait_until_safe()
        # The GPU also drives the desktop: never go above the 0.91 proven stable on this card.
        srv, metrics, mem = try_start(model, extra, (0.91, 0.89, 0.87))
        # SWA models: the sliding-window pool must hold the window (1024) plus a
        # prefill chunk (2048) or requests are never scheduled -- they just wait.
        # A fixed 0.02 ratio gave Gemma bf16 a 1,458-token pool and three
        # one-hour timeouts (2026-09-13). Size it from the measured full pool.
        if srv and model in SWA_MODELS and (metrics.get("swa_tokens") or 0) < 4096:
            full = metrics.get("full_tokens") or metrics.get("kv_tokens") or 1
            ratio = round(min(0.5, 5000 / full), 3)
            print(f"    swa pool {metrics.get('swa_tokens')} too small; restarting with --swa-full-tokens-ratio {ratio}", flush=True)
            srv.stop()
            extra = [x for x in extra]
            i = extra.index("--swa-full-tokens-ratio")
            extra[i + 1] = str(ratio)
            srv, metrics, mem = try_start(model, extra, (mem,) if mem else (0.91, 0.89))
        trial = {"kv": kv, "extra": extra, "mem": mem, "metrics": metrics, "proofs": []}
        if srv and model in SWA_MODELS and (metrics.get("swa_tokens") or 0) < 3200:
            print(f"    swa pool still {metrics.get('swa_tokens')}: cannot schedule requests, skipping {kv}", flush=True)
            srv.stop()
            res["trials"].append(trial)
            continue
        res["trials"].append(trial)
        if not srv:
            continue
        try:
            pool = metrics.get("full_tokens") or metrics.get("kv_tokens") or 0
            ctx = min(native, pool)
            print(f"    started mem={mem} pool={pool} -> candidate max context {ctx}  {metrics}", flush=True)
            for attempt in range(6):
                wait_until_safe()
                p = prove(srv, model, ctx, tok)
                p["ctx"] = ctx
                trial["proofs"].append(p)
                print(f"    proof @ {ctx}: {p}", flush=True)
                if p["ok"] and p.get("alive", True):
                    trial["max_context"] = ctx
                    break
                if srv.proc.poll() is not None:
                    print("    server died during proof", flush=True)
                    break
                ctx = int(ctx * 0.85)
        finally:
            srv.stop()
        json.dump(res, open(os.path.join(OUT, f"{model}.json"), "w"), indent=2)
    best = [t for t in res["trials"] if t.get("max_context")]
    if best:
        b = max(best, key=lambda t: t["max_context"])
        res["max_context"] = b["max_context"]
        res["max_context_kv"] = b["kv"]
        res["max_context_mem"] = b["mem"]
    json.dump(res, open(os.path.join(OUT, f"{model}.json"), "w"), indent=2)
    print(f"== {model}: max usable context {res.get('max_context')} (kv {res.get('max_context_kv')})", flush=True)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    models = list(NATIVE) if "--all" in sys.argv else [a for a in sys.argv[1:] if not a.startswith("-")]
    for m in models:
        probe(m)
