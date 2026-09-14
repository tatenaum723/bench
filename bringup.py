#!/usr/bin/env python3
"""Bring a model up with real weights and prove the serving config is sound
BEFORE benchmarking it. A wrong parser silently zeroes the tool-calling score,
and a context length the KV pool can't hold fails only on long requests, so
both are tested directly here.

  ./bringup.py gemma4-12b
  ./bringup.py gemma4-12b --extra "--swa-full-tokens-ratio 0.1" --tag swa01

Writes results/_bringup/<profile>[-tag].json
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import MODELS  # noqa: E402
from server import Server, chat, stream_timing  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "_bringup")

TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"},
        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}

LEAK = re.compile(r"<think>|</think>|<tool_call>|<function=|<\|tool_call|<\|channel|call:", re.I)


def ask(base, model, content, thinking, tools=None, max_tokens=2048):
    # Send the vendor's sampling explicitly. With none, SGLang falls back to
    # OpenAI defaults (no top-k/top-p) for models whose generation_config lacks
    # them, and Mellum then emitted a corrupted tool call -- a false parser failure.
    sampling = MODELS.get(model, {}).get("sampling", {}).get("think" if thinking else "nothink", {})
    p = {"model": model, "messages": [{"role": "user", "content": content}],
         "max_tokens": max_tokens, **sampling,
         "chat_template_kwargs": {"enable_thinking": thinking}}
    if tools:
        p["tools"] = tools
    t0 = time.time()
    d = chat(base, p)
    m = d["choices"][0]["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
            "tool_calls": m.get("tool_calls") or [],
            "finish": d["choices"][0].get("finish_reason"),
            "usage": d.get("usage", {}), "s": round(time.time() - t0, 1)}


def check(name, ok, detail):
    print(f"   {'PASS' if ok else 'FAIL'}  {name:34} {detail[:110]}")
    return {"name": name, "ok": bool(ok), "detail": detail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--extra", default="", help="extra SGLang args")
    ap.add_argument("--tag", default="")
    ap.add_argument("--long-tokens", type=int, default=28000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    srv = Server(a.profile, extra=a.extra.split())
    print(f"== {a.profile} {a.extra}")
    res = {"profile": a.profile, "extra": a.extra, "checks": []}
    C = res["checks"]
    try:
        res["metrics"] = srv.start()
        print("   metrics:", json.dumps(res["metrics"]))
        B, M = srv.base, a.profile

        r = ask(B, M, "What is 17 * 23? Reply with just the number.", thinking=False)
        C.append(check("no-think: answers, no reasoning", "391" in r["content"] and not r["reasoning"],
                       f"content={r['content'][:60]!r} reasoning_len={len(r['reasoning'])}"))
        C.append(check("no-think: no leaked tags", not LEAK.search(r["content"]), r["content"][:80]))

        r = ask(B, M, "What is 17 * 23? Reply with just the number.", thinking=True, max_tokens=4096)
        C.append(check("think: reasoning separated", bool(r["reasoning"]) and "391" in r["content"],
                       f"content={r['content'][:60]!r} reasoning_len={len(r['reasoning'])} finish={r['finish']}"))
        C.append(check("think: no leaked tags", not LEAK.search(r["content"]), r["content"][:80]))

        for thinking in (False, True):
            r = ask(B, M, "What's the weather in Oslo in celsius? Use the tool.", thinking,
                    tools=[TOOL], max_tokens=4096)
            tc = r["tool_calls"]
            args = {}
            if tc:
                try:
                    args = json.loads(tc[0]["function"]["arguments"])
                except Exception:
                    args = {"_unparseable": tc[0]["function"]["arguments"]}
            ok = bool(tc) and tc[0]["function"]["name"] == "get_weather" and "oslo" in str(args).lower()
            C.append(check(f"tool call parsed (think={thinking})", ok,
                           f"calls={[c['function']['name'] for c in tc]} args={args} content={r['content'][:50]!r}"))
            C.append(check(f"tool: no raw markup (think={thinking})", not LEAK.search(r["content"]),
                           r["content"][:80]))

        # Multi-turn: feed the tool result back and require it to be used.
        msgs = [{"role": "user", "content": "What's the weather in Oslo? Use the tool."},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": json.dumps({"city": "Oslo"})}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(
                    {"city": "Oslo", "temp_c": -7, "condition": "heavy snow"})}]
        d = chat(B, {"model": M, "messages": msgs, "tools": [TOOL], "max_tokens": 2048,
                     "chat_template_kwargs": {"enable_thinking": False}})
        c = d["choices"][0]["message"].get("content") or ""
        C.append(check("tool result consumed", "-7" in c or "snow" in c.lower(), c[:100]))

        # Long context: does a request near the configured context actually run?
        filler = "The quick brown fox jumps over the lazy dog near the river bank. " * (a.long_tokens // 14)
        t0 = time.time()
        try:
            d = chat(B, {"model": M, "messages": [{"role": "user", "content":
                     filler + "\n\nIgnore the text above. Reply with the word DONE."}],
                     "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}}, timeout=900)
            u = d.get("usage", {})
            C.append(check(f"long prompt ~{a.long_tokens} tok", "done" in (d["choices"][0]["message"]
                           .get("content") or "").lower(),
                           f"prompt_tokens={u.get('prompt_tokens')} {time.time()-t0:.1f}s"))
            res["long_prefill_tok_s"] = round((u.get("prompt_tokens") or 0) / (time.time() - t0), 1)
        except Exception as e:
            C.append(check(f"long prompt ~{a.long_tokens} tok", False, f"{type(e).__name__}: {e}"))

        res["speed"] = stream_timing(B, M, "Write a 300-word explanation of how TCP congestion control works.",
                                     max_tokens=512, extra={"chat_template_kwargs": {"enable_thinking": False}})
        print("   speed:", res["speed"], "long prefill tok/s:", res.get("long_prefill_tok_s"))
        alive = srv.proc.poll() is None
        C.append(check("server survived all checks", alive, "" if alive else srv.tail()))
    except Exception as e:
        res["fatal"] = f"{type(e).__name__}: {e}"
        print("   FATAL", res["fatal"][:500])
    finally:
        srv.stop()
        res["peak_vram_mib"] = srv.gpu.peak_mib
        res["peak_temp_c"] = srv.gpu.peak_temp
    res["passed"] = sum(c["ok"] for c in C)
    res["total"] = len(C)
    print(f"   {res['passed']}/{res['total']} checks  peak VRAM {res['peak_vram_mib']} MiB")
    name = a.profile + (f"-{a.tag}" if a.tag else "")
    json.dump(res, open(os.path.join(OUT, name + ".json"), "w"), indent=2)


if __name__ == "__main__":
    main()
