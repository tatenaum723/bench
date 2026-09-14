#!/usr/bin/env python3
"""Diagnose Mellum's bring-up failures before benchmarking: is the malformed
no-think tool call a sampling artefact, a thinking-mode artefact, or systematic?
Does a 28k prompt produce sane output? Writes results/_bringup/mellum_probe.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bringup import TOOL  # noqa: E402
from server import Server, chat  # noqa: E402

profile = sys.argv[1] if len(sys.argv) > 1 else "mellum2-12b"
out = {"profile": profile, "tool": [], "long": []}
srv = Server(profile)
try:
    srv.start()
    B = srv.base
    for thinking in (False, True):
        for temp in (1.0, 0.6, 0.0):
            ok = 0
            examples = []
            for seed in range(8):
                d = chat(B, {"model": profile, "max_tokens": 1024 if not thinking else 4096,
                             "temperature": temp, "top_p": 0.95, "top_k": 20, "seed": seed,
                             "messages": [{"role": "user", "content": "What's the weather in Oslo in celsius? Use the tool."}],
                             "tools": [TOOL], "chat_template_kwargs": {"enable_thinking": thinking}})
                m = d["choices"][0]["message"]
                good = bool(m.get("tool_calls")) and "oslo" in (m["tool_calls"][0]["function"]["arguments"] or "").lower()
                ok += good
                if not good and len(examples) < 2:
                    examples.append((m.get("content") or "")[:300])
            row = {"thinking": thinking, "temperature": temp, "ok": ok, "of": 8, "bad_examples": examples}
            out["tool"].append(row)
            print(json.dumps(row)[:400], flush=True)
    filler_line = "The quick brown fox jumps over the lazy dog near the river bank. "
    for n_tok in (4000, 12000, 20000, 28000):
        for thinking in (False, True):
            filler = filler_line * (n_tok // 14)
            d = chat(B, {"model": profile, "max_tokens": 64 if not thinking else 3000, "temperature": 0.6,
                         "top_p": 0.95, "top_k": 20, "seed": 1,
                         "messages": [{"role": "user", "content": filler + "\n\nIgnore the text above. Reply with the word DONE."}],
                         "chat_template_kwargs": {"enable_thinking": thinking}}, timeout=900)
            m = d["choices"][0]["message"]
            row = {"tokens": d["usage"]["prompt_tokens"], "thinking": thinking,
                   "content": (m.get("content") or "")[:200], "reasoning_head": (m.get("reasoning_content") or "")[:200],
                   "finish": d["choices"][0]["finish_reason"]}
            out["long"].append(row)
            print(json.dumps(row)[:400], flush=True)
finally:
    srv.stop()
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "_bringup",
                                 f"{profile}_probe.json"), "w"), indent=2)
