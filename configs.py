"""Named benchmark configurations.

A CONFIG is everything that can change a model's score:
  model profile (serve-model.sh) + server flags + env + sampling + thinking mode.
Results are stored under results/<model>/<variant>/<mode>/, so sweeps over
quantization, context, KV precision and sampling compare like with like.

Sampling defaults are each vendor's published recommendation (sources inline).
"""

# ---------------------------------------------------------------------------
# Models (one entry per downloaded checkpoint / serve profile)
# ---------------------------------------------------------------------------
MODELS = {
    "gemma4-12b": {
        "repo": "google/gemma-4-12B-it-qat-w4a16-ct",
        "quant": "QAT W4A16 (compressed-tensors)",
        # model card, "Best practices": temperature=1.0, top_p=0.95, top_k=64
        "sampling": {"think": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
                     "nothink": {"temperature": 1.0, "top_p": 0.95, "top_k": 64}},
        "context": 32768,
    },
    "granite42-8b": {
        "repo": "ibm-granite/granite-4.2-8b-nvfp4",
        "quant": "NVFP4 W4A4",
        # model card: "Use temperature=1.0 and top_p=0.95 across all tasks"
        "sampling": {"think": {"temperature": 1.0, "top_p": 0.95},
                     "nothink": {"temperature": 1.0, "top_p": 0.95}},
        "context": 32768,
    },
    "mellum2-12b": {
        "repo": "RedHatAI/Mellum2-12B-A2.5B-Thinking-NVFP4",
        "quant": "NVFP4 W4A4 (MoE 64x, 8 active)",
        # JetBrains card example: temperature=0.6, top_p=0.95, top_k=20
        "sampling": {"think": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                     "nothink": {"temperature": 0.6, "top_p": 0.95, "top_k": 20}},
        "context": 32768,
    },
    "qwen3.5-9b": {
        "repo": "cyankiwi/Qwen3.5-9B-AWQ-4bit",
        "quant": "AWQ W4A16",
        # Qwen card: thinking "precise coding" 0.6/0.95/20/pp0;
        # non-thinking general 0.7/0.8/20 presence_penalty 1.5
        "sampling": {"think": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
                     "nothink": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}},
        "context": 34816,
    },
}

# Quantization variants inherit the base model's sampling (same weights, same vendor guidance).
for _name, _base, _repo, _quant, _ctx in [
    ("gemma4-12b-nvfp4", "gemma4-12b", "RedHatAI/gemma-4-12B-it-NVFP4", "NVFP4 W4A4 (non-QAT)", 32768),
    ("gemma4-12b-awq", "gemma4-12b", "cyankiwi/gemma-4-12B-it-AWQ-INT4", "AWQ W4A16 g32 (non-QAT)", 32768),
    ("granite42-8b-fp8", "granite42-8b", "ibm-granite/granite-4.2-8b-fp8", "FP8 W8A8 + fp8 KV", 16384),
    ("mellum2-12b-awq", "mellum2-12b", "cyankiwi/Mellum2-12B-A2.5B-Thinking-AWQ-INT4", "AWQ W4A16 g32 (MoE)", 32768),
]:
    MODELS[_name] = {**MODELS[_base], "repo": _repo, "quant": _quant, "context": _ctx, "base": _base}

# ---------------------------------------------------------------------------
# Variants: deviations from a model's defaults. "default" = vendor settings,
# profile flags as in serve-model.sh. Add sweep points here.
# ---------------------------------------------------------------------------
VARIANTS = {
    "default": {},
    "pilot": {},          # same as default; kept separate so harness shake-out runs never mix into results
    # sampling sweeps
    "temp0.2": {"sampling": {"temperature": 0.2}},
    "greedy": {"sampling": {"temperature": 0.0}},
    # KV precision: fp8 halves KV memory -> more context, possible quality cost
    "kvfp8": {"server": ["--kv-cache-dtype", "fp8_e4m3"]},
    # context sweeps (the server's KV pool, not just the advertised limit)
    "ctx16k": {"server": ["--context-length", "16384"]},
    "ctx64k-kvfp8": {"server": ["--context-length", "65536", "--kv-cache-dtype", "fp8_e4m3"]},
    # --- model-specific (a variant with "models" only applies to those) ---
    # Granite 4.2 card: low-effort thinking via chat template kwarg
    "low-effort": {"models": ["granite42-8b", "granite42-8b-fp8"],
                   "request": {"chat_template_kwargs": {"low_effort": True}}},
    # Mellum thinking fell into repetition loops on repetitive input
    "presence1.0": {"sampling": {"presence_penalty": 1.0}},
    "reppen1.05": {"sampling": {"repetition_penalty": 1.05}},
    # Gemma: SWA:full KV ratio trades sliding-window headroom for full-attention context
    "swa0.05": {"models": ["gemma4-12b", "gemma4-12b-nvfp4", "gemma4-12b-awq"],
                "server": ["--swa-full-tokens-ratio", "0.05"]},
    "gemma-kvfp8-64k": {"models": ["gemma4-12b", "gemma4-12b-nvfp4", "gemma4-12b-awq"],
                        "server": ["--kv-cache-dtype", "fp8_e4m3", "--context-length", "65536"]},
}

MODES = ("nothink", "think")

# Extra generation budget when thinking is on (reasoning tokens are extra).
THINK_EXTRA_TOKENS = 16000   # 12k let Granite exhaust budgets drafting whole solutions inside <think>; 16k is the most that keeps Kepler (8k + 16k) inside a 32k window


def resolve(model, variant="default", mode="nothink"):
    m = MODELS[model]
    v = VARIANTS[variant]
    if v.get("models") and model not in v["models"]:
        raise ValueError(f"variant {variant!r} only applies to {v['models']}")
    sampling = dict(m["sampling"][mode])
    sampling.update(v.get("sampling", {}))
    return {
        "model": model, "variant": variant, "mode": mode,
        "repo": m["repo"], "quant": m["quant"],
        "server_extra": list(v.get("server", [])),
        "request": dict(v.get("request", {})),
        "env": dict(v.get("env", {})),
        "sampling": sampling,
        "thinking": mode == "think",
        "context": _context(m, v),
    }


def _context(m, v):
    s = v.get("server", [])
    if "--context-length" in s:
        return int(s[s.index("--context-length") + 1])
    return m["context"]
