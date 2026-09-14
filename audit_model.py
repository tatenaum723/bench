#!/usr/bin/env python3
"""Supply-chain audit for a Hugging Face model repo, without downloading weights.

What this CAN establish:
  - file format safety      (safetensors = inert data; pickle = arbitrary code on load)
  - remote code execution   (auto_map / trust_remote_code / .py files in the repo)
  - HF's own scan verdicts  (ClamAV + pickle scanner, exposed per-file in the API)
  - chat-template risk      (Jinja is evaluated at inference time)
  - provenance              (org vs individual, downloads, age, likes)

What this CANNOT establish, and nobody's static tool can:
  - a trojaned/backdoored model: weights that behave normally except on a secret
    trigger. Weights are inert numbers; a backdoor is a BEHAVIOUR, so only
    behavioural testing or provenance can address it.

Usage:  audit_model.py <repo> [<repo> ...]
"""
import json
import re
import sys
import urllib.request

API = "https://huggingface.co/api/models/"
RAW = "https://huggingface.co/{}/resolve/main/{}"

G, Y, R, DIM, B, RST = ("\033[92m", "\033[93m", "\033[91m",
                        "\033[2m", "\033[1m", "\033[0m")

# Extensions that execute code when loaded by torch/pickle.
PICKLE_EXT = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".joblib", ".msgpack")
# Jinja constructs that would be alarming inside a chat template.
TEMPLATE_RED_FLAGS = [
    (r"\bimport\b", "import statement"),
    (r"\b__class__|__mro__|__subclasses__|__globals__", "sandbox-escape dunder"),
    (r"\bos\.|\bsubprocess\b|\bsocket\b|\beval\b|\bexec\b", "system/eval access"),
    (r"https?://", "outbound URL"),
]


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "audit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def j(url):
    return json.loads(get(url))



def arch_of(cfg):
    a = cfg.get("architectures") or (cfg.get("text_config") or {}).get("architectures")
    return (a or [None])[0]


def _sglang_has_arch(arch):
    """True if the installed stock SGLang registers this architecture natively.

    This is the difference between 'auto_map means arbitrary code execution' and
    'auto_map is inert because the serving stack has its own implementation'.
    """
    if not arch:
        return None
    try:
        import subprocess
        out = subprocess.run(
            ["/home/tate/qwen35/.venv/bin/python", "-c",
             "from sglang.srt.models.registry import ModelRegistry;"
             "print('\\n'.join(ModelRegistry.get_supported_archs()))"],
            capture_output=True, text=True, timeout=120).stdout
        return arch if arch in out.split() else None
    except Exception:
        return None


def audit(repo):
    print(f"\n{B}=== {repo} ==={RST}")
    findings = []          # (severity, text)  severity: ok | note | warn | crit

    try:
        meta = j(API + repo + "?blobs=true")
    except Exception as e:
        print(f"  {R}could not fetch metadata: {e}{RST}")
        return
    # Per-file scan verdicts live on the tree endpoint, not the model endpoint.
    try:
        tree = j(API + repo + "/tree/main?expand=true")
        tree = tree if isinstance(tree, list) else []
    except Exception:
        tree = []

    # ---------- provenance ------------------------------------------------
    author = meta.get("author") or repo.split("/")[0]
    dl = meta.get("downloads")
    likes = meta.get("likes")
    created = (meta.get("createdAt") or "")[:10]
    KNOWN_ORGS = {"google", "meta-llama", "microsoft", "ibm-granite", "nvidia",
                  "RedHatAI", "Qwen", "mistralai", "deepseek-ai", "openai",
                  "inclusionAI", "JetBrains", "zai-org", "THUDM", "unsloth",
                  "neuralmagic", "ByteDance-Seed", "allenai"}
    # Three tiers. v1 only knew a hard-coded list and called every other org
    # (cyankiwi, Intel, QUASAR-QAT...) "individual"; ask the Hub instead.
    org = None
    try:
        org = j(f"https://huggingface.co/api/organizations/{author}/overview")
    except Exception:
        org = None
    if author in KNOWN_ORGS:
        findings.append(("ok", f"publisher '{author}' is a known organisation"))
    elif org and org.get("name"):
        verified = "verified" if org.get("isVerified") else "unverified"
        findings.append(("warn", f"publisher '{author}' is a Hub organisation ({verified}) not on the "
                                 f"known list - check who runs it"))
    else:
        findings.append(("warn", f"publisher '{author}' is an individual account "
                                 f"- provenance rests on nothing but the account itself"))
    findings.append(("note", f"downloads={dl}  likes={likes}  created={created}"))

    sibs = meta.get("siblings") or []
    names = [s["rfilename"] for s in sibs]

    # ---------- file format safety ---------------------------------------
    pickles = [n for n in names if n.lower().endswith(PICKLE_EXT)]
    safet = [n for n in names if n.endswith(".safetensors")]
    if pickles:
        findings.append(("crit", f"{len(pickles)} pickle-format file(s) present: "
                                 f"{', '.join(pickles[:4])} - these EXECUTE CODE on load"))
    if safet and not pickles:
        findings.append(("ok", f"all {len(safet)} weight files are .safetensors "
                               f"(inert tensor data, no code execution)"))
    elif not safet:
        findings.append(("warn", "no .safetensors weights found"))

    # ---------- remote code execution ------------------------------------
    pyfiles = [n for n in names if n.endswith(".py")]
    if pyfiles:
        findings.append(("warn", f"{len(pyfiles)} Python file(s) in repo: "
                                 f"{', '.join(pyfiles[:6])}"))
    else:
        findings.append(("ok", "no .py files in the repo"))

    try:
        cfg = j(RAW.format(repo, "config.json").replace("https://huggingface.co/",
                                                        "https://huggingface.co/"))
    except Exception:
        cfg = {}
    if cfg.get("auto_map"):
        native = _sglang_has_arch(arch_of(cfg))
        if native:
            findings.append(("note", f"config.json declares auto_map, BUT stock SGLang "
                                     f"registers {native} natively -- with "
                                     f"trust_remote_code=False (the default) the repo's "
                                     f".py files are never imported or executed"))
            findings.append(("warn", "NEVER pass --trust-remote-code with this repo; "
                                     "that would execute its bundled Python"))
        else:
            findings.append(("crit", f"config.json declares auto_map={list(cfg['auto_map'])} "
                                     f"and no native implementation was found -> loading "
                                     f"would require trust_remote_code and RUN repo code"))
    else:
        findings.append(("ok", "no auto_map in config.json (no remote code execution path)"))
    arch = cfg.get("architectures") or (cfg.get("text_config") or {}).get("architectures")
    if arch:
        findings.append(("note", f"architectures={arch}"))

    # ---------- HF's own scanners ----------------------------------------
    scanned = unsafe = unscanned = 0
    scanners = set()
    for e in tree:
        st = e.get("securityFileStatus") or {}
        if not st:
            continue
        scanned += 1
        for key in ("avScan", "protectAiScan", "pickleImportScan", "jFrogScan"):
            sub = st.get(key)
            if not isinstance(sub, dict):
                continue
            status = (sub.get("status") or "").lower()
            # 'unscanned' is NOT a finding: each scanner only covers the file
            # types it understands (the pickle scanner skips .json, ClamAV skips
            # large tensor blobs). Treating it as a warning produced ~40 false
            # alarms per repo and buried the one real finding.
            if status == "unscanned":
                continue
            scanners.add(key)
            if status in ("unsafe", "malicious", "infected", "suspicious"):
                unsafe += 1
                findings.append(("crit", f"{key} flagged {e.get('path')}: "
                                         f"{sub.get('message') or status}"))
            elif status not in ("safe", "", "not_scanned", "queued", "skipped"):
                findings.append(("warn", f"{key} status '{status}' on {e.get('path')}"))
        if (st.get("status") or "").lower() not in ("safe", ""):
            unscanned += 1
            findings.append(("warn", f"file-level status '{st.get('status')}' "
                                     f"on {e.get('path')}"))
    if scanned and not unsafe:
        findings.append(("ok", f"{scanned} files scanned clean by "
                               f"{', '.join(sorted(scanners)) or 'HF scanners'}"))
    elif not scanned:
        findings.append(("note", "no per-file scan records exposed for this repo"))

    # ---------- chat template --------------------------------------------
    tmpl = None
    for cand in ("chat_template.jinja", "tokenizer_config.json"):
        if cand in names:
            try:
                body = get(RAW.format(repo, cand)).decode("utf-8", "replace")
                tmpl = body if cand.endswith(".jinja") else \
                    (json.loads(body).get("chat_template") or "")
                if tmpl:
                    break
            except Exception:
                pass
    if tmpl:
        hits = [lbl for rx, lbl in TEMPLATE_RED_FLAGS if re.search(rx, tmpl)]
        if hits:
            findings.append(("warn", f"chat template contains: {', '.join(hits)} "
                                     f"- Jinja is evaluated at inference, inspect manually"))
        else:
            findings.append(("ok", "chat template has no imports, dunders, or URLs"))
    else:
        findings.append(("note", "no chat template found to inspect"))

    # ---------- quantization provenance ----------------------------------
    q = cfg.get("quantization_config") or {}
    if q:
        findings.append(("note", f"quant: {q.get('quant_method')} / "
                                 f"{q.get('format', '')} "
                                 f"(ignore list: {len(q.get('ignore') or [])} entries)"))

    order = {"crit": 0, "warn": 1, "ok": 2, "note": 3}
    col = {"crit": R, "warn": Y, "ok": G, "note": DIM}
    mark = {"crit": "CRIT", "warn": "WARN", "ok": " OK ", "note": "note"}
    for sev, txt in sorted(findings, key=lambda f: order[f[0]]):
        print(f"  {col[sev]}[{mark[sev]}]{RST} {txt}")

    worst = min((order[s] for s, _ in findings), default=3)
    verdict = {0: f"{R}DO NOT LOAD without review{RST}",
               1: f"{Y}loadable, but provenance/soft risks noted{RST}",
               2: f"{G}no static red flags{RST}", 3: "inconclusive"}[worst]
    print(f"  -> {verdict}")


if __name__ == "__main__":
    repos = sys.argv[1:] or [
        "google/gemma-4-12B-it-qat-w4a16-ct",
        "RedHatAI/Mellum2-12B-A2.5B-Thinking-NVFP4",
        "inclusionAI/Ling-3.0-tiny-fp8",
        "ibm-granite/granite-4.2-8b-nvfp4",
        "cyankiwi/Qwen3.5-9B-AWQ-4bit",
    ]
    for r in repos:
        audit(r)
    print(f"\n{DIM}Reminder: none of the above can detect a behavioural backdoor "
          f"(weights that misbehave only on a secret trigger).{RST}")
