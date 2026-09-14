# Local model benchmark

## omp harness (current) — models evaluated THROUGH omp

Each task is a real `omp -p --mode json` session in the `bench-omp:1` container
(`omp-sandbox/`): `--network none`, read-only root, no capabilities, non-root. The
container's only route out is a unix socket bridged to SGLang on :30000
(`model_bridge.py`), so model-issued commands can't reach the internet or the host.

```bash
./omp_selftest.py                         # graders vs known-good/bad workspaces (incl. conftest tampering)
./ctx_probe.py gemma4-12b                 # prove max usable context (qwen35 venv python; needs transformers)
./omp_run.py qwen3.5-9b --variant ctx48k-kvfp8-ct          # one config; resumable
./omp_run.py gemma4-12b --variant ctx118k-kvfp8-ct+think   # knob stacked on a ladder rung
./omp_run.py --table --tasks              # live score table
setsid nohup .venv/bin/python -u omp_campaign.py > results_omp/_campaign.log 2>&1 &
```

Variant grammar: `ctx<N>k[-kvfp8][-ct][+knob...]` — context rung, fp8 KV, tuned
compaction (`-ct`), knobs `think`, `max2048`, `max8192`, `temp0.2`, `lean-tools`.
The campaign (Tate's coarse-to-fine scope) ladders each checkpoint from its proven max
down to a 48k floor, tops the best rung up to 3 samples, and stops before knob sweeps
unless `RUN_KNOB_SWEEPS=1`. Local SGLang/model edits it depends on are in `patches/`.

Key findings baked into the harness: omp's default compaction (reserve 16384, keep
20000) breaks local 28-32k windows; see `tuned_compaction()` in `omp_run.py`.

---

# v2 raw-API harness (superseded; kept for reference)

Benchmarks SGLang-served local models on the 12 GB RTX 5070 for use as a
coding agent (omp). Everything a model writes is **executed**, in a
network-less sandbox, and graded on behaviour, not vocabulary.

## Quick start

```bash
cd ~/bench
./selftest.py                          # prove every scorer discriminates (run after ANY scorer change)
./bringup.py gemma4-12b                # real-weights smoke test: parsers, 28k-token prompt, speed
./run.py gemma4-12b                    # full suite, thinking on+off, 3 samples, starts/stops server
./run.py granite42-8b --variant kvfp8  # a sweep point from configs.py
./compare.py                           # table across everything run so far
```

Servers are launched through `~/qwen35/serve-model.sh <profile>`. The runner
stops them by signalling the server's whole **session** (see "Safety").

## Layout

| file | role |
|---|---|
| `suite.py` | single-turn tasks: coding, reasoning, tools, security (prompts + hidden tests) |
| `agentic.py` | multi-turn tool-loop tasks in a sandboxed repo, graded on final repo state |
| `scoring.py` | answer extraction, executed-test scorer, typed tool-call scorers |
| `kepler.py` + `canvas_probe.py` | runs the Kepler page's JS in V8 and measures the drawn orbit |
| `sandbox.py` | Docker sandbox (`bench-sandbox:1`, built from `sandbox/Dockerfile`) |
| `configs.py` | models (vendor sampling) and variants (quant / KV / context / sampling sweeps) |
| `server.py` | start/measure/stop SGLang safely; GPU peak-VRAM sampler |
| `run.py` | runs configs, resumable, 2 concurrent requests |
| `selftest.py` + `fixtures/` | known-good / known-bad responses per task |
| `bringup.py` | pre-flight check of a serving config on real weights |
| `legacy/` | v1 harness and hand-written Claude answers (retired) |

## Categories and scoring

Overall = mean of the five category scores (a category's task count does not
set its influence). Within a category, tasks are weighted.

- **coding** – Kepler orbit (V8-executed, physics gated: ellipse → star at focus
  → equal areas), `parse_duration`, `LRUCache` (incl. an O(1) timing test sized
  from sandbox measurements), interval-merge bug fix. Hidden pytest suites.
- **agentic** – bug fix with visible failing tests (editing the tests is detected
  and capped at 0.25), feature from a README spec, multi-file investigation with
  decoys. Score counts only hidden tests the agent *made* pass (the unmodified
  repo's passes are subtracted). Episodes that don't finish cleanly are ×0.8.
- **tools** – single-turn calls with **type-checked** arguments (a string
  `"true"` is not a boolean), parallel calls, nested args, restraint, and
  refusing to invent a missing identifier.
- **reasoning** – graded from the last `ANSWER:` line of the *content*
  (half credit if correct but the line is missing). Long context is ~20.6k
  tokens of varied logs with a two-hop question plus retired/look-alike
  distractors.
- **security** (defensive) – the fixed code is run against real payloads:
  SQLi/XSS via Flask test client, command + option injection with a fake
  `ping`, KDF cost measured *relative to SHA-256 on the same machine*, tarslip
  with traversal/absolute/symlink/hardlink archives. Explanations are a minority
  of the score.

## What changed from v1 (and why)

| v1 flaw | v2 |
|---|---|
| Graded text included the reasoning trace, so discarded ideas earned credit | content / tool_calls only |
| Security and coding graded by regex; mentioning `shell=True` passed "no shell=True" | code executed against payloads + hidden tests |
| `case_sensitive="true"` (string) passed as boolean | JSON types checked |
| Needle test was one sentence ×300; everything passed | 20k varied tokens, two hops, distractors |
| Kepler checks summed independently; star-at-centre page scored 0.81 | gated physics; same page scores 0.40 |
| One sample at temp 0.2 | 3 seeded samples, vendor sampling, thinking on and off |
| Wrong/missing parsers would silently zero tool scores | `bringup.py` verifies parsers on real weights first |
| No agentic evaluation | 3 multi-turn sandboxed tasks |
| Claude baseline hand-written by the scorer's author | removed |

## Safety

- Model output is untrusted. All execution is in `bench-sandbox:1`:
  `--network none --read-only --cap-drop ALL --security-opt no-new-privileges`,
  uid 10001, pids/memory/cpu limits, throwaway workspace. Host-side file tools
  refuse paths that escape the workspace.
- **Never `kill -9` an SGLang launch_server parent.** Its children get
  reparented to `systemd --user` and SIGQUIT their parent on failure, which
  core-dumps the whole GNOME session (2026-09-13). `server.py` signals every
  process in the server's session (ninja JIT jobs sit in their own process
  groups, so a group kill leaves compilers behind).
- `serve-model.sh` caps `MAX_JOBS=3`: FlashInfer's FP4/MoE kernel JIT otherwise
  runs 16 × ~5 GB compilers and triggers the kernel OOM killer.
