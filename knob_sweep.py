#!/usr/bin/env python3
"""Gemma 4 12B QAT knob sweep at its best rung (118k, fp8 KV, tuned compaction), 2 samples each:
thinking on, 2k and 8k reply budgets, and omp's default compaction. Chosen as the daily model."""
import sys
import omp_campaign as C

MODEL, BEST = "gemma4-12b", "ctx118k-kvfp8-ct"
try:
    C.log(f"SWEEP {MODEL} at {BEST}")
    for knob in ("think", "max2048", "max8192"):
        C.run(MODEL, f"{BEST}+{knob}", 2)
    C.run(MODEL, BEST[:-3], 2)                 # same rung, omp's default compaction
    C.log("KNOB SWEEP COMPLETE")
except C.RunnerFailure as e:
    C.log(f"HALT: {e}")
    sys.exit(2)
