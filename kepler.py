"""Kepler-orbit task scorer: executes the model's page in V8 (canvas_probe)
and measures the geometry it actually draws. Grades content only."""
import re

import json
import os
import subprocess
import sys
import tempfile

try:
    import py_mini_racer  # noqa: F401  (only checking availability; V8 runs in a child process)
    HAVE_V8 = True
except Exception:          # py_mini_racer unavailable
    HAVE_V8 = False

_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from canvas_probe import run_canvas, analyse_orbit
html = open(sys.argv[2]).read()
log, canv, err = run_canvas(html, frames=400)
out = {"err": err, "js_err": sum(1 for e in log if e["op"] == "__error")}
if not err:
    a = analyse_orbit(log, canv)
    a.pop("pts", None)
    out["analysis"] = a
print(json.dumps(out))
"""


def probe_page(html, retries=1):
    """Run the page in V8 in a CHILD process. libmini_racer segfaulted inside the
    benchmark runner on 2026-09-14, killing the runner, orphaning its SGLang server
    and cascading through the campaign. A crash here now costs one retry, not the run."""
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(html)
        path = f.name
    try:
        for attempt in range(retries + 1):
            try:
                r = subprocess.run([sys.executable, "-c", _CHILD, here, path],
                                   capture_output=True, text=True, timeout=180)
            except subprocess.TimeoutExpired:
                return {"err": "page did not finish 400 frames within 180 s"}
            if r.returncode == 0 and r.stdout.strip():
                return json.loads(r.stdout.strip().splitlines()[-1])
            last = f"probe process exited {r.returncode}: {r.stderr[-200:]}"
        return {"err": last, "probe_crashed": True}
    finally:
        os.unlink(path)


def extract_html(text):
    """The page the model produced: a full <!DOCTYPE html>...</html> document,
    else the first fenced block containing a <canvas> or <script>."""
    m = re.search(r"<!DOCTYPE html.*?</html>", text, re.S | re.I)
    if m:
        return m.group(0)
    for m in re.finditer(r"```(?:html)?[ \t]*\n(.*?)```", text, re.S):
        if re.search(r"<canvas|<script", m.group(1), re.I):
            return m.group(1)
    return text if re.search(r"<script", text, re.I) else ""


def kepler_scorer(r, _task):
    """Grade the Kepler task by EXECUTING it, falling back to static checks.

    Static analysis proved insufficient: it scored 83% on output whose script
    did not parse at all (`const mu` declared three times) and which, had it
    parsed, would have drawn a 1-pixel circle with the star at its centre.
    canvas_probe runs the model's own JS against a recording canvas and
    measures the resulting geometry. Verified against a known-good
    implementation: e=0.6000, star-to-focus 0.00 px, Kepler-2 CV 0.00003.
    """
    if not HAVE_V8:
        return (*_kepler_static(r), {"mode": "static-fallback"})

    html = extract_html(r.get("content") or "")

    res = probe_page(html)
    err = res.get("err")
    if err:
        # Does not even run. This is a hard zero on the physics; award only a
        # token amount for having produced plausible-looking structure.
        stat, _ = _kepler_static(r)
        return min(0.15, stat * 0.15), f"CODE DOES NOT RUN: {err[:120]}", {"error": err, **res}

    js_err = [None] * res.get("js_err", 0)
    a = res["analysis"]
    if a.get("error"):
        return 0.15, f"ran but drew nothing usable: {a['error']}", {"analysis": a}

    pts = []          # (score, weight, label)
    cw, ch = a["canvas"]
    frac = a["planet_spread_px"] / max(cw, ch) if max(cw, ch) else 0
    visible = frac > 0.15
    f = a.get("fit")
    ellipse_ok = focus_ok = False
    if not f:
        pts.append((0.0, 0.25, "planet path is not an ellipse (conic fit failed)"))
        pts.append((0.0, 0.25, "no ellipse -> no focus"))
        pts.append((0.0, 0.35, "no ellipse -> Kepler 2 not assessable"))
    else:
        de = abs(f["e"] - 0.6)
        ellipse_ok = de < 0.05
        pts.append((1.0 if ellipse_ok else (0.5 if de < 0.15 else 0.0), 0.25,
                    f"eccentricity {f['e']:.3f} (target 0.600)"))
        fe, ce = a.get("star_focus_err_px"), a.get("star_centre_err_px")
        if fe is None:
            pts.append((0.0, 0.25, "no stationary star drawn"))
        else:
            tol = max(6.0, 0.06 * f["a"])
            focus_ok = de < 0.15 and fe < tol and (ce or 0) > tol
            pts.append((1.0 if focus_ok else 0.0, 0.25,
                        f"star {fe:.1f}px from focus, {ce:.1f}px from centre"))
        cv = a.get("area_rate_cv")
        if not focus_ok:
            pts.append((0.0, 0.35, "star not at focus -> equal-area check not credited"
                        + (f" (CV {cv:.3f})" if cv is not None else "")))
        elif cv is None:
            pts.append((0.0, 0.35, "could not measure swept area"))
        else:
            pts.append((1.0 if cv < 0.10 else (0.5 if cv < 0.30 else 0.0), 0.35,
                        f"Kepler-2 area CV {cv:.3f} (needs <0.10)"))

    has = lambda rx: bool(re.search(rx, html, re.I))
    pts.append((1.0 if has(r"true\s*anomaly|\bnu\b|ν") else 0.0, 0.05, "true anomaly readout"))
    pts.append((1.0 if has(r"radius|\br\s*[:=]") else 0.0, 0.05, "radius readout"))
    pts.append((1.0 if has(r"speed|velocit|vis-?viva") else 0.0, 0.05, "speed readout"))

    got = sum(s * w for s, w, _ in pts)
    tot = sum(w for _, w, _ in pts)
    fails = [lbl for s, _, lbl in pts if s < 1.0]
    score = got / tot
    if not visible:
        score = min(score, 0.3)
        fails.insert(0, f"orbit only {frac*100:.1f}% of canvas")
    if js_err:
        score = min(score, 0.7)
        fails.append(f"{len(js_err)} runtime JS errors")
    note = "; ".join(fails) if fails else "all physics checks pass"
    a.pop("pts", None)
    return score, note[:220], {"analysis": {k: v for k, v in a.items()}, "checks": pts}


def _kepler_static(r):
    """Fallback: grade structure only. Used when the JS engine is unavailable.

    Catches the three static failure modes seen in practice, all of which a keyword grep
    happily passes:
      1. ellipse() drawn with the SAME radius twice -> it's a circle.
      2. No semi-minor axis b = a*sqrt(1-e^2) computed anywhere.
      3. The focus offset computed but not scaled into pixels, so the star
         lands on the ellipse centre.
    """
    t = extract_html(r.get("content") or "")
    checks, notes = [], []

    def add(ok, w, label):
        checks.append((1.0 if ok else 0.0, w))
        if not ok:
            notes.append(label)

    add(re.search(r"<canvas", t, re.I), 1, "no <canvas>")
    add(re.search(r"requestAnimationFrame", t), 1, "no animation loop")

    # --- Kepler's equation, solved iteratively (accept several spellings) ---
    resid = re.search(r"-\s*e[a-z_]*\s*\*\s*(Math\.)?sin\s*\(\s*E|E\s*-\s*e[a-z_]*\s*\*", t, re.I)
    deriv = re.search(r"1(\.0)?\s*-\s*e[a-z_]*\s*\*\s*(Math\.)?cos", t, re.I)
    loop = re.search(r"\b(for|while)\b", t)
    add(resid, 3, "does not form the Kepler residual E - e*sin(E) - M")
    add(deriv and loop, 3, "no Newton-Raphson iteration (needs 1 - e*cos(E) inside a loop)")

    # --- KEPLER 1: is it actually an ellipse? ------------------------------
    semi_minor = re.search(
        r"(sqrt|\*\*\s*0?\.5)\s*\(?\s*1(\.0)?\s*-\s*(e[a-z_]*\s*\*\s*e[a-z_]*|e[a-z_]*\s*\*\*\s*2)", t, re.I)
    add(semi_minor, 3, "no semi-minor axis b = a*sqrt(1-e^2) -> orbit is not an ellipse")

    circle = False
    for m in re.finditer(r"\.ellipse\s*\(([^)]*)\)", t):
        parts = [p.strip() for p in m.group(1).split(",")]
        if len(parts) >= 4 and parts[2] == parts[3]:
            circle = True                    # same radius twice
    add(not circle, 3, "ellipse() called with identical radii -> renders a CIRCLE")

    # --- KEPLER 1: star at a focus, offset actually applied in pixels ------
    foc = re.search(r"[a-z_]*\s*=\s*[a-z_]*\s*\*\s*e[a-z_]*\b|a\s*\*\s*e\b|focus", t, re.I)
    add(foc, 2, "no focus offset (a*e) computed")
    # the offset must reach the draw call in the same units as the orbit
    scaled = re.search(r"(star|sun|focus)[A-Za-z_]*\s*=\s*[^;\n]*\*\s*(SCALE|scale|PX|px|zoom)", t) \
        or re.search(r"\(\s*[a-z_]*\s*\*\s*e[a-z_]*\s*\)\s*\*\s*(SCALE|scale)", t, re.I) \
        or re.search(r"(SCALE|scale)\s*\*\s*[a-z_]*\s*\*\s*e\b", t, re.I)
    has_scale = re.search(r"\bSCALE\b|\bscale\b", t)
    add(scaled or not has_scale, 2,
        "focus offset never multiplied by the pixel scale -> star sits at the ellipse centre")

    # --- readouts ----------------------------------------------------------
    add(re.search(r"true\s*anomaly|\bnu\b|ν", t, re.I), 1, "no true anomaly readout")
    add(re.search(r"radius|\br\s*[:=]", t, re.I), 1, "no radius readout")
    add(re.search(r"speed|velocit|vis-?viva|2\s*/\s*r", t, re.I), 1, "no speed readout")

    got = sum(s * w for s, w in checks)
    tot = sum(w for _, w in checks)
    note = "; ".join(notes) if notes else "geometry checks pass - still RENDER IT to confirm"
    return got / tot, note[:200]


