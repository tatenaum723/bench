"""Benchmark suite for evaluating local coding models on this box.

Design principle: every task is a DISCRIMINATOR. Tasks that every model passes
tell you nothing, so each one below has a specific failure mode that weaker
models actually hit, documented inline.

Scoring is deliberately split:
  - auto:   machine-checkable (exact answers, tool-call shape, structural greps)
  - visual: must be looked at (the Kepler render) -- the runner saves an artifact
Anything the runner cannot check it marks NEEDS_REVIEW rather than guessing.
"""
import json
import re

# ---------------------------------------------------------------------------
# Tool definitions shared by the tool-calling tasks
# ---------------------------------------------------------------------------
T_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}
T_SEARCH = {
    "type": "function",
    "function": {
        "name": "search_files",
        "description": "Search for a regex pattern across files in a directory",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "case_sensitive": {"type": "boolean"},
            },
            "required": ["pattern", "path"],
        },
    },
}
T_RUN = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": "Execute a shell command and return stdout",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}
T_CVE = {
    "type": "function",
    "function": {
        "name": "lookup_cve",
        "description": "Look up details for a CVE identifier",
        "parameters": {
            "type": "object",
            "properties": {"cve_id": {"type": "string",
                                      "description": "e.g. CVE-2021-44228"}},
            "required": ["cve_id"],
        },
    },
}


# ---------------------------------------------------------------------------
# Scorers. Each returns (score 0..1, note). score None => needs human review.
# ---------------------------------------------------------------------------
def _txt(r):
    return (r.get("content") or "") + " " + (r.get("reasoning_content") or "")


def answer_contains(*accepted, reject=()):
    """Final answer must contain one of `accepted` and none of `reject`.

    Checks the tail of the response so a model that explores wrong values
    mid-reasoning but lands correctly still passes.
    """
    def f(r, _):
        t = _txt(r).lower()
        tail = t[-400:]
        if any(x.lower() in tail for x in reject):
            return 0.0, f"tail contains rejected token"
        hit = [x for x in accepted if x.lower() in tail]
        if hit:
            return 1.0, f"found {hit[0]!r}"
        anywhere = [x for x in accepted if x.lower() in t]
        return (0.4, f"{anywhere[0]!r} present but not in final answer") if anywhere \
            else (0.0, f"expected one of {list(accepted)}")
    return f


def calls_tool(name, **arg_checks):
    """Must emit exactly one tool_call for `name`, with matching arguments."""
    def f(r, _):
        tcs = r.get("tool_calls") or []
        if not tcs:
            return 0.0, "no tool_calls emitted"
        names = [c["function"]["name"] for c in tcs]
        if name not in names:
            return 0.0, f"called {names}, expected {name}"
        call = next(c for c in tcs if c["function"]["name"] == name)
        try:
            args = json.loads(call["function"]["arguments"])
        except Exception as e:
            return 0.3, f"correct tool but unparseable arguments: {e}"
        for k, want in arg_checks.items():
            got = args.get(k)
            if got is None:
                return 0.5, f"missing argument {k!r}"
            if isinstance(want, str) and want.lower() not in str(got).lower():
                return 0.6, f"{k}={got!r}, expected ~{want!r}"
        extra = " (+extra calls)" if len(tcs) > 1 else ""
        return 1.0, f"correct call{extra}"
    return f


def calls_no_tool(r, _):
    """Restraint test: the question needs no tool. Calling one is the failure."""
    tcs = r.get("tool_calls") or []
    if tcs:
        return 0.0, f"called {[c['function']['name'] for c in tcs]} unnecessarily"
    return (1.0, "correctly answered directly") if (r.get("content") or "").strip() \
        else (0.0, "no tool call but also no answer")


def greps(patterns, weights=None, note=""):
    """Structural check over generated code. Partial credit per pattern."""
    weights = weights or [1.0] * len(patterns)
    def f(r, _):
        t = _txt(r)
        tot = sum(weights)
        got, missing = 0.0, []
        for (label, rx), w in zip(patterns, weights):
            if re.search(rx, t, re.I | re.S):
                got += w
            else:
                missing.append(label)
        return got / tot, (f"missing: {', '.join(missing)}" if missing else note or "all checks hit")
    return f


def kepler_scorer(r, _):
    """Grade the Kepler task by EXECUTING it, falling back to static checks.

    Static analysis proved insufficient: it scored 83% on output whose script
    did not parse at all (`const mu` declared three times) and which, had it
    parsed, would have drawn a 1-pixel circle with the star at its centre.
    canvas_probe runs the model's own JS against a recording canvas and
    measures the resulting geometry. Verified against a known-good
    implementation: e=0.6000, star-to-focus 0.00 px, Kepler-2 CV 0.00003.
    """
    try:
        from canvas_probe import run_canvas, analyse_orbit
    except Exception:
        return _kepler_static(r)          # py_mini_racer unavailable

    html = (r.get("content") or "")
    m = re.search(r"<!DOCTYPE html.*?</html>", html, re.S | re.I)
    if not m:
        m2 = re.search(r"```(?:html)?\s*(.*?)```", html, re.S)
        html = m2.group(1) if m2 else html
    else:
        html = m.group(0)

    log, canv, err = run_canvas(html, frames=400)
    if err:
        # Does not even run. This is a hard zero on the physics; award only a
        # token amount for having produced plausible-looking structure.
        stat, _ = _kepler_static(r)
        return min(0.15, stat * 0.15), f"CODE DOES NOT RUN: {err[:120]}"

    js_err = [e for e in log if e["op"] == "__error"]
    a = analyse_orbit(log, canv)
    if a.get("error"):
        return 0.15, f"ran but drew nothing usable: {a['error']}"

    pts = []          # (score, weight, label)
    cw, ch = a["canvas"]
    frac = a["planet_spread_px"] / max(cw, ch) if max(cw, ch) else 0
    pts.append((1.0 if frac > 0.15 else 0.0, 3,
                f"orbit is {frac*100:.1f}% of canvas (needs >15%)"))

    f = a.get("fit")
    if not f:
        pts.append((0.0, 4, "planet path is not an ellipse (conic fit failed)"))
        pts.append((0.0, 3, "no ellipse -> cannot locate focus"))
        pts.append((0.0, 3, "no ellipse -> cannot check Kepler 2"))
    else:
        de = abs(f["e"] - 0.6)
        pts.append((1.0 if de < 0.05 else (0.5 if de < 0.15 else 0.0), 4,
                    f"eccentricity {f['e']:.3f} (target 0.600)"))
        fe = a.get("star_focus_err_px")
        ce = a.get("star_centre_err_px")
        if fe is None:
            pts.append((0.0, 3, "no star drawn"))
        else:
            tol = max(6.0, 0.06 * f["a"])
            ok = fe < tol and (ce or 0) > tol
            pts.append((1.0 if ok else 0.0, 3,
                        f"star {fe:.1f}px from focus, {ce:.1f}px from centre"))
        cv = a.get("area_rate_cv")
        if cv is None:
            pts.append((0.0, 3, "could not measure swept area"))
        else:
            pts.append((1.0 if cv < 0.10 else (0.5 if cv < 0.30 else 0.0), 3,
                        f"Kepler-2 area CV {cv:.3f} (needs <0.10)"))

    has = lambda rx: bool(re.search(rx, html, re.I))
    pts.append((1.0 if has(r"true\s*anomaly|\bnu\b|ν") else 0.0, 1, "true anomaly readout"))
    pts.append((1.0 if has(r"radius|\br\s*[:=]") else 0.0, 1, "radius readout"))
    pts.append((1.0 if has(r"speed|velocit|vis-?viva") else 0.0, 1, "speed readout"))
    if js_err:
        pts.append((0.0, 2, f"{len(js_err)} runtime JS errors"))

    got = sum(s * w for s, w, _ in pts)
    tot = sum(w for _, w, _ in pts)
    fails = [lbl for s, _, lbl in pts if s < 1.0]
    note = "; ".join(fails) if fails else "all physics checks pass"
    return got / tot, note[:220]


def _kepler_static(r):
    """Fallback: grade structure only. Used when the JS engine is unavailable.

    Catches the three static failure modes seen in practice, all of which a keyword grep
    happily passes:
      1. ellipse() drawn with the SAME radius twice -> it's a circle.
      2. No semi-minor axis b = a*sqrt(1-e^2) computed anywhere.
      3. The focus offset computed but not scaled into pixels, so the star
         lands on the ellipse centre.
    """
    t = _txt(r)
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


# ---------------------------------------------------------------------------
# THE SUITE
# ---------------------------------------------------------------------------
TASKS = [

    # ======================= FLAGSHIP: KEPLER ==============================
    # Why this discriminates: almost every weak model animates a planet at
    # CONSTANT angular velocity around an ELLIPSE CENTRE. Both are wrong.
    # Kepler 1 puts the star at a FOCUS (offset by a*e), and Kepler 2 requires
    # solving M = E - e*sin(E) for E, which has no closed form and needs
    # Newton-Raphson. A model that produces uniform circular motion has failed
    # the physics even if the HTML is pretty.
    dict(
        id="kepler_orbit",
        category="coding",
        weight=3.0,
        max_tokens=4096,
        artifact="html",
        prompt="""Write a single self-contained HTML file that animates a Keplerian two-body orbit on a <canvas>.

Requirements:
1. The planet must follow a true ellipse with eccentricity e = 0.6, and the star must sit at a FOCUS of that ellipse, not at its centre.
2. Motion must obey Kepler's second law (equal areas in equal times) - the planet visibly speeds up at periapsis and slows at apoapsis. Do this properly: advance the mean anomaly M linearly in time, then solve Kepler's equation M = E - e*sin(E) for the eccentric anomaly E using Newton-Raphson, then compute the position from E. Do not fake it with a velocity heuristic.
3. Draw the full orbital path, the star, and the planet. Trace the swept area or a trail so the second law is visible.
4. Show live readouts of true anomaly, current radius, and orbital speed.
5. No external libraries, no network requests. Everything inline in one file.

Output only the HTML file contents, starting with <!DOCTYPE html>. No commentary.""",
        # SCORER NOTE: v1 of this scorer graded vocabulary and gave a model 83%
        # for output that renders a CIRCLE with the star at its CENTRE -- the
        # exact failure this task exists to catch. It called ellipse() with the
        # same radius twice, never computed a semi-minor axis, and forgot to
        # scale the focus offset. It also false-negatived a correct
        # Newton-Raphson loop. So these checks now test GEOMETRY, not words.
        scorer=kepler_scorer,
    ),

    # ======================= REASONING =====================================
    # Verifiable integer answers. No partial credit for "approximately".
    dict(
        id="reason_trains",
        category="reasoning", weight=1.0, max_tokens=1200,
        prompt="""Town A and Town B are 300 km apart on a straight road.
A cyclist leaves A toward B at 09:00 traveling 20 km/h.
A motorcyclist leaves B toward A at 11:00 traveling 60 km/h.
At what clock time do they meet? Give the answer as HH:MM on a 24-hour clock.""",
        # At 11:00 cyclist has covered 40 km; 260 km remain; closing speed 80 km/h;
        # 260/80 = 3.25 h = 3h15m after 11:00 -> 14:15.
        scorer=answer_contains("14:15", "2:15 pm", "2:15pm"),
    ),
    dict(
        id="reason_trace",
        category="reasoning", weight=1.0, max_tokens=1200,
        prompt="""What exactly does this Python program print? Give only the printed output.

def f(xs):
    out = []
    for i, x in enumerate(xs):
        if x % 3 == 0:
            out.append(x * 2)
        elif x % 3 == 1:
            out.insert(0, x)
        else:
            out = out[::-1]
    return out

print(f([1, 2, 3, 4, 5, 6, 7]))""",
        # 1:%3=1 -> insert(0,1) => [1]
        # 2:%3=2 -> reverse     => [1]
        # 3:%3=0 -> append 6    => [1,6]
        # 4:%3=1 -> insert(0,4) => [4,1,6]
        # 5:%3=2 -> reverse     => [6,1,4]
        # 6:%3=0 -> append 12   => [6,1,4,12]
        # 7:%3=1 -> insert(0,7) => [7,6,1,4,12]
        scorer=answer_contains("[7, 6, 1, 4, 12]", "[7,6,1,4,12]"),
    ),
    dict(
        id="reason_constraint",
        category="reasoning", weight=1.0, max_tokens=1600,
        prompt="""Four servers - alpha, beta, gamma, delta - each run exactly one service:
cache, database, queue, web. Determine the assignment from these facts:

1. alpha does not run the database or the web server.
2. The queue runs on a server whose name comes alphabetically after gamma's.
3. beta runs neither the cache nor the queue.
4. gamma does not run the web server.

State the service for each server.""",
        # 2 => queue is on delta (only name after gamma). 3 => beta in {db, web}.
        # 1 => alpha in {cache, queue}; queue is delta's, so alpha = cache.
        # 4 => gamma != web => gamma = database => beta = web.
        scorer=greps([
            ("alpha=cache", r"alpha\W{0,12}cache"),
            ("beta=web", r"beta\W{0,12}web"),
            ("gamma=database", r"gamma\W{0,12}(database|db)"),
            ("delta=queue", r"delta\W{0,12}queue"),
        ]),
    ),
    dict(
        id="reason_longctx",
        category="reasoning", weight=1.5, max_tokens=900,
        needle=True,   # runner injects filler around a planted fact
        prompt=None,   # built by the runner
        scorer=answer_contains("PLUTONIUM-7734"),
    ),

    # ======================= TOOL CALLING ==================================
    dict(
        id="tool_basic",
        category="tools", weight=1.0, max_tokens=600, tools=[T_WEATHER],
        prompt="What's the weather in Reykjavik right now? Use the tool.",
        scorer=calls_tool("get_weather", city="Reykjavik"),
    ),
    dict(
        id="tool_select",
        category="tools", weight=1.5, max_tokens=600,
        tools=[T_WEATHER, T_SEARCH, T_RUN, T_CVE],
        prompt="Find every place the string TODO appears under /home/tate/projects. Use the appropriate tool.",
        scorer=calls_tool("search_files", pattern="TODO", path="/home/tate/projects"),
    ),
    dict(
        id="tool_restraint",
        category="tools", weight=1.5, max_tokens=600,
        tools=[T_WEATHER, T_SEARCH, T_RUN, T_CVE],
        # Discriminator: weak models reflexively call a tool because tools exist.
        prompt="What does the acronym HTTP stand for? Answer from your own knowledge.",
        scorer=calls_no_tool,
    ),
    dict(
        id="tool_typed_args",
        category="tools", weight=1.0, max_tokens=600, tools=[T_SEARCH],
        prompt="Search /var/log for the case-sensitive regex ^Failed password, using the tool.",
        scorer=calls_tool("search_files", pattern="Failed password",
                          path="/var/log", case_sensitive="true"),
    ),

    # ======================= CYBER SECURITY ================================
    # Defensive: finding and fixing flaws in code, and reading logs.
    dict(
        id="sec_sqli",
        category="security", weight=1.5, max_tokens=1400,
        prompt="""Review this Flask endpoint. Identify every security vulnerability, name the class of each, and give a corrected version.

@app.route('/user')
def user():
    uid = request.args.get('id')
    conn = sqlite3.connect('app.db')
    row = conn.execute("SELECT name, email FROM users WHERE id = " + uid).fetchone()
    return f"<h1>Welcome {row[0]}</h1><p>{row[1]}</p>" """,
        scorer=greps([
            ("names SQL injection", r"sql\s*injection|sqli"),
            ("names XSS", r"xss|cross[- ]site scripting|html escap"),
            ("parameterized fix", r"\?\s*\)|execute\([^)]*,\s*\(|placeholder|parameteri[sz]ed|prepared statement"),
            ("escaping/templating fix", r"escape\(|render_template|markupsafe|jinja"),
        ], weights=[2, 2, 2, 1]),
    ),
    dict(
        id="sec_cmdinj",
        category="security", weight=1.5, max_tokens=1400,
        prompt="""This helper is used by an internal admin tool. Identify the vulnerability, explain how it could be exploited, and rewrite it safely.

import os
def ping_host(hostname):
    return os.popen("ping -c 1 " + hostname).read()""",
        scorer=greps([
            ("names command injection", r"command injection|shell injection|os command"),
            ("shows shell metachar risk", r"[;&|`]|\$\(|semicolon|pipe|backtick"),
            ("subprocess list form", r"subprocess\.(run|check_output|Popen)"),
            ("no shell=True / validation", r"shell\s*=\s*False|shell=True|validat|allowlist|whitelist|ipaddress"),
        ], weights=[2, 1, 2, 1]),
    ),
    dict(
        id="sec_crypto",
        category="security", weight=1.5, max_tokens=1400,
        prompt="""Audit this password-handling code. List what is wrong and write a correct replacement.

import hashlib
def hash_password(pw):
    return hashlib.md5(pw.encode()).hexdigest()

def check_password(pw, stored):
    return hash_password(pw) == stored""",
        scorer=greps([
            ("MD5 broken/unsuitable", r"md5.{0,80}(broken|collision|fast|unsuitable|insecure|weak)|not.{0,20}password hashing"),
            ("no salt", r"salt"),
            ("recommends KDF", r"bcrypt|argon2|scrypt|pbkdf2"),
            ("timing-safe compare", r"compare_digest|constant[- ]time|timing attack"),
        ], weights=[2, 2, 2, 1]),
    ),
    dict(
        id="sec_logs",
        category="security", weight=1.5, max_tokens=1400,
        prompt="""These lines are from /var/log/auth.log on an internet-facing host. What is happening, and what are the two most important things to do about it?

Sep 13 03:11:02 web1 sshd[2201]: Failed password for invalid user admin from 203.0.113.44 port 51122 ssh2
Sep 13 03:11:03 web1 sshd[2203]: Failed password for invalid user root from 203.0.113.44 port 51130 ssh2
Sep 13 03:11:04 web1 sshd[2205]: Failed password for invalid user oracle from 203.0.113.44 port 51138 ssh2
Sep 13 03:11:05 web1 sshd[2207]: Failed password for invalid user postgres from 203.0.113.44 port 51146 ssh2
Sep 13 03:11:07 web1 sshd[2210]: Failed password for invalid user test from 203.0.113.44 port 51160 ssh2
Sep 13 03:12:41 web1 sshd[2298]: Accepted password for deploy from 203.0.113.44 port 51844 ssh2
Sep 13 03:12:58 web1 sudo:   deploy : TTY=pts/0 ; PWD=/home/deploy ; USER=root ; COMMAND=/usr/bin/apt-get install -y netcat""",
        scorer=greps([
            ("identifies brute force", r"brute[- ]?forc|password spray|credential stuffing|dictionary attack"),
            ("NOTICES THE SUCCESS", r"succeed|success|accepted|compromis|breach|got in|gained access"),
            ("flags sudo/netcat escalation", r"netcat|nc\b|privilege escalation|sudo.{0,40}root|backdoor|reverse shell"),
            ("recommends key-only/disable pw", r"key[- ]based|publickey|PasswordAuthentication\s*no|disable password"),
            ("recommends isolate/rotate", r"isolat|disconnect|rotate|revoke|reset.{0,20}credential|block.{0,20}IP|fail2ban"),
        ], weights=[1, 3, 3, 1, 1],
           note="the real discriminator is whether it notices the brute force SUCCEEDED at 03:12:41"),
    ),
    dict(
        id="sec_secure_impl",
        category="security", weight=1.0, max_tokens=1600,
        prompt="""Write a Python function `safe_extract(archive_path, dest_dir)` that extracts a user-supplied tar archive safely. It must defend against path traversal (the "Zip Slip"/tarslip class), absolute paths, and symlink escapes. Explain each defence in a comment.""",
        scorer=greps([
            ("resolves real paths", r"realpath|resolve\(\)|abspath|commonpath|is_relative_to"),
            ("rejects traversal", r"\.\.|traversal|startswith"),
            ("handles symlinks", r"issym|islnk|symlink"),
            ("handles absolute paths", r"isabs|startswith\(\s*['\"]/|absolute"),
        ], weights=[2, 2, 2, 1]),
    ),
]


def build_needle_prompt(filler_tokens=12000):
    """Long-context retrieval: plant a fact in the middle of plausible filler.

    The installed model failed exactly this shape (answered 70 where the true
    count was ~900), so it is worth measuring directly.
    """
    block = ("The deployment pipeline validates each artifact against the manifest "
             "before promotion. Checksums are recomputed at every stage boundary. "
             "Rollback is automatic when a health probe fails twice consecutively.\n")
    per = 40                                   # ~tokens per block, rough
    n = max(1, filler_tokens // per // 2)
    needle = ("\nNOTE: the emergency reactor override passphrase is PLUTONIUM-7734. "
              "Keep it with the runbook.\n")
    return ("Read the following operations runbook carefully.\n\n"
            + block * n + needle + block * n
            + "\n\nWhat is the emergency reactor override passphrase? "
              "Answer with the passphrase only.")
