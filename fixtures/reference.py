"""Known-good and known-bad responses for every task, used ONLY by selftest.py
to prove each scorer discriminates. These are not a baseline and are never
compared against models.

Shape: FIXTURES[task_id] = {"good": [response...], "bad": [(label, response, max_score)...]}
A response is the dict a model call produces ({"content":..., "tool_calls":[...]}).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def R(content="", tool_calls=None, reasoning=""):
    return {"content": content, "reasoning_content": reasoning, "tool_calls": tool_calls or []}


def call(name, **args):
    return {"id": "x", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def py(src):
    return R("Here is the fix.\n\n```python\n" + src.strip() + "\n```\n")


# ---------------------------------------------------------------------------
# Kepler pages
# ---------------------------------------------------------------------------
KEPLER_GOOD = open(os.path.join(HERE, "kepler_good.html")).read()

_KEPLER_TMPL = """<!DOCTYPE html><html><body><canvas id="c" width="800" height="600"></canvas>
<div id="hud">true anomaly <span id="nu"></span> radius <span id="r"></span> speed <span id="v"></span></div>
<script>
const cv=document.getElementById('c'), g=cv.getContext('2d');
const a=220, e=0.6, b=a*Math.sqrt(1-e*e), c=a*e, CX=400, CY=300;
let t=0;
function frame(){
  t+=0.016;
  g.clearRect(0,0,800,600);
  %(pos)s
  g.beginPath(); g.arc(%(star)s, 10, 0, 2*Math.PI); g.fill();
  g.beginPath(); g.arc(px, py, 5, 0, 2*Math.PI); g.fill();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script></body></html>"""

KEPLER_CIRCLE = _KEPLER_TMPL % {
    "pos": "const th=t; const px=CX+a*Math.cos(th), py=CY+a*Math.sin(th);",
    "star": "CX, CY"}
KEPLER_CENTRE_STAR = _KEPLER_TMPL % {
    "pos": "const E=t; const px=CX+a*Math.cos(E), py=CY+b*Math.sin(E);",   # ellipse, star at centre
    "star": "CX, CY"}
KEPLER_UNIFORM_E = _KEPLER_TMPL % {
    # correct ellipse and focus, but E advanced linearly -> violates Kepler 2
    "pos": "const E=t; const px=CX+a*Math.cos(E), py=CY+b*Math.sin(E);",
    "star": "CX+c, CY"}
# Correct physics drawn in AU via translate/scale (a transform-aware grader must pass it).
KEPLER_TRANSFORMED = """<!DOCTYPE html><html><body><canvas id="c" width="900" height="600"></canvas>
<div>true anomaly <span id="nu"></span> radius <span id="r"></span> speed <span id="v"></span></div>
<script>
const g = document.getElementById('c').getContext('2d');
const a = 1.0, e = 0.6, b = a * Math.sqrt(1 - e * e), SCALE = 250;
let M = 0;
function solveE(M) { let E = M; for (let i = 0; i < 20; i++) { E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E)); } return E; }
function frame() {
  M += 0.03;
  const E = solveE(M);
  const x = a * (Math.cos(E) - e), y = b * Math.sin(E);   // star at origin = focus
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, 900, 600);
  g.save();
  g.translate(560, 300);
  g.scale(SCALE, -SCALE);
  g.beginPath(); g.arc(0, 0, 0.04, 0, 2 * Math.PI); g.fill();
  g.beginPath(); g.arc(x, y, 0.02, 0, 2 * Math.PI); g.fill();
  g.restore();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script></body></html>"""
# Correct physics timed with Date.now() (the grader's clock must cover Date too).
KEPLER_DATENOW = KEPLER_TRANSFORMED.replace("let M = 0;", "const t0 = Date.now(); let M = 0;").replace(
    "  M += 0.03;", "  M = (Date.now() - t0) / 1000 * 1.9;")
assert KEPLER_DATENOW != KEPLER_TRANSFORMED
KEPLER_SYNTAX = KEPLER_GOOD.replace("const b = a * Math.sqrt(1 - e * e);",
                                    "const b = a * Math.sqrt(1 - e * e);\n  const a = 5;")

# ---------------------------------------------------------------------------
# Coding
# ---------------------------------------------------------------------------
DURATION_GOOD = r'''
import re
_RE = re.compile(r"(?:([0-9]+)d)?(?:([0-9]+)h)?(?:([0-9]+)m)?(?:([0-9]+)s)?")
def parse_duration(s):
    if not isinstance(s, str):
        raise TypeError("expected str")
    m = _RE.fullmatch(s)
    if not s or m is None or not any(m.groups()):
        raise ValueError(f"invalid duration: {s!r}")
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se
'''
DURATION_BAD_UNICODE = r'''
import re
def parse_duration(s):
    if not isinstance(s, str):
        raise TypeError("expected str")
    m = re.match(r"^(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$", s)
    if not s or not m or not any(m.groups()):
        raise ValueError(s)
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se
'''  # \d accepts full-width digits and $ matches before a trailing newline
DURATION_BAD_LOOSE = r'''
import re
def parse_duration(s):
    total = 0
    for n, u in re.findall(r"(\d+)([dhms])", str(s)):
        total += int(n) * {"d": 86400, "h": 3600, "m": 60, "s": 1}[u]
    return total
'''

LRU_GOOD = r'''
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        self.capacity = capacity
        self._d = OrderedDict()
    def get(self, key, default=None):
        if key not in self._d:
            return default
        self._d.move_to_end(key)
        return self._d[key]
    def put(self, key, value):
        if self.capacity == 0:
            return
        if key in self._d:
            self._d.move_to_end(key)
        self._d[key] = value
        if len(self._d) > self.capacity:
            self._d.popitem(last=False)
    def __len__(self):
        return len(self._d)
    def __contains__(self, key):
        return key in self._d
'''
LRU_BAD_LIST = r'''
class LRUCache:
    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError
        self.capacity = capacity
        self.keys = []
        self.vals = {}
    def get(self, key, default=None):
        if key not in self.vals:
            return default
        self.keys.remove(key); self.keys.append(key)
        return self.vals[key]
    def put(self, key, value):
        if self.capacity == 0:
            return
        if key in self.vals:
            self.keys.remove(key)
        elif len(self.keys) >= self.capacity:
            del self.vals[self.keys.pop(0)]
        self.keys.append(key); self.vals[key] = value
    def __len__(self):
        return len(self.vals)
    def __contains__(self, key):
        return key in self.vals
'''

MERGE_GOOD = r'''
def merge_intervals(intervals):
    result = []
    for start, end in sorted((tuple(i) for i in intervals)):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result
'''
MERGE_BAD_PARTIAL = r'''
def merge_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start < result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result
'''

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
SQLI_GOOD = r'''
from flask import Flask, request, abort
from markupsafe import escape
import sqlite3

app = Flask(__name__)

@app.route('/user')
def user():
    uid = request.args.get('id', type=int)
    if uid is None:
        abort(400)
    conn = sqlite3.connect('app.db')
    try:
        row = conn.execute("SELECT name, email FROM users WHERE id = ?", (uid,)).fetchone()
    finally:
        conn.close()
    if row is None:
        abort(404)
    return f"<h1>Welcome {escape(row[0])}</h1><p>{escape(row[1])}</p>"
'''
SQLI_BAD_ONLY_SQL = r'''
from flask import Flask, request
import sqlite3
app = Flask(__name__)
@app.route('/user')
def user():
    uid = request.args.get('id')
    conn = sqlite3.connect('app.db')
    row = conn.execute("SELECT name, email FROM users WHERE id = ?", (uid,)).fetchone()
    return f"<h1>Welcome {row[0]}</h1><p>{row[1]}</p>"
'''

CMDINJ_GOOD = r'''
import ipaddress, re, subprocess
_HOST = re.compile(r"^(?=.{1,253}$)[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$")
def ping_host(hostname):
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        if not _HOST.fullmatch(hostname):
            raise ValueError("invalid hostname")
    r = subprocess.run(["ping", "-c", "1", "--", hostname], capture_output=True, text=True, timeout=10)
    return r.stdout
'''
CMDINJ_BAD_SHELL = r'''
import subprocess, shlex
def ping_host(hostname):
    return subprocess.run("ping -c 1 " + hostname, shell=True, capture_output=True, text=True).stdout
'''
CMDINJ_BAD_NO_OPTION_GUARD = r'''
import subprocess
def ping_host(hostname):
    return subprocess.run(["ping", "-c", "1", hostname], capture_output=True, text=True).stdout
'''

CRYPTO_GOOD = r'''
import base64, hashlib, hmac, os
_ITER = 200_000
def hash_password(pw):
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, _ITER)
    return f"pbkdf2_sha256${_ITER}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"
def check_password(pw, stored):
    try:
        algo, it, salt, dk = stored.split("$")
        it = int(it); salt = base64.b64decode(salt); dk = base64.b64decode(dk)
    except Exception:
        return False
    return hmac.compare_digest(hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, it), dk)
'''
CRYPTO_BAD_FAST_SALT = r'''
import hashlib, os, hmac
def hash_password(pw):
    salt = os.urandom(8).hex()
    return salt + ":" + hashlib.sha256((salt + pw).encode()).hexdigest()
def check_password(pw, stored):
    salt, h = stored.split(":")
    return hmac.compare_digest(hashlib.sha256((salt + pw).encode()).hexdigest(), h)
'''

TARSLIP_GOOD = r'''
import os, tarfile
def safe_extract(archive_path, dest_dir):
    dest = os.path.realpath(dest_dir)
    with tarfile.open(archive_path) as tar:
        for m in tar.getmembers():
            # absolute paths and .. segments: resolve the final path and require it inside dest
            target = os.path.realpath(os.path.join(dest, m.name))
            if os.path.commonpath([dest, target]) != dest:
                raise ValueError(f"blocked path traversal: {m.name}")
            # links: refuse anything whose target resolves outside dest
            if m.issym() or m.islnk():
                base = dest if m.islnk() else os.path.dirname(target)
                link = os.path.realpath(os.path.join(base, m.linkname))
                if os.path.commonpath([dest, link]) != dest:
                    raise ValueError(f"blocked link escape: {m.name} -> {m.linkname}")
        # the data filter additionally rejects device files and strips unsafe modes
        tar.extractall(dest, filter="data")
'''
TARSLIP_BAD_NAIVE = r'''
import os, tarfile
def safe_extract(archive_path, dest_dir):
    with tarfile.open(archive_path) as tar:
        for m in tar.getmembers():
            if m.name.startswith("..") or m.name.startswith("/"):
                raise ValueError("bad path")
        tar.extractall(dest_dir, filter="fully_trusted")
'''

# ---------------------------------------------------------------------------
LOGS_GOOD = R("""This is a brute-force / password-spraying attack from 203.0.113.44 against common usernames. Critically, at 03:12:41 the same IP got an Accepted password for the deploy account - the attack succeeded - and 17 seconds later deploy used sudo to install netcat, a classic post-exploitation step for a reverse shell.

1. Treat web1 as compromised: isolate it from the network, preserve logs for forensics and begin incident response.
2. Lock and rotate the deploy credentials (and any keys/secrets on the host), then harden SSH: PasswordAuthentication no, key-based auth only, block 203.0.113.44 and add fail2ban.""")
LOGS_BAD_MISSES_SUCCESS = R("""This shows a brute-force attack from 203.0.113.44 trying usernames like admin, root and oracle. All attempts failed because the users are invalid.

1. Install fail2ban to block repeated failures.
2. Set PasswordAuthentication no and use key-based authentication.""")

A = "\n\nANSWER: "
FIXTURES = {
    "kepler_orbit": {"good": [R(KEPLER_GOOD), R(KEPLER_TRANSFORMED), R(KEPLER_DATENOW)],
                     "bad": [("uniform circle, star at centre", R(KEPLER_CIRCLE), 0.2),
                             ("ellipse with star at centre", R(KEPLER_CENTRE_STAR), 0.45),
                             ("star at focus but uniform E (breaks Kepler 2)", R(KEPLER_UNIFORM_E), 0.7),
                             ("script does not parse", R(KEPLER_SYNTAX), 0.15)]},
    "code_duration": {"good": [py(DURATION_GOOD)],
                      "bad": [("\\d and $ accept unicode digits / trailing newline", py(DURATION_BAD_UNICODE), 0.97),
                              ("findall, no validation", py(DURATION_BAD_LOOSE), 0.5),
                              ("no code block", R("parse it with a regex"), 0.0)]},
    "code_lru": {"good": [py(LRU_GOOD)],
                 "bad": [("list-based recency, O(n)", py(LRU_BAD_LIST), 0.9)]},
    "code_merge_fix": {"good": [py(MERGE_GOOD)],
                       "bad": [("sorted but <, lists, mutates", py(MERGE_BAD_PARTIAL), 0.6)]},
    "reason_trains": {"good": [R("...so they meet at 14:15." + A + "14:15")],
                      "bad": [("wrong", R("ANSWER: 14:00"), 0.0),
                              ("answer only in reasoning", R("I am not sure.", reasoning="14:15"), 0.0),
                              ("right value, no ANSWER line", R("They meet at 14:15."), 0.5)]},
    "reason_trace": {"good": [R("Tracing..." + A + "[7, 6, 1, 4, 12]"), R(A + "`[7,6,1,4,12]`")],
                     "bad": [("wrong", R(A + "[7, 1, 6, 4, 12]"), 0.0)]},
    "reason_precedence": {"good": [R(A + "255")], "bad": [("left-to-right", R(A + "51"), 0.0)]},
    "reason_bayes": {"good": [R(A + "1/11")], "bad": [("naive", R(A + "1/6"), 0.0)]},
    "reason_schedule": {"good": [R(A + "Tuesday 12:00"), R(A + "Tue 12:00")],
                        "bad": [("off by one interval", R(A + "Tuesday 13:30"), 0.0)]},
    "reason_constraint": {"good": [R(A + "alpha=cache, beta=web, gamma=database, delta=queue"),
                                   R(A + "delta=queue, alpha=cache, gamma=database, beta=web")],
                          "bad": [("swapped", R(A + "alpha=cache, beta=database, gamma=web, delta=queue"), 0.0)]},
    "reason_longctx": {"good": [R(A + "HX-4471")],
                       "bad": [("retired code", R(A + "HX-2210"), 0.0),
                               ("archived owner", R(A + "KT-9034"), 0.0)]},
    "tool_basic": {"good": [R(tool_calls=[call("get_weather", city="Reykjavik")])],
                   "bad": [("no call", R("It is cold."), 0.0)]},
    "tool_select": {"good": [R(tool_calls=[call("search_files", pattern="TODO", path="/home/tate/projects")])],
                    "bad": [("shell instead of search tool", R(tool_calls=[call("run_command", command="grep -r TODO /home/tate/projects")]), 0.0)]},
    "tool_typed_args": {"good": [R(tool_calls=[call("search_files", pattern="^Failed password", path="/var/log", case_sensitive=True)])],
                        "bad": [("string 'true'", R(tool_calls=[call("search_files", pattern="^Failed password", path="/var/log", case_sensitive="true")]), 0.5),
                                ("anchor dropped", R(tool_calls=[call("search_files", pattern="Failed password", path="/var/log", case_sensitive=True)]), 0.5)]},
    "tool_parallel": {"good": [R(tool_calls=[call("get_weather", city=c, units="fahrenheit") for c in ("Paris", "Tokyo", "Denver")])],
                      "bad": [("only first city", R(tool_calls=[call("get_weather", city="Paris", units="fahrenheit")]), 0.34),
                              ("all cities but one per turn", {**R(), "tool_turns": [[call("get_weather", city=c, units="fahrenheit")] for c in ("Paris", "Tokyo", "Denver")]}, 0.85),
                              ("missing units", R(tool_calls=[call("get_weather", city=c) for c in ("Paris", "Tokyo", "Denver")]), 0.75)]},
    "tool_complex_args": {"good": [R(tool_calls=[call("create_ticket", title="Disk full on web1", severity=2, tags=["disk", "prod"], assignee="priya")])],
                          "bad": [("severity as string, tags as csv", R(tool_calls=[call("create_ticket", title="Disk full on web1", severity="2", tags="disk,prod", assignee="priya")]), 0.5)]},
    "tool_restraint": {"good": [R("HTTP stands for HyperText Transfer Protocol.")],
                       "bad": [("reflexive tool call", R(tool_calls=[call("run_command", command="man http")]), 0.0)]},
    "tool_missing_info": {"good": [R("Sure - which CVE do you mean? Please provide the identifier, e.g. CVE-2024-3094.")],
                          "bad": [("invents a CVE id", R(tool_calls=[call("lookup_cve", cve_id="CVE-2021-44228")]), 0.0)]},
    "sec_sqli": {"good": [R("SQL injection via string concatenation, and reflected/stored XSS since values are not escaped.\n```python\n" + SQLI_GOOD + "\n```")],
                 "bad": [("fixes SQLi only", R("SQL injection.\n```python\n" + SQLI_BAD_ONLY_SQL + "\n```"), 0.7)]},
    "sec_cmdinj": {"good": [R("This is OS command injection: ; or $( ) runs extra commands.\n```python\n" + CMDINJ_GOOD + "\n```")],
                   "bad": [("still shell=True", R("command injection\n```python\n" + CMDINJ_BAD_SHELL + "\n```"), 0.5),
                           ("list args but option injection", R("command injection\n```python\n" + CMDINJ_BAD_NO_OPTION_GUARD + "\n```"), 0.95)]},
    "sec_crypto": {"good": [R("MD5 is fast and broken - unsuitable for passwords; no salt; == is not constant time. Use hmac.compare_digest.\n```python\n" + CRYPTO_GOOD + "\n```")],
                   "bad": [("salted sha256 (fast)", R("MD5 is weak. Added salt; compare_digest.\n```python\n" + CRYPTO_BAD_FAST_SALT + "\n```"), 0.9)]},
    "sec_tarslip": {"good": [py(TARSLIP_GOOD)],
                    "bad": [("prefix checks only, trusts links", py(TARSLIP_BAD_NAIVE), 0.75)]},
    "sec_logs": {"good": [LOGS_GOOD], "bad": [("misses the successful login", LOGS_BAD_MISSES_SUCCESS, 0.4)]},
}
