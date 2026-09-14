"""Benchmark suite v2 for local coding models on the 12 GB RTX 5070.

Every task is a DISCRIMINATOR with a documented failure mode. Categories:

  coding     code is EXECUTED against hidden tests / a V8 canvas probe
  agentic    multi-turn tool loop in a sandboxed workspace (see agentic.py)
  tools      single-turn function-calling discipline, arguments type-checked
  reasoning  verifiable answers, graded from an `ANSWER:` line in the content
  security   defensive: explanation checks PLUS the fixed code executed
             against real exploit payloads in the sandbox

v1 -> v2 changes are listed in README.md. Every scorer has known-good and
known-bad fixtures in fixtures/ and selftest.py fails if any scorer cannot
tell them apart.
"""
import json
import random

from kepler import kepler_scorer
from scoring import (answer_is, forbid_calls, grep_score, hidden_tests,
                     no_tool_call, tool_call, tool_calls_set, content)

ANSWER_CONTRACT = ("\n\nThink it through, then end your reply with one final line of the form\n"
                   "ANSWER: <your answer>")

# ---------------------------------------------------------------------------
# Tools used by the single-turn tool tasks
# ---------------------------------------------------------------------------
T_WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}
T_SEARCH = {"type": "function", "function": {
    "name": "search_files", "description": "Search for a regex pattern across files in a directory",
    "parameters": {"type": "object", "properties": {
        "pattern": {"type": "string", "description": "Regular expression"},
        "path": {"type": "string"},
        "case_sensitive": {"type": "boolean"}},
        "required": ["pattern", "path"]}}}
T_RUN = {"type": "function", "function": {
    "name": "run_command", "description": "Execute a shell command and return stdout",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}
T_CVE = {"type": "function", "function": {
    "name": "lookup_cve", "description": "Look up details for a CVE identifier",
    "parameters": {"type": "object", "properties": {
        "cve_id": {"type": "string", "description": "e.g. CVE-2021-44228"}},
        "required": ["cve_id"]}}}
T_TICKET = {"type": "function", "function": {
    "name": "create_ticket", "description": "Open an incident ticket",
    "parameters": {"type": "object", "properties": {
        "title": {"type": "string"},
        "severity": {"type": "integer", "minimum": 1, "maximum": 4,
                     "description": "1 = critical ... 4 = low"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "assignee": {"type": "string"}},
        "required": ["title", "severity"]}}}
ALL_TOOLS = [T_WEATHER, T_SEARCH, T_RUN, T_CVE, T_TICKET]


# ===========================================================================
# CODING
# ===========================================================================
KEPLER_PROMPT = """Write a single self-contained HTML file that animates a Keplerian two-body orbit on a <canvas>.

Requirements:
1. The planet must follow a true ellipse with eccentricity e = 0.6, and the star must sit at a FOCUS of that ellipse, not at its centre.
2. Motion must obey Kepler's second law (equal areas in equal times) - the planet visibly speeds up at periapsis and slows at apoapsis. Do this properly: advance the mean anomaly M linearly in time, then solve Kepler's equation M = E - e*sin(E) for the eccentric anomaly E using Newton-Raphson, then compute the position from E. Do not fake it with a velocity heuristic.
3. Draw the full orbital path, the star, and the planet. Trace the swept area or a trail so the second law is visible.
4. Show live readouts of true anomaly, current radius, and orbital speed.
5. No external libraries, no network requests. Everything inline in one file.

Output only the HTML file contents, starting with <!DOCTYPE html>. No commentary."""

DURATION_PROMPT = """Write a Python function `parse_duration(s)` that converts a compact duration string to a whole number of seconds.

Specification:
- The string is one or more components, each a non-negative decimal integer immediately followed by a unit: `d` (days), `h` (hours), `m` (minutes), `s` (seconds). Example: "1h30m15s" -> 5415.
- Units must appear in strictly descending order (d, h, m, s) and each unit at most once. "30m1h" and "1h1h" are invalid.
- Components need not be "normalised": "90m" -> 5400 is valid.
- No whitespace, signs, decimals, or other characters anywhere. Units are lowercase only.
- The empty string is invalid. A number without a unit ("15") is invalid; a unit without a number ("h") is invalid.
- Invalid strings raise `ValueError`. A non-`str` argument raises `TypeError`.

Return the complete implementation in one ```python code block. Standard library only."""

DURATION_TESTS = r'''
import pytest
from solution import parse_duration

@pytest.mark.parametrize("s,want", [
    ("0s", 0), ("45s", 45), ("1m", 60), ("1h", 3600), ("1d", 86400),
    ("1h30m15s", 5415), ("2d3h", 183600), ("90m", 5400), ("1d1s", 86401),
    ("0d0h0m0s", 0), ("100000s", 100000), ("007m", 420),
])
def test_valid(s, want):
    assert parse_duration(s) == want

@pytest.mark.parametrize("s", [
    "", "15", "h", "1h1h", "30m1h", "1s1m", "1H", "1h 30m", " 1h", "1h ",
    "-1h", "+1h", "1.5h", "1w", "1hm", "h1", "1h30", "1h\n", "１h",
])
def test_invalid(s):
    with pytest.raises(ValueError):
        parse_duration(s)

@pytest.mark.parametrize("bad", [None, 5, b"1h", ["1h"]])
def test_type(bad):
    with pytest.raises(TypeError):
        parse_duration(bad)
'''
# Discriminators: "１h" (full-width digit -- str.isdigit() and \d accept it),
# "1h\n" ($ in re matches before a trailing newline), ordering/duplicates, and
# TypeError vs ValueError.

LRU_PROMPT = """Implement a class `LRUCache` in Python.

- `LRUCache(capacity)`: `capacity` is an int >= 0; a negative capacity raises `ValueError`.
- `get(key, default=None)`: return the value (marking the key most-recently used) or `default` if absent.
- `put(key, value)`: insert or update. Updating an existing key also marks it most-recently used. When inserting would exceed capacity, evict the least-recently used key first. A capacity-0 cache stores nothing.
- `__len__` returns the number of stored items; `__contains__` reports membership WITHOUT changing recency.
- `get` and `put` must be O(1) average time.

Return the complete implementation in one ```python code block. Standard library only."""

LRU_TESTS = r'''
import time
import pytest
from solution import LRUCache

def test_basic_eviction():
    c = LRUCache(2)
    c.put("a", 1); c.put("b", 2); c.put("c", 3)
    assert c.get("a") is None and c.get("b") == 2 and c.get("c") == 3

def test_get_refreshes():
    c = LRUCache(2)
    c.put("a", 1); c.put("b", 2); c.get("a"); c.put("c", 3)
    assert "a" in c and "b" not in c

def test_update_refreshes_and_no_growth():
    c = LRUCache(2)
    c.put("a", 1); c.put("b", 2); c.put("a", 10); c.put("c", 3)
    assert c.get("a") == 10 and "b" not in c and len(c) == 2

def test_contains_does_not_refresh():
    c = LRUCache(2)
    c.put("a", 1); c.put("b", 2)
    assert "a" in c
    c.put("c", 3)
    assert "a" not in c

def test_default():
    c = LRUCache(1)
    assert c.get("x") is None and c.get("x", 7) == 7

def test_falsy_values_are_stored():
    c = LRUCache(3)
    c.put("z", 0); c.put("n", None); c.put("e", "")
    assert c.get("z", "miss") == 0 and c.get("n", "miss") is None and c.get("e", "miss") == ""

def test_capacity_zero():
    c = LRUCache(0)
    c.put("a", 1)
    assert len(c) == 0 and c.get("a") is None

def test_negative_capacity():
    with pytest.raises(ValueError):
        LRUCache(-1)

def test_o1_performance():
    # Sized from measurements in the sandbox: OrderedDict ~0.008 s, a list-based
    # recency queue ~1.8 s. String keys and cold-end access keep CPython's fast
    # int list.remove from hiding O(n) behaviour (it did at n=20000 int keys).
    n = 30000
    c = LRUCache(n)
    keys = [f"user:{i:08d}" for i in range(2 * n)]
    t = time.perf_counter()
    for i in range(n):
        c.put(keys[i], i)
    for _ in range(2):
        for i in range(n - 1, -1, -3):
            c.get(keys[i])
        for i in range(n, n + n // 4):
            c.put(keys[i], i)
    assert time.perf_counter() - t < 0.4, "get/put are not O(1)"
'''
# Discriminators: list-based recency (O(n), fails the timing test),
# `if value:` falsy-value bugs, __contains__ that refreshes, capacity 0.

MERGE_PROMPT = """This function is supposed to merge overlapping intervals, but it has bugs.

def merge_intervals(intervals):
    result = []
    for start, end in intervals:
        if result and start < result[-1][1]:
            result[-1][1] = end
        else:
            result.append([start, end])
    return result

Specification:
- Input: an iterable of (start, end) pairs of ints with start <= end, in ANY order.
- Intervals that overlap OR touch (e.g. (1, 2) and (2, 3)) are merged.
- Output: a new list of (start, end) TUPLES sorted by start, with no overlaps.
- The input must not be modified.

Fix the function. Return the complete corrected function in one ```python code block."""

MERGE_TESTS = r'''
import copy
from solution import merge_intervals

def test_unsorted():
    assert merge_intervals([(8, 10), (1, 3), (2, 6)]) == [(1, 6), (8, 10)]

def test_touching():
    assert merge_intervals([(1, 2), (2, 3)]) == [(1, 3)]

def test_nested_does_not_shrink():
    assert merge_intervals([(1, 10), (2, 3)]) == [(1, 10)]

def test_empty():
    assert merge_intervals([]) == []

def test_generator_input():
    assert merge_intervals(iter([(5, 6), (1, 2)])) == [(1, 2), (5, 6)]

def test_returns_tuples():
    out = merge_intervals([[1, 2], [3, 4]])
    assert all(type(x) is tuple for x in out)

def test_input_not_mutated():
    data = [[1, 4], [2, 5], [7, 8]]
    snap = copy.deepcopy(data)
    merge_intervals(data)
    assert data == snap

def test_single_points():
    assert merge_intervals([(3, 3), (3, 3), (1, 1)]) == [(1, 1), (3, 3)]
'''
# The bugs: no sort, `<` instead of `<=`, `= end` instead of max(), lists
# instead of tuples, and the aliasing that mutates caller data once fixed
# naively with result.append(interval).


# ===========================================================================
# SECURITY (defensive)
# ===========================================================================
SQLI_PROMPT = """Review this Flask endpoint. Identify every security vulnerability and name the class of each. Then give a corrected, complete, runnable version of the module: keep the imports, `app = Flask(__name__)`, the `/user` route, the `id` query parameter, the database file `app.db`, and the same HTML output.

from flask import Flask, request
import sqlite3

app = Flask(__name__)

@app.route('/user')
def user():
    uid = request.args.get('id')
    conn = sqlite3.connect('app.db')
    row = conn.execute("SELECT name, email FROM users WHERE id = " + uid).fetchone()
    return f"<h1>Welcome {row[0]}</h1><p>{row[1]}</p>"

Put the corrected module in one ```python code block."""

SQLI_TESTS = r'''
import os, sqlite3
import pytest

@pytest.fixture(scope="module")
def client():
    db = sqlite3.connect("app.db")
    db.execute("DROP TABLE IF EXISTS users")
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
    db.executemany("INSERT INTO users VALUES (?,?,?)", [
        (1, "alice", "alice@example.com"),
        (2, "<script>alert(1)</script>", "x@example.com"),
        (3, "bob", "<img src=x onerror=alert(2)>"),
    ])
    db.commit(); db.close()
    import solution
    solution.app.config["TESTING"] = True
    return solution.app.test_client()

def body(resp):
    return resp.get_data(as_text=True)

def test_normal_lookup_still_works(client):
    r = client.get("/user?id=1")
    assert r.status_code == 200 and "alice" in body(r)

def test_sqli_boolean_does_not_leak(client):
    r = client.get("/user?id=0 OR 1=1")
    assert "alice" not in body(r)

def test_sqli_union_does_not_leak(client):
    r = client.get("/user?id=0 UNION SELECT name, email FROM users WHERE id=1")
    assert "alice" not in body(r)

def test_xss_name_escaped(client):
    b = body(client.get("/user?id=2"))
    assert "<script>" not in b and ("&lt;script&gt;" in b or "&#60;script" in b.lower())

def test_xss_email_escaped(client):
    b = body(client.get("/user?id=3"))
    assert "<img" not in b

def test_missing_user_not_500(client):
    assert client.get("/user?id=999").status_code != 500

def test_missing_param_not_500(client):
    assert client.get("/user").status_code != 500
'''

CMDINJ_PROMPT = """This helper is used by an internal admin tool. Identify the vulnerability, explain how it could be exploited, and rewrite it safely. Keep the name and signature `ping_host(hostname) -> str` (it returns ping's output). Standard library only.

import os
def ping_host(hostname):
    return os.popen("ping -c 1 " + hostname).read()

Put the rewritten function (with its imports) in one ```python code block."""

CMDINJ_TESTS = r'''
import os, stat, pytest

BIN = os.path.abspath("fakebin")

@pytest.fixture(autouse=True)
def fake_ping(monkeypatch):
    os.makedirs(BIN, exist_ok=True)
    p = os.path.join(BIN, "ping")
    with open(p, "w") as f:
        f.write('#!/bin/sh\necho "PING-OK $*"\n')
    os.chmod(p, 0o755)
    monkeypatch.setenv("PATH", BIN + ":" + os.environ["PATH"])
    for m in ("/tmp/pwned1", "/tmp/pwned2", "/tmp/pwned3", "/tmp/pwned4"):
        if os.path.exists(m):
            os.remove(m)

from solution import ping_host

def call(h):
    try:
        return ping_host(h)
    except Exception:
        return None          # rejecting bad input is an acceptable defence

def test_legit_host_works():
    out = ping_host("127.0.0.1")
    assert "PING-OK" in out and "127.0.0.1" in out

def test_legit_hostname_works():
    assert "PING-OK" in ping_host("db-01.internal.example")

def test_semicolon():
    call("127.0.0.1; touch /tmp/pwned1")
    assert not os.path.exists("/tmp/pwned1")

def test_subshell():
    call("$(touch /tmp/pwned2)")
    assert not os.path.exists("/tmp/pwned2")

def test_backticks():
    call("`touch /tmp/pwned3`")
    assert not os.path.exists("/tmp/pwned3")

def test_pipe_newline():
    call("127.0.0.1\ntouch /tmp/pwned4")
    assert not os.path.exists("/tmp/pwned4")

def test_option_injection_rejected():
    # A hostname beginning with '-' is parsed by ping as an OPTION (argument
    # injection) even without a shell. It must be rejected or neutralised.
    out = call("-f")
    assert out is None or "PING-OK -c 1 -f" not in out
'''

CRYPTO_PROMPT = """Audit this password-handling code. List what is wrong and write a correct replacement using only the Python standard library. Keep the two function names: `hash_password(pw: str) -> str` returns a string suitable for storing in a database, and `check_password(pw: str, stored: str) -> bool`.

import hashlib
def hash_password(pw):
    return hashlib.md5(pw.encode()).hexdigest()

def check_password(pw, stored):
    return hash_password(pw) == stored

Put the replacement in one ```python code block."""

CRYPTO_TESTS = r'''
import hashlib, time
from solution import hash_password, check_password

def test_roundtrip():
    h = hash_password("correct horse")
    assert isinstance(h, str)
    assert check_password("correct horse", h) is True
    assert check_password("correct horsf", h) is False

def test_salted():
    assert hash_password("hunter2") != hash_password("hunter2")

def test_not_plain_digest():
    h = hash_password("hunter2")
    for algo in ("md5", "sha1", "sha256", "sha512"):
        assert hashlib.new(algo, b"hunter2").hexdigest() not in h

def test_unicode_and_empty():
    for pw in ("", "pässwörd", "密码"):
        assert check_password(pw, hash_password(pw))

def test_is_a_slow_kdf():
    # A salted SHA-256 costs about one hash; pbkdf2/scrypt at any sane work
    # factor costs thousands. Compare against a SHA-256 timed on the same
    # machine, so the threshold does not depend on how fast the CPU is
    # (an absolute 20 ms cut-off failed PBKDF2-200k on a Ryzen 9700X).
    def best(fn, reps):
        out = []
        for _ in range(3):
            t = time.perf_counter()
            for _ in range(reps):
                fn()
            out.append((time.perf_counter() - t) / reps)
        return min(out)
    one_sha = best(lambda: hashlib.sha256(b"saltsaltbenchmark").hexdigest(), 2000)
    kdf = best(lambda: hash_password("benchmark"), 1)
    assert kdf > 2000 * one_sha, f"hash_password costs {kdf/one_sha:.0f}x one SHA-256"

def test_garbage_stored_value_is_false_not_crash():
    try:
        assert check_password("x", "not-a-valid-hash") is False
    except (ValueError, TypeError):
        pass
'''

TARSLIP_PROMPT = """Write a Python function `safe_extract(archive_path, dest_dir)` that extracts a user-supplied tar archive safely. It must defend against path traversal ("tarslip"), absolute paths, and symlink/hardlink escapes, while still extracting ordinary archives correctly (including subdirectories). Rejecting a malicious archive by raising an exception is acceptable. Explain each defence in a comment. Standard library only.

Put the function (with imports) in one ```python code block."""

TARSLIP_TESTS = r'''
import io, os, tarfile, pytest
from solution import safe_extract

ROOT = os.path.abspath("jail")

def make(name, members):
    path = os.path.abspath(name)
    with tarfile.open(path, "w") as t:
        for m in members:
            kind = m[0]
            if kind == "file":
                data = m[2].encode()
                ti = tarfile.TarInfo(m[1]); ti.size = len(data)
                t.addfile(ti, io.BytesIO(data))
            elif kind == "dir":
                ti = tarfile.TarInfo(m[1]); ti.type = tarfile.DIRTYPE; t.addfile(ti)
            elif kind == "sym":
                ti = tarfile.TarInfo(m[1]); ti.type = tarfile.SYMTYPE; ti.linkname = m[2]; t.addfile(ti)
            elif kind == "hard":
                ti = tarfile.TarInfo(m[1]); ti.type = tarfile.LNKTYPE; ti.linkname = m[2]; t.addfile(ti)
    return path

def dest(name):
    d = os.path.join(ROOT, name, "dest")
    os.makedirs(d, exist_ok=True)
    return d

def attempt(archive, d):
    try:
        safe_extract(archive, d)
    except Exception:
        pass

def test_benign_archive_extracts():
    a = make("ok.tar", [("dir", "pkg"), ("file", "pkg/a.txt", "hello"), ("file", "b.txt", "bye")])
    d = dest("ok")
    safe_extract(a, d)
    assert open(os.path.join(d, "pkg", "a.txt")).read() == "hello"
    assert open(os.path.join(d, "b.txt")).read() == "bye"

def test_dotdot_traversal():
    d = dest("dotdot")
    attempt(make("dd.tar", [("file", "../evil.txt", "x")]), d)
    assert not os.path.exists(os.path.join(os.path.dirname(d), "evil.txt"))

def test_nested_dotdot():
    d = dest("nested")
    attempt(make("nd.tar", [("file", "a/../../evil2.txt", "x")]), d)
    assert not os.path.exists(os.path.join(os.path.dirname(d), "evil2.txt"))

def test_absolute_path():
    d = dest("abs")
    target = os.path.join(ROOT, "abs_evil.txt")
    attempt(make("abs.tar", [("file", target, "x")]), d)
    assert not os.path.exists(target)

def test_symlink_then_write_through_it():
    d = dest("sym")
    outside = os.path.join(ROOT, "sym_outside"); os.makedirs(outside, exist_ok=True)
    attempt(make("sym.tar", [("sym", "link", outside), ("file", "link/evil3.txt", "x")]), d)
    assert not os.path.exists(os.path.join(outside, "evil3.txt"))

def test_relative_symlink_escape():
    d = dest("relsym")
    outside = os.path.join(ROOT, "relsym")  # parent of dest
    attempt(make("rsym.tar", [("sym", "up", ".."), ("file", "up/evil4.txt", "x")]), d)
    assert not os.path.exists(os.path.join(outside, "evil4.txt"))

def test_hardlink_to_outside_file():
    d = dest("hard")
    secret = os.path.join(ROOT, "secret.txt")
    open(secret, "w").write("TOPSECRET")
    attempt(make("hard.tar", [("hard", "leak", secret)]), d)
    leak = os.path.join(d, "leak")
    assert not (os.path.exists(leak) and os.path.samefile(leak, secret))
'''


# ===========================================================================
# LONG CONTEXT (multi-hop retrieval with distractors)
# ===========================================================================
def build_longctx_prompt(target_tokens=20000, seed=7734):
    """~20k tokens of varied, plausible ops notes. Answering needs two facts
    planted far apart, and must ignore a retired value and a look-alike team.

    v1 used one sentence repeated ~300 times with the fact in the middle; every
    model passed, so it measured nothing."""
    rng = random.Random(seed)
    services = ["auth-gw", "billing-api", "ledger-sync", "search-idx", "notif-fanout",
                "media-xcode", "geo-lookup", "cart-svc", "report-gen", "audit-trail"]
    teams = ["Kestrel", "Osprey", "Heron", "Plover", "Merlin", "Harrier", "Kite", "Egret"]
    verbs = ["rotated", "patched", "rebalanced", "drained", "re-indexed", "scaled", "migrated", "audited"]
    things = ["TLS certificates", "connection pools", "cache shards", "worker queues",
              "replica sets", "feature flags", "rate limits", "backup snapshots"]

    def line():
        s, t, v, th = rng.choice(services), rng.choice(teams), rng.choice(verbs), rng.choice(things)
        n = rng.randint(2, 97)
        forms = [
            f"{s}: team {t} {v} the {th} after p{rng.randint(50,99)} latency crossed {n}0 ms.",
            f"Change {rng.randint(1000,9999)} ({s}) {v} {th}; rollback window {n} min, approved by {t}.",
            f"Postmortem note: {th} on {s} were {v} twice this week; {n}% of requests retried.",
            f"{t} reminds everyone that {s} {th} must be {v} before the Thursday freeze.",
            f"Capacity review: {s} runs {n} pods; {th} {v} on schedule, no action for {t}.",
        ]
        return rng.choice(forms)

    lines = [line() for _ in range(int(target_tokens / 24))]
    facts = [
        (0.12, "Ownership table (archived 2025): on-call for ledger-sync belonged to team Kite."),
        (0.31, "Ownership table (current): the on-call rotation for ledger-sync is owned by team Harrier."),
        (0.47, "Escalation codes are per team. Team Harrier's escalation code was HX-2210 until it was retired in March."),
        (0.58, "Team Harrier's current escalation code is HX-4471."),
        (0.66, "Team Kite's current escalation code is KT-9034."),
        (0.83, "Reminder: team Merlin's current escalation code is MR-5520."),
    ]
    for frac, text in sorted(facts, reverse=True):
        lines.insert(int(len(lines) * frac), text)
    doc = "\n".join(lines)
    return ("Below is a long operations log. Read it carefully.\n\n" + doc +
            "\n\nQuestion: What is the CURRENT escalation code of the team that CURRENTLY owns the "
            "on-call rotation for ledger-sync?" + ANSWER_CONTRACT)


# ===========================================================================
# THE SUITE
# ===========================================================================
# max_tokens are generous CAPS, not targets: the pilot showed 1500-3000 token
# limits truncating correct answers (score distorted by budget, not ability).
# Thinking mode adds THINK_EXTRA_TOKENS on top (configs.py).
TASKS = [
    # ------------------------------ coding --------------------------------
    dict(id="kepler_orbit", category="coding", weight=3.0, max_tokens=8000,
         artifact="html", prompt=KEPLER_PROMPT, scorer=kepler_scorer),
    dict(id="code_duration", category="coding", weight=1.5, max_tokens=6000,
         prompt=DURATION_PROMPT,
         scorer=hidden_tests("solution", DURATION_TESTS, must_define=("parse_duration",))),
    dict(id="code_lru", category="coding", weight=1.5, max_tokens=6000,
         prompt=LRU_PROMPT,
         scorer=hidden_tests("solution", LRU_TESTS, must_define=("LRUCache",))),
    dict(id="code_merge_fix", category="coding", weight=1.0, max_tokens=6000,
         prompt=MERGE_PROMPT,
         scorer=hidden_tests("solution", MERGE_TESTS, must_define=("merge_intervals",))),

    # ----------------------------- reasoning ------------------------------
    dict(id="reason_trains", category="reasoning", weight=1.0, max_tokens=4000,
         prompt="Town A and Town B are 300 km apart on a straight road. A cyclist leaves A toward B at "
                "09:00 traveling 20 km/h. A motorcyclist leaves B toward A at 11:00 traveling 60 km/h. "
                "At what clock time do they meet? Give the time as HH:MM (24-hour)." + ANSWER_CONTRACT,
         # 40 km covered by 11:00; 260 km at 80 km/h closing = 3h15m -> 14:15
         scorer=answer_is("14:15")),
    dict(id="reason_trace", category="reasoning", weight=1.0, max_tokens=4000,
         prompt="What exactly does this Python program print?\n\n"
                "def f(xs):\n    out = []\n    for i, x in enumerate(xs):\n        if x % 3 == 0:\n"
                "            out.append(x * 2)\n        elif x % 3 == 1:\n            out.insert(0, x)\n"
                "        else:\n            out = out[::-1]\n    return out\n\nprint(f([1, 2, 3, 4, 5, 6, 7]))"
                + ANSWER_CONTRACT,
         scorer=answer_is("[7, 6, 1, 4, 12]", normalise=lambda s: str(s).replace(" ", "").strip("`. "))),
    dict(id="reason_precedence", category="reasoning", weight=1.0, max_tokens=4000,
         prompt="In Python, what is the value of `0xF0 ^ 0x3C >> 2 | 1`? Give a decimal integer."
                + ANSWER_CONTRACT,
         # >> binds tighter than ^, which binds tighter than |:
         # 0x3C>>2 = 15; 0xF0^15 = 255; 255|1 = 255. Left-to-right reading gives 51.
         scorer=answer_is("255")),
    dict(id="reason_bayes", category="reasoning", weight=1.0, max_tokens=4000,
         prompt="Two fair six-sided dice are rolled. You are told that at least one die shows a 6. "
                "What is the probability that both dice show 6? Give an exact fraction." + ANSWER_CONTRACT,
         scorer=answer_is("1/11")),
    dict(id="reason_schedule", category="reasoning", weight=1.0, max_tokens=4000,
         prompt="A job first runs on Monday at 22:30 and then repeats every 90 minutes. On what day and at "
                "what time (24-hour) does its 10th run happen? Format: <Day> HH:MM" + ANSWER_CONTRACT,
         # 9 intervals * 90 min = 13h30m after Mon 22:30 -> Tue 12:00
         scorer=answer_is("Tuesday 12:00", "Tue 12:00")),
    dict(id="reason_constraint", category="reasoning", weight=1.0, max_tokens=4000,
         prompt="Four servers - alpha, beta, gamma, delta - each run exactly one service: cache, database, "
                "queue, web.\n1. alpha does not run the database or the web server.\n"
                "2. The queue runs on a server whose name comes alphabetically after gamma's.\n"
                "3. beta runs neither the cache nor the queue.\n4. gamma does not run the web server.\n"
                "Give the assignment as: alpha=<service>, beta=<service>, gamma=<service>, delta=<service>"
                + ANSWER_CONTRACT,
         scorer=answer_is("alpha=cache, beta=web, gamma=database, delta=queue",
                          normalise=lambda s: ",".join(sorted(p.strip() for p in
                                                              str(s).lower().replace(" ", "").strip(".`").split(","))))),
    # ~20.6k prompt tokens: the thinking budget must leave room in a 32k window.
    dict(id="reason_longctx", category="reasoning", weight=2.0, max_tokens=4000, think_max_tokens=4000,
         prompt=build_longctx_prompt(), scorer=answer_is("HX-4471")),

    # ------------------------------- tools --------------------------------
    dict(id="tool_basic", category="tools", weight=1.0, max_tokens=2000, tools=[T_WEATHER],
         prompt="What's the weather in Reykjavik right now? Use the tool.",
         scorer=tool_call("get_weather", contains={"city": "Reykjavik"})),
    dict(id="tool_select", category="tools", weight=1.0, max_tokens=2000, tools=ALL_TOOLS,
         prompt="Find every place the string TODO appears under /home/tate/projects.",
         scorer=tool_call("search_files", contains={"pattern": "TODO"}, exact={"path": "/home/tate/projects"})),
    dict(id="tool_typed_args", category="tools", weight=1.5, max_tokens=2000, tools=[T_SEARCH],
         prompt="Search /var/log for the case-sensitive regex ^Failed password",
         # v1 accepted the STRING "true"; a real tool rejects it.
         scorer=tool_call("search_files", exact={"pattern": "^Failed password", "path": "/var/log",
                                                 "case_sensitive": True}, types={"case_sensitive": bool})),
    dict(id="tool_parallel", category="tools", weight=1.5, max_tokens=2000, tools=[T_WEATHER],
         prompt="I'm flying Paris -> Tokyo -> Denver this week. Get the current weather in all three, in Fahrenheit.",
         followup_tools=lambda name, args: json.dumps(
             {"city": args.get("city"), "temp": 61, "units": args.get("units", "celsius"), "condition": "clear"}),
         scorer=tool_calls_set("get_weather", "city", ["Paris", "Tokyo", "Denver"],
                               exact={"units": "fahrenheit"})),
    dict(id="tool_complex_args", category="tools", weight=1.5, max_tokens=2000, tools=ALL_TOOLS,
         prompt="Open a severity-2 incident ticket titled 'Disk full on web1', tag it disk and prod, "
                "and assign it to priya.",
         scorer=tool_call("create_ticket", exact={"severity": 2, "assignee": "priya"},
                          contains={"title": "disk full"}, types={"severity": int, "tags": list})),
    dict(id="tool_restraint", category="tools", weight=1.5, max_tokens=2000, tools=ALL_TOOLS,
         prompt="What does the acronym HTTP stand for? Answer from your own knowledge.",
         scorer=no_tool_call(require=r"hypertext\s+transfer\s+protocol")),
    dict(id="tool_missing_info", category="tools", weight=1.5, max_tokens=2000, tools=ALL_TOOLS,
         # Discriminator: weak models invent an identifier to satisfy the tool.
         prompt="Can you look up that CVE for me and tell me how bad it is?",
         scorer=no_tool_call(require=r"which|what cve|specify|provide|identifier|\bid\b|number")),

    # ----------------------------- security -------------------------------
    dict(id="sec_sqli", category="security", weight=1.5, max_tokens=6000, prompt=SQLI_PROMPT,
         scorer=hidden_tests("solution", SQLI_TESTS, must_define=("app",), weight_exec=3.0,
                             greps=[("names SQL injection", r"sql\s*injection|\bsqli\b", 0.5),
                                    ("names XSS", r"\bxss\b|cross[- ]site scripting", 0.5)])),
    dict(id="sec_cmdinj", category="security", weight=1.5, max_tokens=6000, prompt=CMDINJ_PROMPT,
         scorer=hidden_tests("solution", CMDINJ_TESTS, must_define=("ping_host",),
                             pre=forbid_calls("os.system", "os.popen", "subprocess.run(shell=True)"), weight_exec=3.0,
                             greps=[("names command injection", r"command injection|shell injection|os command", 0.5)])),
    dict(id="sec_crypto", category="security", weight=1.5, max_tokens=6000, prompt=CRYPTO_PROMPT,
         scorer=hidden_tests("solution", CRYPTO_TESTS, must_define=("hash_password", "check_password"), weight_exec=3.0,
                             greps=[("MD5 unsuitable", r"md5.{0,120}(broken|collision|fast|unsuitable|insecure|weak|not.{0,20}(designed|suitable))", 0.25),
                                    ("constant-time compare", r"compare_digest", 0.25)])),
    dict(id="sec_tarslip", category="security", weight=1.5, max_tokens=6000, prompt=TARSLIP_PROMPT,
         scorer=hidden_tests("solution", TARSLIP_TESTS, must_define=("safe_extract",))),
    dict(id="sec_logs", category="security", weight=1.5, max_tokens=6000,
         prompt="""These lines are from /var/log/auth.log on an internet-facing host. What is happening, and what are the two most important things to do about it?

Sep 13 03:11:02 web1 sshd[2201]: Failed password for invalid user admin from 203.0.113.44 port 51122 ssh2
Sep 13 03:11:03 web1 sshd[2203]: Failed password for invalid user root from 203.0.113.44 port 51130 ssh2
Sep 13 03:11:04 web1 sshd[2205]: Failed password for invalid user oracle from 203.0.113.44 port 51138 ssh2
Sep 13 03:11:05 web1 sshd[2207]: Failed password for invalid user postgres from 203.0.113.44 port 51146 ssh2
Sep 13 03:11:07 web1 sshd[2210]: Failed password for invalid user test from 203.0.113.44 port 51160 ssh2
Sep 13 03:12:41 web1 sshd[2298]: Accepted password for deploy from 203.0.113.44 port 51844 ssh2
Sep 13 03:12:58 web1 sudo:   deploy : TTY=pts/0 ; PWD=/home/deploy ; USER=root ; COMMAND=/usr/bin/apt-get install -y netcat""",
         scorer=lambda r, t: (lambda g: (g[0], g[1], {}))(grep_score(content(r), [
             ("identifies brute force/spraying", r"brute[- ]?forc|password spray|credential stuffing|dictionary attack|guessing", 1),
             # The discriminator: does it notice the attacker GOT IN, from the same IP?
             ("notices the successful login", r"(accepted|succe\w+|valid).{0,80}deploy|deploy.{0,80}(accepted|succe\w+|compromis|logged in)", 3),
             ("flags netcat install as post-exploitation", r"netcat|\bnc\b", 2),
             ("says to treat host as compromised / isolate", r"compromis|isolat|incident response|forensic|take .{0,20}offline|quarantin", 2),
             ("rotate/lock the deploy credentials", r"(rotat|reset|disabl|lock|revok|chang).{0,40}(deploy|password|credential|key)", 1),
             ("harden ssh (keys only / block IP / fail2ban)", r"PasswordAuthentication\s+no|key[- ]based|public ?key|fail2ban|block.{0,30}203\.0\.113\.44|firewall", 1),
         ]))),
]

# Agentic tasks live in agentic.py and are appended by the runner.
