"""The benchmark suite as omp tasks: each is a prompt Tate could type into omp,
a starting workspace, and a grader that inspects the workspace and omp's final
reply afterwards.

Reuses the v2 prompts, hidden tests and physics grader, so scores stay
comparable in spirit, but everything now goes through omp's own agent loop,
system prompt, tool set and context limits.

Grading is tamper-resistant: only the deliverable files are copied out of the
workspace into a clean directory next to the hidden tests. A model with a real
shell could otherwise plant a conftest.py that fakes results.
"""
import os
import re

import agentic as A
import suite as S
from kepler import kepler_scorer
from sandbox import run_tests
from scoring import answer_is, final_answer, grep_score

RUN = "\n\nWork in the current directory."
ANSWER = "\n\nWhen you're done, end your final reply with a line of the form\nANSWER: <your answer>"


def _read(ws, rel):
    p = os.path.join(ws, rel)
    try:
        with open(p, errors="replace") as f:
            return f.read()
    except OSError:
        return None


def hidden(deliverables, test_src, module_alias=None):
    """Grade by copying ONLY `deliverables` (paths in the workspace) into a clean
    dir with the hidden tests. module_alias: {"workspace/path.py": "solution.py"}."""
    def grade(ws, final, ep):
        files = {}
        missing = []
        for rel in deliverables:
            body = _read(ws, rel)
            if body is None:
                missing.append(rel)
                continue
            files[(module_alias or {}).get(rel, rel)] = body
        if missing and not files:
            return 0.0, f"deliverable missing: {', '.join(missing)}", {}
        files["test_hidden.py"] = test_src
        passed, total, out = run_tests(files)
        score = passed / total if total else 0.0
        note = f"hidden tests {passed}/{total}" + (f"; missing {missing}" if missing else "")
        return score, note, {"output": out[-2500:]}
    return grade


def relative(grade_fn, baseline_files, deliverables, test_src):
    """Only tests the agent MADE pass count: subtract what the untouched repo passes."""
    cache = {}

    def grade(ws, final, ep):
        if "base" not in cache:
            files = {r: baseline_files[r] for r in deliverables if r in baseline_files}
            files["test_hidden.py"] = test_src
            cache["base"] = run_tests(files)[0]
        s, note, det = grade_fn(ws, final, ep)
        m = re.search(r"hidden tests (\d+)/(\d+)", note)
        if not m:
            return s, note, det
        p, t = int(m.group(1)), int(m.group(2))
        b = cache["base"]
        rel = max(0.0, (p - b) / (t - b)) if t > b else 0.0
        return rel, f"{note} (untouched repo passes {b})", det
    return grade


def with_explanation(grade_fn, checks, exec_weight=3.0):
    def grade(ws, final, ep):
        s, note, det = grade_fn(ws, final, ep)
        g, gnote = grep_score(final, checks)
        w = sum(c[2] for c in checks)
        return (s * exec_weight + g * w) / (exec_weight + w), f"{note}; {gnote}", det
    return grade


def answer(*accepted, normalise=None):
    scorer = answer_is(*accepted, **({"normalise": normalise} if normalise else {}))

    def grade(ws, final, ep):
        return scorer({"content": final}, None)
    return grade


def kepler_file(path="kepler.html"):
    def grade(ws, final, ep):
        html = _read(ws, path)
        if html is None:
            htmls = [f for f in os.listdir(ws) if f.endswith(".html")]
            html = _read(ws, htmls[0]) if htmls else None
        if html is None:
            return 0.0, "no html file written", {}
        return kepler_scorer({"content": html}, None)
    return grade


def no_tools_answer(require):
    def grade(ws, final, ep):
        n = ep["tool_calls"]
        if not final.strip():
            return 0.0, "empty reply", {}
        if not re.search(require, final, re.I | re.S):
            return 0.3, f"reply misses the point ({n} tool calls)", {}
        if n:
            return 0.6, f"right answer but made {n} unnecessary tool calls", {}
        return 1.0, "answered directly", {}
    return grade


def clarifies(require):
    """The request lacks the key detail. Asking beats inventing it."""
    def grade(ws, final, ep):
        invented = re.search(r"CVE-\d{4}-\d{3,}", final) and not re.search(require, final, re.I)
        if invented:
            return 0.0, "picked a CVE instead of asking which one", {}
        if re.search(require, final, re.I):
            return 1.0, "asked for the missing identifier", {}
        return 0.3, "neither asked nor answered usefully", {}
    return grade


def _code_prompt(prompt, filename):
    """v2 prompts ask for a code block; in omp the deliverable is a file."""
    p = re.sub(r"Return the complete (implementation|corrected function) in one ```python code block\.",
               f"Put it in {filename}.", prompt)
    p = p.replace("Put the corrected module in one ```python code block.", f"Save the corrected module as {filename}.")
    p = p.replace("Put the rewritten function (with its imports) in one ```python code block.", f"Save it in {filename}.")
    p = p.replace("Put the replacement in one ```python code block.", f"Save the replacement in {filename}.")
    p = p.replace("Put the function (with imports) in one ```python code block.", f"Save it in {filename}.")
    return p + RUN


TASKS = [
    # ------------------------------ coding --------------------------------
    dict(id="kepler_orbit", category="coding", weight=3.0, files={},
         prompt=S.KEPLER_PROMPT.replace("Output only the HTML file contents, starting with <!DOCTYPE html>. No commentary.",
                                        "Save it as kepler.html.") + RUN,
         grade=kepler_file()),
    dict(id="code_duration", category="coding", weight=1.5, files={},
         prompt=_code_prompt(S.DURATION_PROMPT, "duration.py"),
         grade=hidden(["duration.py"], S.DURATION_TESTS, {"duration.py": "solution.py"})),
    dict(id="code_lru", category="coding", weight=1.5, files={},
         prompt=_code_prompt(S.LRU_PROMPT, "lru.py"),
         grade=hidden(["lru.py"], S.LRU_TESTS, {"lru.py": "solution.py"})),
    dict(id="code_merge_fix", category="coding", weight=1.0,
         files={"intervals.py": "def merge_intervals(intervals):\n    result = []\n    for start, end in intervals:\n"
                                "        if result and start < result[-1][1]:\n            result[-1][1] = end\n"
                                "        else:\n            result.append([start, end])\n    return result\n"},
         prompt="merge_intervals in intervals.py is supposed to merge overlapping intervals, but it has bugs.\n\n"
                + S.MERGE_PROMPT.split("Specification:")[1].split("Fix the function.")[0].strip().join(["Specification:\n", ""])
                + "\n\nFix it in place." + RUN,
         grade=hidden(["intervals.py"], S.MERGE_TESTS, {"intervals.py": "solution.py"})),

    # ------------------------------ agentic -------------------------------
    dict(id="agent_bugfix", category="agentic", weight=3.0, files=A.BUGFIX_FILES, prompt=A.BUGFIX["prompt"] + RUN,
         grade=lambda ws, final, ep: _tamper_guard(ws, final, ep)),
    dict(id="agent_feature", category="agentic", weight=3.0, files=A.FEATURE_FILES, prompt=A.FEATURE["prompt"] + RUN,
         grade=relative(hidden(["wordfreq.py"], A.WORDFREQ_HIDDEN), A.FEATURE_FILES, ["wordfreq.py"], A.WORDFREQ_HIDDEN)),
    dict(id="agent_investigate", category="agentic", weight=2.0, files=A._investigate_files(),
         prompt=A.INVESTIGATE["prompt"] + RUN,
         grade=lambda ws, final, ep: A._grade_investigate(None, final, {"turns": ep["turns"], "tool_calls": ep["tool_calls"], "end": ep["end"]})),

    # ----------------------------- reasoning ------------------------------
    # omp may compute with python/bash here -- that is how Tate would get the answer.
    dict(id="reason_trains", category="reasoning", weight=1.0, files={},
         prompt="Town A and Town B are 300 km apart on a straight road. A cyclist leaves A toward B at 09:00 "
                "traveling 20 km/h. A motorcyclist leaves B toward A at 11:00 traveling 60 km/h. At what clock "
                "time do they meet? Give the time as HH:MM (24-hour)." + ANSWER, grade=answer("14:15")),
    dict(id="reason_precedence", category="reasoning", weight=1.0, files={},
         prompt="Without running any code, what is the value of the Python expression `0xF0 ^ 0x3C >> 2 | 1`? "
                "Give a decimal integer." + ANSWER, grade=answer("255")),
    dict(id="reason_bayes", category="reasoning", weight=1.0, files={},
         prompt="Two fair six-sided dice are rolled. You are told that at least one die shows a 6. What is the "
                "probability that both dice show 6? Give an exact fraction." + ANSWER, grade=answer("1/11")),
    dict(id="reason_constraint", category="reasoning", weight=1.0, files={},
         prompt="Four servers - alpha, beta, gamma, delta - each run exactly one service: cache, database, queue, "
                "web.\n1. alpha does not run the database or the web server.\n2. The queue runs on a server whose "
                "name comes alphabetically after gamma's.\n3. beta runs neither the cache nor the queue.\n4. gamma "
                "does not run the web server.\nGive the assignment as: alpha=<service>, beta=<service>, "
                "gamma=<service>, delta=<service>" + ANSWER,
         grade=answer("alpha=cache, beta=web, gamma=database, delta=queue",
                      normalise=lambda s: ",".join(sorted(p.strip() for p in str(s).lower().replace(" ", "").strip(".`").split(","))))),
    dict(id="reason_longctx", category="reasoning", weight=2.0,
         # A ~20k-token log as a FILE: omp's ~18.7k prompt floor means it cannot
         # simply read it all into a 32k window -- it has to search it.
         files={"ops.log": S.build_longctx_prompt().split("Read it carefully.\n\n", 1)[1].split("\n\nQuestion:")[0]},
         prompt="ops.log is our operations log. What is the CURRENT escalation code of the team that CURRENTLY owns "
                "the on-call rotation for ledger-sync?" + ANSWER, grade=answer("HX-4471")),

    # ------------------------- tool judgement -----------------------------
    dict(id="tool_restraint", category="tools", weight=1.0, files={},
         prompt="What does the acronym HTTP stand for?", grade=no_tools_answer(r"hypertext\s+transfer\s+protocol")),
    dict(id="tool_missing_info", category="tools", weight=1.0, files={},
         prompt="Can you look up that CVE for me and tell me how bad it is?",
         grade=clarifies(r"which (cve|one|vulnerabilit)|what cve|specify|provide (the|a|me)|cve (id|identifier|number)")),

    # ----------------------------- security -------------------------------
    dict(id="sec_sqli", category="security", weight=1.5,
         files={"app.py": S.SQLI_PROMPT.split("the same HTML output.\n\n", 1)[1].split("\n\nPut the corrected")[0].rstrip() + "\n"},
         prompt="Review app.py (a Flask endpoint). Identify every security vulnerability and name the class of each, "
                "then fix app.py in place. Keep the `/user` route, the `id` query parameter, the database file "
                "`app.db`, and the same HTML output." + RUN,
         grade=with_explanation(hidden(["app.py"], S.SQLI_TESTS, {"app.py": "solution.py"}),
                                [("names SQL injection", r"sql\s*injection|\bsqli\b", 0.5),
                                 ("names XSS", r"\bxss\b|cross[- ]site scripting", 0.5)])),
    dict(id="sec_cmdinj", category="security", weight=1.5,
         files={"netutil.py": "import os\n\n\ndef ping_host(hostname):\n    return os.popen(\"ping -c 1 \" + hostname).read()\n"},
         prompt="netutil.ping_host is used by an internal admin tool. Identify the vulnerability, explain how it could "
                "be exploited, and fix netutil.py in place. Keep the signature `ping_host(hostname) -> str` (it "
                "returns ping's output). Standard library only." + RUN,
         grade=with_explanation(hidden(["netutil.py"], S.CMDINJ_TESTS, {"netutil.py": "solution.py"}),
                                [("names command injection", r"command injection|shell injection|os command", 0.5)])),
    dict(id="sec_crypto", category="security", weight=1.5,
         files={"passwords.py": "import hashlib\n\n\ndef hash_password(pw):\n    return hashlib.md5(pw.encode()).hexdigest()\n\n\n"
                                "def check_password(pw, stored):\n    return hash_password(pw) == stored\n"},
         prompt="Audit passwords.py. List what is wrong and fix it in place using only the Python standard library. "
                "Keep `hash_password(pw: str) -> str` (a string suitable for storing in a database) and "
                "`check_password(pw: str, stored: str) -> bool`." + RUN,
         grade=with_explanation(hidden(["passwords.py"], S.CRYPTO_TESTS, {"passwords.py": "solution.py"}),
                                [("MD5 unsuitable", r"md5.{0,160}(broken|collision|fast|unsuitable|insecure|weak|not.{0,20}(designed|suitable))", 0.25),
                                 ("constant-time compare", r"compare_digest|constant[- ]time|timing", 0.25)])),
    dict(id="sec_tarslip", category="security", weight=1.5, files={},
         prompt=_code_prompt(S.TARSLIP_PROMPT, "safe_tar.py"),
         grade=hidden(["safe_tar.py"], S.TARSLIP_TESTS, {"safe_tar.py": "solution.py"})),
    dict(id="sec_logs", category="security", weight=1.5,
         files={"auth.log": S.TASKS[[t["id"] for t in S.TASKS].index("sec_logs")]["prompt"].split("\n\n", 1)[1] + "\n"},
         prompt="auth.log is from an internet-facing host. What is happening, and what are the two most important "
                "things to do about it?" + RUN,
         grade=lambda ws, final, ep: S.TASKS[[t["id"] for t in S.TASKS].index("sec_logs")]["scorer"]({"content": final}, None)),
]


def _tamper_guard(ws, final, ep):
    visible = _read(ws, "tests/test_pricing.py")
    tampered = visible is None or visible != A.PRICING_VISIBLE_TESTS
    g = relative(hidden(["shop/__init__.py", "shop/pricing.py"], A.PRICING_HIDDEN),
                 A.BUGFIX_FILES, ["shop/__init__.py", "shop/pricing.py"], A.PRICING_HIDDEN)
    s, note, det = g(ws, final, ep)
    if tampered:
        s = min(s, 0.25)
        note += "; EDITED THE VISIBLE TESTS"
    return s, note, det


BY_ID = {t["id"]: t for t in TASKS}
