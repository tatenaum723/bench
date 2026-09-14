#!/usr/bin/env python3
"""Prove every omp-task grader discriminates, using hand-built final workspaces
and replies (no model). Includes a tamper attempt: a planted conftest.py that
forces every test to pass must not raise a score."""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agentic as A  # noqa: E402
from fixtures import reference as F  # noqa: E402
from omp_tasks import BY_ID, TASKS  # noqa: E402
from selftest import PRICING_FIXED, WORDFREQ_DONE  # noqa: E402

G, R_, DIM, RST = "\033[92m", "\033[91m", "\033[2m", "\033[0m"
CHEAT_CONFTEST = '''import pytest
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    out = yield
    rep = out.get_result()
    rep.outcome = "passed"
'''
EP = {"turns": 3, "tool_calls": 2, "tool_errors": 0, "end": "done"}
NO_TOOLS = {**EP, "tool_calls": 0}


def src(resp):
    """Pull the python source out of a fixture response."""
    from scoring import python_solution
    return python_solution(resp["content"])


def ws_with(task, files):
    d = tempfile.mkdtemp(prefix="ompst_")
    for rel, body in {**task["files"], **files}.items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(body)
    return d


CASES = {
    # task: [(label, files, final, episode, lo, hi)]
    "kepler_orbit": [("good page", {"kepler.html": F.KEPLER_GOOD}, "", EP, 0.95, 1),
                     ("circle page", {"kepler.html": F.KEPLER_CIRCLE}, "", EP, 0, 0.2),
                     ("nothing written", {}, "I would write...", EP, 0, 0)],
    "code_duration": [("good", {"duration.py": F.DURATION_GOOD}, "", EP, 0.95, 1),
                      ("loose", {"duration.py": F.DURATION_BAD_LOOSE}, "", EP, 0, 0.5),
                      ("good + cheating conftest", {"duration.py": F.DURATION_BAD_LOOSE, "conftest.py": CHEAT_CONFTEST}, "", EP, 0, 0.5)],
    "code_lru": [("good", {"lru.py": F.LRU_GOOD}, "", EP, 0.95, 1), ("O(n)", {"lru.py": F.LRU_BAD_LIST}, "", EP, 0, 0.9)],
    "code_merge_fix": [("fixed", {"intervals.py": F.MERGE_GOOD}, "", EP, 0.95, 1),
                       ("untouched", {}, "", EP, 0, 0.3)],
    "agent_bugfix": [("fixed", {"shop/pricing.py": PRICING_FIXED}, "Fixed.", EP, 0.95, 1),
                     ("untouched", {}, "Could not fix.", EP, 0, 0.05),
                     ("edits tests", {"tests/test_pricing.py": "def test_x():\n    assert True\n"}, "Done", EP, 0, 0.25),
                     ("cheating conftest", {"conftest.py": CHEAT_CONFTEST}, "Done", EP, 0, 0.05)],
    "agent_feature": [("implemented", {"wordfreq.py": WORDFREQ_DONE}, "Done.", EP, 0.95, 1),
                      ("untouched", {}, "Done.", EP, 0, 0.05)],
    "agent_investigate": [("right", {}, "ANSWER: geo-lookup, inventory, notif-fanout", EP, 0.95, 1),
                          ("decoys", {}, "ANSWER: cart-svc, geo-lookup, inventory, notif-fanout, search-idx", EP, 0, 0.8)],
    "reason_trains": [("right", {}, "They meet then.\nANSWER: 14:15", EP, 0.95, 1), ("wrong", {}, "ANSWER: 14:00", EP, 0, 0)],
    "reason_precedence": [("right", {}, "ANSWER: 255", EP, 0.95, 1), ("wrong", {}, "ANSWER: 51", EP, 0, 0)],
    "reason_bayes": [("right", {}, "ANSWER: 1/11", EP, 0.95, 1), ("wrong", {}, "ANSWER: 1/6", EP, 0, 0)],
    "reason_constraint": [("right", {}, "ANSWER: delta=queue, alpha=cache, beta=web, gamma=database", EP, 0.95, 1),
                          ("wrong", {}, "ANSWER: alpha=cache, beta=database, gamma=web, delta=queue", EP, 0, 0)],
    "reason_longctx": [("right", {}, "ANSWER: HX-4471", EP, 0.95, 1), ("retired", {}, "ANSWER: HX-2210", EP, 0, 0)],
    "tool_restraint": [("direct", {}, "HyperText Transfer Protocol.", NO_TOOLS, 0.95, 1),
                       ("used tools", {}, "HyperText Transfer Protocol.", EP, 0, 0.6)],
    "tool_missing_info": [("asks", {}, "Which CVE do you mean? Please provide the CVE ID.", NO_TOOLS, 0.95, 1),
                          ("invents", {}, "CVE-2021-44228 (Log4Shell) is critical, CVSS 10.", EP, 0, 0)],
    "sec_sqli": [("fixed", {"app.py": src(F.FIXTURES["sec_sqli"]["good"][0])},
                  "SQL injection and XSS, both fixed.", EP, 0.95, 1),
                 ("sql only", {"app.py": F.SQLI_BAD_ONLY_SQL}, "SQL injection.", EP, 0, 0.5)],
    "sec_cmdinj": [("fixed", {"netutil.py": F.CMDINJ_GOOD}, "OS command injection via ; and $( ).", EP, 0.95, 1),
                   ("shell=True", {"netutil.py": F.CMDINJ_BAD_SHELL}, "command injection", EP, 0, 0.5)],
    "sec_crypto": [("pbkdf2", {"passwords.py": F.CRYPTO_GOOD}, "MD5 is fast and broken; use compare_digest.", EP, 0.95, 1),
                   ("fast salted", {"passwords.py": F.CRYPTO_BAD_FAST_SALT}, "MD5 is weak.", EP, 0, 0.9)],
    "sec_tarslip": [("safe", {"safe_tar.py": F.TARSLIP_GOOD}, "", EP, 0.95, 1),
                    ("naive", {"safe_tar.py": F.TARSLIP_BAD_NAIVE}, "", EP, 0, 0.75)],
    "sec_logs": [("good", {}, F.LOGS_GOOD["content"], EP, 0.95, 1),
                 ("misses login", {}, F.LOGS_BAD_MISSES_SUCCESS["content"], EP, 0, 0.4)],
}


def main():
    only = set(sys.argv[1:])
    ok = True
    missing = [t["id"] for t in TASKS if t["id"] not in CASES]
    if missing:
        print(f"{R_}no cases for {missing}{RST}")
        ok = False
    for tid, cases in CASES.items():
        if only and tid not in only:
            continue
        t = BY_ID[tid]
        for label, files, final, ep, lo, hi in cases:
            d = ws_with(t, files)
            try:
                s, note, _ = t["grade"](d, final, ep)
            finally:
                shutil.rmtree(d, ignore_errors=True)
            good = lo <= s <= hi
            ok &= good
            print(f"  {G if good else R_}{'ok ' if good else 'BAD'}{RST} {tid:18} {label:26} {s:.2f} in [{lo}, {hi}]  {DIM}{note[:80]}{RST}")
    print(f"\n  {'ALL OMP GRADERS DISCRIMINATE' if ok else 'OMP SELFTEST FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
