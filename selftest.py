#!/usr/bin/env python3
"""Prove every scorer discriminates before any model is graded with it.

For each task: every known-good fixture must score >= 0.95 and every known-bad
fixture must score <= its stated ceiling. Agentic tasks are driven by scripted
agents (an oracle, and a cheater that edits the tests) through the real loop,
sandbox and graders. Exit code 1 on any failure.

  ./selftest.py            # everything
  ./selftest.py code_lru   # one task
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentic import AGENT_TASKS, run_episode  # noqa: E402
from fixtures.reference import FIXTURES  # noqa: E402
from suite import TASKS  # noqa: E402

G, R_, DIM, RST = "\033[92m", "\033[91m", "\033[2m", "\033[0m"


def check_task(t):
    fx = FIXTURES.get(t["id"])
    if not fx:
        print(f"  {R_}NO FIXTURES{RST} {t['id']}")
        return False
    ok = True
    for i, resp in enumerate(fx["good"]):
        s, note, _ = t["scorer"](resp, t)
        good = s >= 0.95
        ok &= good
        print(f"  {G if good else R_}{'ok ' if good else 'BAD'}{RST} {t['id']:18} good#{i}  {s:.2f}  {DIM}{note[:90]}{RST}")
    for label, resp, ceiling in fx["bad"]:
        s, note, _ = t["scorer"](resp, t)
        good = s <= ceiling
        ok &= good
        print(f"  {G if good else R_}{'ok ' if good else 'BAD'}{RST} {t['id']:18} bad     {s:.2f} (<= {ceiling})  "
              f"{label}  {DIM}{note[:70]}{RST}")
    return ok


# ---------------------------------------------------------------------------
# Scripted agents
# ---------------------------------------------------------------------------
PRICING_FIXED = '''"""Order pricing for the shop."""
from decimal import Decimal, ROUND_HALF_UP


def apply_discount(price, discount):
    value = Decimal(str(discount["value"]))
    if discount["type"] == "percent":
        new = price * (Decimal(100) - value) / Decimal(100)
    elif discount["type"] == "fixed":
        new = price - value
    else:
        raise ValueError(f"unknown discount type {discount['type']!r}")
    return max(new, Decimal(0))


def line_total(unit_price, qty, discount=None):
    price = Decimal(str(unit_price))
    if discount:
        price = apply_discount(price, discount)
    return price * qty


def order_total(lines, tax_rate="0.0825"):
    subtotal = sum((line_total(**l) for l in lines), Decimal(0))
    total = subtotal * (1 + Decimal(tax_rate))
    return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
'''

WORDFREQ_DONE = '''"""Word frequency utilities."""
import re
import sys

WORD_RE = re.compile(r"[A-Za-z']+")


def count_words(text):
    counts = {}
    for w in WORD_RE.findall(text):
        counts[w] = counts.get(w, 0) + 1
    return counts


def top_n(text, n, *, ignore_case=True, stopwords=None):
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive int")
    fold = (lambda w: w.lower()) if ignore_case else (lambda w: w)
    stop = {fold(s) for s in (stopwords or ())}
    counts = {}
    for raw in WORD_RE.findall(text):
        w = fold(raw.strip("'"))
        if not w or w in stop:
            continue
        counts[w] = counts.get(w, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    top = None
    if argv[:1] == ["--top"]:
        top = int(argv[1]); argv = argv[2:]
    text = open(argv[0]).read() if argv else sys.stdin.read()
    if top is not None:
        for word, c in top_n(text, top):
            print(f"{word}\\t{c}")
        return 0
    for word, c in sorted(count_words(text).items()):
        print(f"{word}\\t{c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def scripted(steps):
    """chat_fn that replays a fixed list of assistant turns."""
    it = iter(steps)

    def chat_fn(messages, tools, max_tokens):
        step = next(it, {"content": "Done."})
        calls = [{"id": f"c{i}", "type": "function",
                  "function": {"name": n, "arguments": json.dumps(a)}}
                 for i, (n, a) in enumerate(step.get("calls", []))]
        return {"choices": [{"finish_reason": "tool_calls" if calls else "stop",
                             "message": {"content": step.get("content", ""), "tool_calls": calls}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}
    return chat_fn


from fixtures.reference import KEPLER_CIRCLE, KEPLER_GOOD, KEPLER_SYNTAX  # noqa: E402

AGENT_CASES = {
    "agent_kepler": [
        ("oracle", [{"calls": [("write_file", {"path": "kepler.html", "content": KEPLER_GOOD})]},
                    {"calls": [("run_page", {"path": "kepler.html"})]},
                    {"content": "Done."}], 0.95, 1.0),
        ("fixes a syntax error after run_page", [
            {"calls": [("write_file", {"path": "kepler.html", "content": KEPLER_SYNTAX})]},
            {"calls": [("run_page", {"path": "kepler.html"})]},
            {"calls": [("write_file", {"path": "kepler.html", "content": KEPLER_GOOD})]},
            {"calls": [("run_page", {"path": "kepler.html"})]},
            {"content": "Fixed the duplicate declaration."}], 0.95, 1.0),
        ("circle with star at centre", [{"calls": [("write_file", {"path": "kepler.html", "content": KEPLER_CIRCLE})]},
                                        {"content": "Done."}], 0.0, 0.2),
        ("never writes a file", [{"content": "Here is how you would do it..."}], 0.0, 0.0),
    ],
    "agent_bugfix": [
        ("oracle", [{"calls": [("list_files", {})]},
                    {"calls": [("read_file", {"path": "shop/pricing.py"})]},
                    {"calls": [("write_file", {"path": "shop/pricing.py", "content": PRICING_FIXED})]},
                    {"calls": [("run_command", {"command": "python -m pytest -q"})]},
                    {"content": "Fixed percent discount, floor at zero, half-up rounding."}], 0.95, 1.0),
        ("cheater edits tests", [{"calls": [("write_file", {"path": "tests/test_pricing.py",
                                                           "content": "def test_ok():\n    assert True\n"})]},
                                 {"content": "Tests pass now."}], 0.0, 0.25),
        ("gives up", [{"content": "I could not figure it out."}], 0.0, 0.1),
    ],
    "agent_feature": [
        ("oracle", [{"calls": [("read_file", {"path": "README.md"})]},
                    {"calls": [("write_file", {"path": "wordfreq.py", "content": WORDFREQ_DONE})]},
                    {"content": "Implemented top_n and --top."}], 0.95, 1.0),
        ("ignores bool and apostrophes", [{"calls": [("write_file", {"path": "wordfreq.py", "content":
            WORDFREQ_DONE.replace("isinstance(n, bool) or ", "").replace('raw.strip("\'")', "raw")})]},
            {"content": "done"}], 0.0, 0.9),
    ],
    "agent_investigate": [
        ("oracle", [{"calls": [("run_command", {"command": "grep -r . services defaults.yaml"})]},
                    {"content": "Checked.\nANSWER: geo-lookup, inventory, notif-fanout"}], 0.95, 1.0),
        ("falls for decoys", [{"content": "ANSWER: cart-svc, geo-lookup, inventory, legacy-batch, notif-fanout, search-idx"}], 0.0, 0.7),
        ("no answer line", [{"content": "geo-lookup and inventory"}], 0.0, 0.0),
        ("malformed tool args survive", [{"calls": [("read_file", {"nope": 1})]},
                                         {"content": "ANSWER: geo-lookup, inventory, notif-fanout"}], 0.95, 1.0),
    ],
}


def check_agentic(task):
    ok = True
    for label, steps, lo, hi in AGENT_CASES[task["id"]]:
        s, note, det = run_episode(scripted(steps), task)
        good = lo <= s <= hi
        ok &= good
        print(f"  {G if good else R_}{'ok ' if good else 'BAD'}{RST} {task['id']:18} {label:30} {s:.2f} "
              f"in [{lo}, {hi}]  {DIM}{note[:80]}{RST}")
        if not good and det.get("hidden_output"):
            print(DIM + det["hidden_output"][-1500:] + RST)
    return ok


def main():
    only = set(sys.argv[1:])
    ok = True
    for t in TASKS:
        if not only or t["id"] in only:
            ok &= check_task(t)
    for t in AGENT_TASKS:
        if not only or t["id"] in only:
            ok &= check_agentic(t)
    print(f"\n  {'ALL SCORERS DISCRIMINATE' if ok else 'SELFTEST FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
