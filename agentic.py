"""Multi-turn agentic tasks: the model drives a tool loop in a sandboxed repo.

This is the category closest to real use (omp): the model must read before it
writes, run the tests, react to failures, and stop when done. Single-turn tool
tests cannot see any of that.

Loop contract (standard OpenAI tool calling):
  - tools: list_files, read_file, write_file, run_command
  - each assistant turn may call tools; results are fed back as role=tool
  - the episode ends when the model replies with no tool calls, or on limits
Grading happens AFTER the episode, on the workspace state, with hidden tests
the model never saw. Visible test files are hashed: editing them to make them
pass is detected and penalised.
"""
import hashlib
import json
import re
import time

from sandbox import Workspace
from scoring import final_answer

AGENT_SYSTEM = (
    "You are a coding agent working in a repository at /work. Use the tools to inspect files, "
    "make changes and run commands. Commands run in a Linux sandbox with Python 3.12 and pytest, "
    "without network access. Work step by step, verify your changes by running the tests, and when "
    "you are finished reply with a short summary and no tool calls.")

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List all files in the repository (recursive).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a UTF-8 text file from the repository.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to the repository root"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or overwrite a file with the given full content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a bash command in the repository root and return its output.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
]


def _sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def run_episode(chat_fn, task, max_turns=24, turn_tokens=4096, ctx_budget=30000):
    """Drive one episode. chat_fn(messages, tools, max_tokens) -> OpenAI response dict."""
    tools = task.get("tools") or TOOLS
    ws = Workspace(task["files"])
    messages = [{"role": "system", "content": AGENT_SYSTEM},
                {"role": "user", "content": task["prompt"]}]
    trace = []
    stats = {"turns": 0, "tool_calls": 0, "tool_errors": 0, "bad_json": 0,
             "completion_tokens": 0, "max_prompt_tokens": 0, "end": None}
    t0 = time.time()
    try:
        for turn in range(max_turns):
            try:
                d = chat_fn(messages, tools, turn_tokens)
            except Exception as e:
                stats["end"] = f"request failed: {type(e).__name__}: {str(e)[:200]}"
                break
            stats["turns"] += 1
            u = d.get("usage") or {}
            stats["completion_tokens"] += u.get("completion_tokens") or 0
            stats["max_prompt_tokens"] = max(stats["max_prompt_tokens"], u.get("prompt_tokens") or 0)
            ch = d["choices"][0]
            msg = ch["message"]
            calls = msg.get("tool_calls") or []
            # History carries content + tool calls, not the reasoning trace
            # (every template here strips prior reasoning anyway).
            hist = {"role": "assistant", "content": msg.get("content") or ""}
            if calls:
                hist["tool_calls"] = [{"id": c.get("id") or f"call_{turn}_{i}", "type": "function",
                                       "function": {"name": c["function"]["name"],
                                                    "arguments": c["function"].get("arguments") or "{}"}}
                                      for i, c in enumerate(calls)]
            messages.append(hist)
            trace.append({"turn": turn, "content": (msg.get("content") or "")[:2000],
                          "reasoning_len": len(msg.get("reasoning_content") or ""),
                          "finish": ch.get("finish_reason"),
                          "calls": [(c["function"]["name"], (c["function"].get("arguments") or "")[:300])
                                    for c in calls]})
            if not calls:
                stats["end"] = "truncated" if ch.get("finish_reason") == "length" else "done"
                break
            for c in hist["tool_calls"]:
                stats["tool_calls"] += 1
                out = _dispatch(ws, c["function"]["name"], c["function"]["arguments"], stats,
                                task.get("extra_tools"))
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
                trace[-1].setdefault("results", []).append(out[:400])
            if stats["max_prompt_tokens"] > ctx_budget:
                stats["end"] = "context budget exhausted"
                break
        else:
            stats["end"] = "turn limit"
        stats["elapsed"] = round(time.time() - t0, 1)
        final = next((m["content"] for m in reversed(messages)
                      if m["role"] == "assistant" and m.get("content")), "")
        score, note, details = task["grade"](ws, final, stats)
    finally:
        ws.close()
    details["stats"] = stats
    details["trace"] = trace
    return score, note, details


def _dispatch(ws, name, raw_args, stats, extra_tools=None):
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        if not isinstance(args, dict):
            raise ValueError("arguments must be an object")
    except Exception as e:
        stats["bad_json"] += 1
        return f"ERROR: arguments are not valid JSON: {e}"
    try:
        if extra_tools and name in extra_tools:
            return extra_tools[name](ws, args)
        if name == "list_files":
            return "\n".join(ws.listing()) or "(empty)"
        if name == "read_file":
            return ws.read(args["path"])[:20000]
        if name == "write_file":
            ws.write(args["path"], args["content"])
            return f"wrote {len(args['content'])} bytes to {args['path']}"
        if name == "run_command":
            code, out = ws.exec(args["command"], timeout=60)
            return f"exit code {code}\n{out[-6000:]}"
        stats["tool_errors"] += 1
        return f"ERROR: unknown tool {name!r}"
    except KeyError as e:
        stats["tool_errors"] += 1
        return f"ERROR: missing argument {e}"
    except Exception as e:
        stats["tool_errors"] += 1
        return f"ERROR: {type(e).__name__}: {e}"


def _pytest(ws, test_file_body, name="test_hidden_grader.py"):
    """Run hidden tests against the final workspace, then remove them."""
    ws.write(name, test_file_body)
    code, out = ws.exec(f"python -m pytest -q -p no:cacheprovider {name}", timeout=120)
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+) error", out)) else 0
    return passed, passed + failed + errors, out


_BASELINES = {}


def _baseline(task_id, files, hidden):
    """Hidden tests passing on the UNMODIFIED repo. An agent that does nothing
    must score ~0, not the fraction the buggy code already passes."""
    if task_id not in _BASELINES:
        with Workspace(files) as ws:
            _BASELINES[task_id] = _pytest(ws, hidden)[0]
    return _BASELINES[task_id]


def _relative(passed, total, base):
    if total <= base:
        return 0.0
    return max(0.0, (passed - base) / (total - base))


def _efficiency_note(stats):
    return f"{stats['turns']} turns, {stats['tool_calls']} calls, end={stats['end']}"


# ===========================================================================
# Task 1: bug fix in a small package, visible failing tests
# ===========================================================================
PRICING = '''"""Order pricing for the shop."""
from decimal import Decimal


def apply_discount(price, discount):
    """Apply a discount to a price.

    `discount` is a dict: {"type": "percent", "value": 15} takes 15% off,
    {"type": "fixed", "value": "5.00"} takes 5.00 off. Prices never go below 0.
    """
    value = Decimal(str(discount["value"]))
    if discount["type"] == "percent":
        return price - value
    if discount["type"] == "fixed":
        return price - value
    raise ValueError(f"unknown discount type {discount['type']!r}")


def line_total(unit_price, qty, discount=None):
    """Total for one order line: unit price * quantity, discount applied per unit."""
    price = Decimal(str(unit_price))
    if discount:
        price = apply_discount(price, discount)
    return price * qty


def order_total(lines, tax_rate="0.0825"):
    """Sum of line totals plus tax, rounded to cents (half-up)."""
    subtotal = sum(line_total(**l) for l in lines)
    total = subtotal * (1 + Decimal(tax_rate))
    return round(total, 2)
'''

PRICING_VISIBLE_TESTS = '''from decimal import Decimal
from shop.pricing import apply_discount, line_total, order_total


def test_percent_discount():
    assert apply_discount(Decimal("200.00"), {"type": "percent", "value": 15}) == Decimal("170.00")


def test_fixed_discount_floor():
    assert apply_discount(Decimal("3.00"), {"type": "fixed", "value": "5.00"}) == Decimal("0")


def test_order_total_rounds_half_up():
    lines = [{"unit_price": "10.00", "qty": 1}]
    # 10.00 * 1.0825 = 10.825 -> 10.83 (half-up), not 10.82 (banker's)
    assert order_total(lines) == Decimal("10.83")
'''

PRICING_HIDDEN = '''from decimal import Decimal
import pytest
from shop.pricing import apply_discount, line_total, order_total

D = Decimal

def test_percent():
    assert apply_discount(D("80"), {"type": "percent", "value": 25}) == D("60")

def test_percent_decimal_value():
    assert apply_discount(D("100"), {"type": "percent", "value": "12.5"}) == D("87.5")

def test_percent_hundred_floor():
    assert apply_discount(D("9.99"), {"type": "percent", "value": 100}) == D("0")

def test_fixed():
    assert apply_discount(D("10.00"), {"type": "fixed", "value": "2.50"}) == D("7.50")

def test_fixed_floor():
    assert apply_discount(D("1.00"), {"type": "fixed", "value": 3}) == D("0")

def test_unknown_type():
    with pytest.raises(ValueError):
        apply_discount(D("1"), {"type": "bogo", "value": 1})

def test_line_total_discount_per_unit():
    assert line_total("20.00", 3, {"type": "percent", "value": 10}) == D("54.00")

def test_order_total_half_up_cases():
    assert order_total([{"unit_price": "10.00", "qty": 1}]) == D("10.83")
    assert order_total([{"unit_price": "1.00", "qty": 1}], tax_rate="0.125") == D("1.13")

def test_order_total_two_places():
    t = order_total([{"unit_price": "3.30", "qty": 3}])
    assert t == D("10.72") and t.as_tuple().exponent == -2

def test_order_total_empty():
    assert order_total([]) == D("0.00")
'''


def _grade_bugfix(ws, final, stats):
    tampered = _sha(ws.read("tests/test_pricing.py")) != _sha(PRICING_VISIBLE_TESTS) \
        if ws.exists("tests/test_pricing.py") else True
    passed, total, out = _pytest(ws, PRICING_HIDDEN)
    base = _baseline("agent_bugfix", BUGFIX_FILES, PRICING_HIDDEN)
    score = _relative(passed, total, base)
    notes = [f"hidden {passed}/{total} (unmodified repo passes {base})"]
    if tampered:
        score = min(score, 0.25)
        notes.append("EDITED THE VISIBLE TESTS")
    if stats["end"] not in ("done",):
        score *= 0.8
    notes.append(_efficiency_note(stats))
    return score, "; ".join(notes), {"hidden_output": out[-2000:], "tampered": tampered}


BUGFIX_FILES = {"shop/__init__.py": "", "shop/pricing.py": PRICING,
                "tests/__init__.py": "", "tests/test_pricing.py": PRICING_VISIBLE_TESTS,
                "README.md": "# shop\n\nRun tests with `python -m pytest -q`.\n"}
BUGFIX = dict(
    id="agent_bugfix", category="agentic", weight=3.0,
    files=BUGFIX_FILES,
    prompt="The pricing tests in this repo are failing. Find and fix the bugs in the library code so the "
           "behaviour matches the docstrings. Do not modify the tests.",
    grade=_grade_bugfix,
)


# ===========================================================================
# Task 2: implement a feature from a spec in the README
# ===========================================================================
WORDFREQ = '''"""Word frequency utilities."""
import re
import sys

WORD_RE = re.compile(r"[A-Za-z']+")


def count_words(text):
    """Return a dict of word -> count (case-sensitive, apostrophes kept)."""
    counts = {}
    for w in WORD_RE.findall(text):
        counts[w] = counts.get(w, 0) + 1
    return counts


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    text = open(argv[0]).read() if argv else sys.stdin.read()
    for word, n in sorted(count_words(text).items()):
        print(f"{word}\\t{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

WORDFREQ_README = '''# wordfreq

## TODO: `top_n`

Add `top_n(text, n, *, ignore_case=True, stopwords=None)` to `wordfreq.py`.

- Words are matched with the existing `WORD_RE`, then leading/trailing apostrophes are stripped
  (`'tis` -> `tis`, `dogs'` -> `dogs`, but `don't` stays `don't`). A token that is only apostrophes is dropped.
- With `ignore_case=True` words are lowercased before counting.
- `stopwords` is an optional iterable of words to exclude; the comparison respects `ignore_case`
  (when ignoring case, `"The"` in stopwords excludes `"the"`).
- Returns a list of `(word, count)` tuples: highest count first, ties broken alphabetically,
  at most `n` items. `n` must be a positive int, otherwise raise `ValueError`.

Also add a CLI flag: `python wordfreq.py --top N [FILE]` prints the top N as `word<TAB>count`
lines using the defaults above. Without `--top`, behaviour is unchanged.
'''

WORDFREQ_VISIBLE = '''from wordfreq import count_words

def test_count_words():
    assert count_words("a b a") == {"a": 2, "b": 1}
'''

WORDFREQ_HIDDEN = '''import subprocess, sys, pytest
from wordfreq import top_n, count_words

def test_basic_order_and_ties():
    assert top_n("b a c b a d", 3) == [("a", 2), ("b", 2), ("c", 1)]

def test_ignore_case_default():
    assert top_n("The the THE cat", 1) == [("the", 3)]

def test_case_sensitive():
    assert top_n("The the THE", 3, ignore_case=False) == [("THE", 1), ("The", 1), ("the", 1)]

def test_apostrophes():
    assert top_n("'tis dogs' don't don't ''", 5) == [("don't", 2), ("dogs", 1), ("tis", 1)]

def test_stopwords_respect_case_folding():
    assert top_n("The cat and the hat", 5, stopwords=["The", "AND"]) == [("cat", 1), ("hat", 1)]

def test_stopwords_case_sensitive_mode():
    assert top_n("The the", 5, ignore_case=False, stopwords=["the"]) == [("The", 1)]

def test_n_larger_than_vocab():
    assert top_n("x y", 10) == [("x", 1), ("y", 1)]

@pytest.mark.parametrize("bad", [0, -1, 1.5, "3", None, True])
def test_bad_n(bad):
    with pytest.raises(ValueError):
        top_n("a", bad)

def test_existing_function_unchanged():
    assert count_words("The the") == {"The": 1, "the": 1}

def test_cli_top(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("b a b c b a")
    out = subprocess.run([sys.executable, "wordfreq.py", "--top", "2", str(f)],
                         capture_output=True, text=True, timeout=30).stdout
    assert out.strip().splitlines() == ["b\\t3", "a\\t2"]

def test_cli_default_unchanged(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("b a")
    out = subprocess.run([sys.executable, "wordfreq.py", str(f)],
                         capture_output=True, text=True, timeout=30).stdout
    assert out.strip().splitlines() == ["a\\t1", "b\\t1"]
'''
# `True` is an int subclass -- a model that checks isinstance(n, int) alone
# accepts it. The spec says positive int; bool is a known trap, graded low-key
# by being one parametrised case among many.


def _grade_feature(ws, final, stats):
    passed, total, out = _pytest(ws, WORDFREQ_HIDDEN)
    base = _baseline("agent_feature", FEATURE_FILES, WORDFREQ_HIDDEN)
    score = _relative(passed, total, base)
    notes = [f"hidden {passed}/{total} (unmodified repo passes {base})"]
    if stats["end"] != "done":
        score *= 0.8
    notes.append(_efficiency_note(stats))
    return score, "; ".join(notes), {"hidden_output": out[-2000:]}


FEATURE_FILES = {"wordfreq.py": WORDFREQ, "README.md": WORDFREQ_README, "test_wordfreq.py": WORDFREQ_VISIBLE}
FEATURE = dict(
    id="agent_feature", category="agentic", weight=3.0,
    files=FEATURE_FILES,
    prompt="Implement the TODO described in README.md. Keep existing behaviour working, and add tests for "
           "what you build.",
    grade=_grade_feature,
)


# ===========================================================================
# Task 3: investigation across many files, answer-only
# ===========================================================================
def _investigate_files():
    files = {"defaults.yaml": "retry_limit: 3\ntimeout_ms: 2000\ncircuit_breaker: enabled\n"}
    svc = {
        # name: (retry_limit override or None, circuit_breaker override or None)
        "auth-gw": (5, None), "billing-api": (2, "disabled"), "cart-svc": (None, "disabled"),
        "geo-lookup": (8, "disabled"), "ledger-sync": (4, "enabled"), "media-xcode": (None, None),
        "notif-fanout": (6, "disabled"), "report-gen": (3, "disabled"), "search-idx": (10, None),
        "audit-trail": (1, None), "inventory": (7, "disabled"), "pricing": (None, "enabled"),
    }
    for name, (retry, cb) in svc.items():
        body = [f"name: {name}", f"owner: team-{name.split('-')[0]}", "timeout_ms: 1500"]
        if retry is not None:
            body.append(f"retry_limit: {retry}")
        if cb is not None:
            body.append(f"circuit_breaker: {cb}")
        files[f"services/{name}.yaml"] = "\n".join(body) + "\n"
    # Decoys: a commented-out override, a retired file, and a changelog quoting old values.
    files["services/inventory.yaml"] = files["services/inventory.yaml"].replace(
        "retry_limit: 7", "retry_limit: 7\n# retry_limit: 2  (old)")
    files["services/cart-svc.yaml"] += "# retry_limit: 9   <- proposed in RFC-12, NOT applied\n"
    files["retired/legacy-batch.yaml"] = "name: legacy-batch\nretry_limit: 12\ncircuit_breaker: disabled\n"
    files["CHANGELOG.md"] = ("- 2025-02: search-idx circuit_breaker disabled temporarily (re-enabled 2025-03)\n"
                             "- 2025-04: auth-gw retry_limit 9 -> 5\n")
    files["README.md"] = ("Service configs live in services/. Each value not set in a service file "
                          "falls back to defaults.yaml. Files under retired/ are not deployed.\n")
    return files


INVESTIGATE_ANSWER = ["geo-lookup", "inventory", "notif-fanout"]
# retry > 3 : auth-gw(5) geo-lookup(8) ledger-sync(4) notif-fanout(6) search-idx(10) inventory(7)
# ...and circuit breaker disabled in the service file: geo-lookup, notif-fanout, inventory.
# Traps: cart-svc's commented retry 9, retired/legacy-batch, CHANGELOG's search-idx note,
# report-gen (retry 3 is NOT higher than the default).


def _grade_investigate(ws, final, stats):
    fa = final_answer(final)
    if fa is None:
        return 0.0, "no ANSWER line; " + _efficiency_note(stats), {"answer": None}
    got = {x.strip().strip("`'\".").lower() for x in re.split(r"[,\s]+", fa) if x.strip()}
    want = set(INVESTIGATE_ANSWER)
    tp = len(got & want)
    precision = tp / len(got) if got else 0
    recall = tp / len(want)
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return f1, f"ANSWER={sorted(got)} F1={f1:.2f}; {_efficiency_note(stats)}", {"answer": sorted(got)}


INVESTIGATE = dict(
    id="agent_investigate", category="agentic", weight=2.0,
    files=_investigate_files(),
    prompt="Using the repository, determine which DEPLOYED services have a retry_limit strictly higher than "
           "the default AND have their circuit breaker disabled. Values not set in a service's file fall back "
           "to defaults.yaml. End your final reply with a line of the form\n"
           "ANSWER: service1, service2, ...\n(alphabetical, comma-separated).",
    grade=_grade_investigate,
)

AGENT_TASKS = [BUGFIX, FEATURE, INVESTIGATE]


# ===========================================================================
# Task 4: Kepler orbit, built iteratively with a way to run the page
# ===========================================================================
# The one-shot kepler_orbit task gives the model no way to see its page run, so
# a duplicate `const`, a stray line of prose in the script, or an orbit drawn
# 1 px wide is fatal. In real agent use the model could run it and fix it. This
# task measures that loop. run_page reports what a developer would see in a
# browser -- errors, whether it animates, how big things are -- but never the
# grader's physics measurements (eccentricity, focus offset, equal areas), so
# the model cannot tune to the scorer.

def _run_page(ws, args):
    from canvas_probe import run_canvas
    path = args.get("path") or "kepler.html"
    try:
        html = ws.read(path)
    except Exception as e:
        return f"ERROR: cannot read {path}: {e}"
    log, canvases, err = run_canvas(html, frames=240)
    if err:
        return f"PAGE FAILED TO RUN: {err}"
    lines = []
    js_err = []
    for e in log:
        if e["op"] == "__error" and e["a"][0] not in js_err:
            js_err.append(e["a"][0])
    if js_err:
        lines.append(f"runtime errors ({len(js_err)} distinct): " + " | ".join(x[:160] for x in js_err[:4]))
    else:
        lines.append("no runtime errors in 240 animation frames")
    frames = sorted({e["f"] for e in log if e["op"] != "__error"})
    lines.append(f"frames with drawing: {len(frames)} of 240" + (" (static page, no animation?)" if len(frames) <= 1 else ""))
    for cid, w, h in canvases:
        lines.append(f"canvas #{cid}: {w}x{h}")
    ops = {}
    for e in log:
        ops[e["op"]] = ops.get(e["op"], 0) + 1
    lines.append("draw calls: " + ", ".join(f"{k}={v}" for k, v in sorted(ops.items()) if not k.startswith("__")))
    # Circles/arcs grouped by radius, with how far each moves: what you'd see by eye.
    groups = {}
    for e in log:
        if e["op"] == "arc":
            x, y, r = e["a"][:3]
            if all(isinstance(v, (int, float)) for v in (x, y, r)):
                groups.setdefault(round(r, 1), []).append((x, y))
    for r, pts in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:5]:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        lines.append(f"arc radius {r}: drawn {len(pts)}x, centre x {min(xs):.0f}..{max(xs):.0f}, "
                     f"y {min(ys):.0f}..{max(ys):.0f}")
    texts = [e["a"][0] for e in log if e["op"] == "fillText"][-6:]
    if texts:
        lines.append("last canvas text: " + " | ".join(t[:40] for t in texts))
    return "\n".join(lines)


def _grade_kepler(ws, final, stats):
    from kepler import kepler_scorer
    html_files = [f for f in ws.listing() if f.endswith(".html")]
    if not html_files:
        return 0.0, "no .html file written; " + _efficiency_note(stats), {}
    path = "kepler.html" if "kepler.html" in html_files else html_files[0]
    score, note, details = kepler_scorer({"content": ws.read(path)}, None)
    if stats["end"] != "done":
        score *= 0.9
    return score, f"{note}; {_efficiency_note(stats)}", {"file": path, **details}


KEPLER_AGENT = dict(
    id="agent_kepler", category="agentic", weight=2.0, turn_tokens=10000,
    files={"README.md": "Deliverable: kepler.html (single self-contained file).\n"},
    tools=[t for t in TOOLS if t["function"]["name"] != "run_command"] + [
        {"type": "function", "function": {
            "name": "run_page",
            "description": "Load an HTML file in a headless browser harness, run 240 animation frames, and "
                           "report script errors, whether it animates, canvas size, draw calls, where circles "
                           "are drawn and any canvas text.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "HTML file to run, e.g. kepler.html"}},
                "required": ["path"]}}}],
    extra_tools={"run_page": _run_page},
    prompt=None,   # set below from the one-shot prompt so both tasks ask for the same physics
    grade=_grade_kepler,
)


def _kepler_agent_prompt():
    from suite import KEPLER_PROMPT
    body = KEPLER_PROMPT.split("Output only the HTML")[0].strip()
    return (body + "\n\nWrite the page to kepler.html using write_file. Use run_page to run it and check the "
            "result, fix any problems you find, and reply with a short summary when it is done.")


KEPLER_AGENT["prompt"] = _kepler_agent_prompt()
AGENT_TASKS.append(KEPLER_AGENT)
