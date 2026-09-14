"""Scoring primitives shared by every task.

Rules that every scorer here follows (each one fixes a v1 flaw):
  1. Grade the FINAL ANSWER (message content / tool_calls), never the
     reasoning trace. v1 concatenated reasoning into the graded text, so a
     model could earn credit for a keyword it considered and then discarded.
  2. Execute code instead of grepping it wherever execution is possible.
     Execution happens only inside the sandbox (sandbox.py).
  3. Types are checked as types: a JSON boolean is not the string "true".
  4. A scorer returns (score in [0,1], note, details_dict).
"""
import ast
import json
import re

from sandbox import Workspace, run_tests


def content(r):
    return r.get("content") or ""


# ---------------------------------------------------------------------------
# Final-answer extraction
# ---------------------------------------------------------------------------
ANSWER_RE = re.compile(r"^\s*\**\s*ANSWER\s*\**\s*[:：]\s*(.+?)\s*$", re.I | re.M)


def final_answer(text):
    """Value of the LAST `ANSWER: ...` line, or None."""
    hits = ANSWER_RE.findall(text or "")
    return hits[-1].strip().strip("*`").strip() if hits else None


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().strip(".").lower()


def answer_is(*accepted, normalise=norm):
    """Exact match on the ANSWER line. Half credit when the right value appears
    only in the last 300 chars without the requested ANSWER line (the model got
    it right but ignored the output contract an agent would parse)."""
    acc = [normalise(a) for a in accepted]

    def f(r, _task):
        c = content(r)
        fa = final_answer(c)
        if fa is not None:
            ok = normalise(fa) in acc
            return (1.0 if ok else 0.0), f"ANSWER={fa[:60]!r}", {"answer": fa}
        tail = normalise(c[-300:])
        if any(a in tail for a in acc):
            return 0.5, "correct value but no ANSWER line", {"answer": None}
        return 0.0, "no ANSWER line, value not found", {"answer": None}
    return f


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------
FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.S)


def code_blocks(text, lang=None):
    out = []
    for m in FENCE_RE.finditer(text or ""):
        tag = m.group(1).lower()
        if lang is None or tag in lang:
            out.append(m.group(2))
    return out


def python_solution(text, must_define=()):
    """Pick the python block that defines the required names; fall back to the
    longest block that parses. Returns source or None."""
    blocks = code_blocks(text, lang=("python", "py", "python3", "")) or []
    parsed = []
    for b in blocks:
        try:
            tree = ast.parse(b)
        except SyntaxError:
            continue
        names = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))}
        names |= {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign)
                  for t in n.targets if isinstance(t, ast.Name)}
        parsed.append((all(m in names for m in must_define), len(b), b))
    if not parsed:
        return None
    parsed.sort(key=lambda x: (x[0], x[1]))
    return parsed[-1][2]


# ---------------------------------------------------------------------------
# Executed-code scorer
# ---------------------------------------------------------------------------
def hidden_tests(module, test_src, must_define=(), extra_files=None,
                 pre=None, weight_exec=1.0, greps=None, timeout=90):
    """Extract the solution, drop it in as `module`.py next to hidden pytest
    tests, run in the sandbox. Score = pass fraction (optionally blended with
    explanation greps over the final answer)."""
    def f(r, _task):
        c = content(r)
        src = python_solution(c, must_define)
        details = {}
        if src is None:
            exec_score, note = 0.0, "no parseable python code block"
        else:
            if pre:
                bad = pre(src)
                if bad:
                    details["static_violation"] = bad
            files = {f"{module}.py": src, "test_hidden.py": test_src, **(extra_files or {})}
            passed, total, out = run_tests(files, timeout=timeout)
            details.update(passed=passed, total=total, output=out[-3000:])
            exec_score = passed / total if total else 0.0
            note = f"hidden tests {passed}/{total}"
            if total == 0:
                note = "tests did not run: " + (out.strip().splitlines() or ["?"])[-1][:120]
            if details.get("static_violation"):
                exec_score = min(exec_score, 0.5)
                note += f"; {details['static_violation']}"
        if not greps:
            return exec_score, note, details
        g_score, g_note = grep_score(c, greps)
        details["grep"] = g_note
        total_w = weight_exec + sum(w for _, _, w in greps)
        score = (exec_score * weight_exec + g_score * sum(w for _, _, w in greps)) / total_w
        return score, f"{note}; {g_note}", details
    return f


def grep_score(text, checks):
    """checks: [(label, regex, weight)]. Returns (fraction, note)."""
    got = tot = 0.0
    missing = []
    for label, rx, w in checks:
        tot += w
        if re.search(rx, text, re.I | re.S):
            got += w
        else:
            missing.append(label)
    return (got / tot if tot else 0.0), ("missing: " + ", ".join(missing) if missing else "explanation checks hit")


def forbid_calls(*dotted):
    """Static check: return a violation string if the source calls any of these
    (e.g. 'os.system', 'subprocess.run(shell=True)')."""
    def check(src):
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            name = ast.unparse(n.func)
            for d in dotted:
                if d.endswith("(shell=True)"):
                    if name.startswith("subprocess.") or name in ("run", "Popen", "check_output", "call"):
                        for kw in n.keywords:
                            if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value:
                                return "uses shell=True"
                elif name == d:
                    return f"calls {d}"
        return None
    return check


# ---------------------------------------------------------------------------
# Tool-call scorers (single turn)
# ---------------------------------------------------------------------------
def _calls(r):
    out = []
    for c in r.get("tool_calls") or []:
        fn = c.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = None
        out.append((fn.get("name"), args))
    return out


def tool_call(name, exact=None, contains=None, types=None, count=1):
    """Require `count` calls to `name` and nothing else.
      exact:    {arg: value}   must equal (strings compared case-insensitively)
      contains: {arg: substr}  substring match
      types:    {arg: type}    isinstance check (bool is NOT str)
    """
    def f(r, _task):
        calls = _calls(r)
        if not calls:
            return 0.0, "no tool call", {}
        wrong = [n for n, _ in calls if n != name]
        mine = [a for n, a in calls if n == name]
        if not mine:
            return 0.0, f"called {[n for n, _ in calls]}, expected {name}", {}
        if any(a is None for a in mine):
            return 0.2, "arguments are not valid JSON", {}
        problems = []
        for a in mine:
            for k, v in (exact or {}).items():
                got = a.get(k)
                same = (got.lower() == v.lower()) if isinstance(got, str) and isinstance(v, str) else got == v
                if not same:
                    problems.append(f"{k}={got!r} (want {v!r})")
            for k, v in (contains or {}).items():
                if v.lower() not in str(a.get(k, "")).lower():
                    problems.append(f"{k}={a.get(k)!r} (want ~{v!r})")
            for k, t in (types or {}).items():
                if k not in a or type(a[k]) is not t:
                    problems.append(f"{k} type {type(a.get(k)).__name__} (want {t.__name__})")
        score = 1.0
        if problems:
            score = 0.5
        if len(mine) != count:
            score = min(score, 0.5)
            problems.append(f"{len(mine)} calls (want {count})")
        if wrong:
            score = min(score, 0.5)
            problems.append(f"also called {wrong}")
        return score, ("; ".join(problems) if problems else "correct"), {"calls": calls}
    return f


def tool_calls_set(name, key, values, exact=None, sequential_cap=0.85):
    """One call to `name` per value of `key` (order-free).

    Runs multi-turn (the runner feeds back tool results, see `followup_tools`),
    because issuing calls one per turn is a legitimate agent strategy: the pilot
    showed Granite doing exactly that, which a single-turn check scored as a 33%
    failure. Covering every target across turns earns up to `sequential_cap`;
    covering them in ONE turn earns full credit."""
    want = {v.lower() for v in values}

    def f(r, _task):
        turns = r.get("tool_turns") or [r.get("tool_calls") or []]
        calls = [(n, a) for t in turns for n, a in _calls({"tool_calls": t})]
        mine = [a for n, a in calls if n == name and isinstance(a, dict)]
        got = {str(a.get(key, "")).lower() for a in mine}
        hit = len(want & got)
        score = hit / len(want)
        notes = [f"{hit}/{len(want)} targets"]
        turns_used = sum(1 for t in turns if t)
        if hit == len(want) and turns_used > 1:
            score = min(score, sequential_cap)
            notes.append(f"sequential over {turns_used} turns")
        if len(mine) > len(want):
            score *= 0.75
            notes.append(f"{len(mine)} calls")
        for a in mine:
            for k, v in (exact or {}).items():
                if a.get(k) != v:
                    score *= 0.75
                    notes.append(f"{k}={a.get(k)!r}")
                    break
        if any(n != name for n, _ in calls):
            score *= 0.5
            notes.append("called other tools")
        return score, "; ".join(notes), {"calls": calls, "turns": turns_used}
    return f


def no_tool_call(require=None):
    """Restraint: answer without calling anything. `require` = regex the answer must match."""
    def f(r, _task):
        calls = _calls(r)
        if calls:
            return 0.0, f"called {[n for n, _ in calls]}", {"calls": calls}
        c = content(r)
        if not c.strip():
            return 0.0, "empty answer", {}
        if require and not re.search(require, c, re.I | re.S):
            return 0.5, "no call, but answer misses the point", {}
        return 1.0, "answered without tools", {}
    return f
