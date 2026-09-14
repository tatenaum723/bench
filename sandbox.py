"""Run model-generated code inside a locked-down container.

Model output is untrusted input. Everything executed on its behalf -- hidden
tests against its code, shell commands issued by agentic tasks -- runs in the
bench-sandbox image with no network, a read-only root, no capabilities, a
non-root user and pid/memory/cpu limits. Only the per-episode workspace is
writable, and it is a throwaway temp dir.
"""
import os
import shutil
import subprocess
import tempfile
import uuid

IMAGE = "bench-sandbox:1"
HARDEN = [
    "--network", "none",
    "--read-only",
    "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges",
    "--pids-limit", "128",
    "--memory", "768m", "--memory-swap", "768m",
    "--cpus", "2",
    "--tmpfs", "/tmp:size=64m,exec",
    "--user", "10001:10001",
    "-e", "PYTHONDONTWRITEBYTECODE=1",
    "-e", "HOME=/tmp",
]


class Workspace:
    """A temp dir mounted at /work inside one long-lived container."""

    def __init__(self, files=None):
        self.dir = tempfile.mkdtemp(prefix="bench_ws_")
        os.chmod(self.dir, 0o777)
        for rel, body in (files or {}).items():
            self.write(rel, body)
        self.name = f"bench_{uuid.uuid4().hex[:10]}"
        subprocess.run(
            ["docker", "run", "-d", "--rm", "--name", self.name, *HARDEN,
             "-v", f"{self.dir}:/work", "-w", "/work", IMAGE, "sleep", "3600"],
            check=True, capture_output=True, text=True)

    # --- host-side file access (never follows paths outside the workspace) ---
    def _safe(self, rel):
        p = os.path.realpath(os.path.join(self.dir, rel.lstrip("/").removeprefix("work/")))
        if p != self.dir and not p.startswith(self.dir + os.sep):
            raise PermissionError(f"path escapes workspace: {rel}")
        return p

    def write(self, rel, body):
        p = self._safe(rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(body)
        os.chmod(p, 0o666)

    def read(self, rel):
        with open(self._safe(rel), errors="replace") as f:
            return f.read()

    def exists(self, rel):
        return os.path.exists(self._safe(rel))

    def listing(self):
        out = []
        for root, dirs, files in os.walk(self.dir):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache")]
            for f in files:
                out.append(os.path.relpath(os.path.join(root, f), self.dir))
        return sorted(out)

    def exec(self, cmd, timeout=60):
        """Run a shell command in the container. Returns (exit_code, output)."""
        try:
            r = subprocess.run(
                ["docker", "exec", self.name, "timeout", "-s", "KILL", str(timeout),
                 "bash", "-c", cmd],
                capture_output=True, text=True, timeout=timeout + 15)
            out = (r.stdout + r.stderr)
            code = r.returncode
        except subprocess.TimeoutExpired:
            out, code = "timed out", 124
        if code == 137:
            out += "\n[killed: exceeded time limit]"
        return code, out[-12000:]

    def close(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
        # Files created inside the container are owned by uid 10001.
        subprocess.run(["docker", "run", "--rm", "-v", f"{self.dir}:/work", "--user", "0:0",
                        "--network", "none", IMAGE, "sh", "-c", "chmod -R 777 /work"],
                       capture_output=True)
        shutil.rmtree(self.dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def run_tests(files, test_cmd="python -m pytest -q -p no:cacheprovider", timeout=90):
    """One-shot: write files, run the test command, return (passed, total, output)."""
    import re
    with Workspace(files) as ws:
        code, out = ws.exec(test_cmd, timeout=timeout)
    passed = failed = errors = 0
    if (m := re.search(r"(\d+) passed", out)):
        passed = int(m.group(1))
    if (m := re.search(r"(\d+) failed", out)):
        failed = int(m.group(1))
    if (m := re.search(r"(\d+) error", out)):
        errors = int(m.group(1))
    return passed, passed + failed + errors, out
