"""Start, measure and stop an SGLang server safely.

The server runs in its own session/process group (start_new_session=True), so
stopping it signals the WHOLE group. That matters: killing only the
launch_server parent reparents its scheduler/detokenizer to systemd --user,
and SGLang children SIGQUIT their parent when they fail -- which core-dumped
the entire GNOME session on 2026-09-13.
"""
import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.request

SERVE = "/home/tate/qwen35/serve-model.sh"


class GpuSampler(threading.Thread):
    """Poll nvidia-smi for peak VRAM and temperature while the server runs."""

    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_mib = 0
        self.peak_temp = 0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout
                mem, temp = (int(x) for x in out.strip().split(","))
                self.peak_mib = max(self.peak_mib, mem)
                self.peak_temp = max(self.peak_temp, temp)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


class Server:
    def __init__(self, profile, port=30000, extra=(), env=None, log=None):
        self.profile = profile
        self.port = port
        self.extra = list(extra)
        self.env = dict(os.environ, PORT=str(port), **(env or {}))
        self.log_path = log or f"/tmp/bench_server_{profile}.log"
        self.proc = None
        self.gpu = GpuSampler()

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self, timeout=900):
        if port_in_use(self.port):
            stopped = stop_orphaned_bench_servers(self.port)
            if stopped:
                print(f"  [server] stopped orphaned bench server session(s) {stopped} on :{self.port}", flush=True)
            if port_in_use(self.port):
                raise RuntimeError(f"port {self.port} already in use -- stop that server first")
        self.logf = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            [SERVE, self.profile, *self.extra], env=self.env,
            stdout=self.logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True)
        self.gpu.start()
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited during startup (code {self.proc.returncode}); "
                                   f"see {self.log_path}\n" + self.tail())
            text = self.read_log()
            if "The server is fired up" in text:
                self.startup_s = time.time() - t0
                return self.metrics()
            # The torchcodec/libavutil traceback at startup is harmless, so only
            # treat a scheduler exception as fatal.
            if "Scheduler hit an exception" in text or "Received sigquit" in text:
                self.stop()
                raise RuntimeError("scheduler failed during startup:\n" + self.tail(40))
            time.sleep(2)
        self.stop()
        raise TimeoutError(f"server not ready after {timeout}s")

    def read_log(self):
        try:
            with open(self.log_path, errors="replace") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def tail(self, n=25):
        lines = [l for l in self.read_log().splitlines()
                 if not re.search(r"torchcodec|libavutil|ctypes|_dlopen|load_library|^\s+\^", l)]
        return "\n".join(lines[-n:])

    def metrics(self):
        """Pull the memory/KV numbers SGLang logs at startup."""
        t = self.read_log()
        m = {}
        if (x := re.search(r"Load weight end.*?mem usage=([\d.]+) GB", t)):
            m["weights_gb"] = float(x.group(1))
        if (x := re.search(r"max_total_num_tokens=(\d+)", t)):
            m["kv_tokens"] = int(x.group(1))
        if (x := re.search(r"context_len=(\d+)", t)):
            m["context_len"] = int(x.group(1))
        if (x := re.search(r"max_running_requests=(\d+)", t)):
            m["max_running_requests"] = int(x.group(1))
        if (x := re.search(r"swa size: (\d+), full size: (\d+)", t)):
            m["swa_tokens"], m["full_tokens"] = int(x.group(1)), int(x.group(2))
        if (x := re.search(r"available_gpu_mem=([\d.]+) GB", t)):
            m["headroom_gb"] = float(x.group(1))
        m["startup_s"] = round(getattr(self, "startup_s", 0), 1)
        return m

    def stop(self, grace=45):
        self.gpu.stop()
        if not self.proc:
            return
        sid = self.proc.pid          # start_new_session=True => pid == sid == pgid
        for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 15)):
            # Signal EVERY process in the server's session, not just its process
            # group: ninja starts JIT compile jobs in their own groups, so a
            # group-only kill leaves cicc/nvcc orphans holding GBs of RAM.
            # Parent and children die together, so nothing is left to SIGQUIT
            # systemd --user.
            for pid in session_pids(sid):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            t0 = time.time()
            while time.time() - t0 < wait and session_pids(sid):
                time.sleep(0.5)
            if not session_pids(sid):
                break
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass
        self.logf.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def session_pids(sid):
    """PIDs whose session id is `sid` (read from /proc, no pattern matching)."""
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                # field 6 is session; split after the ')' that ends comm
                fields = f.read().rsplit(")", 1)[1].split()
            if int(fields[3]) == sid:
                out.append(int(d))
        except (OSError, IndexError, ValueError):
            pass
    return out


def port_in_use(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def chat(base, payload, timeout=900):
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def stream_timing(base, model, prompt, max_tokens=512, extra=None):
    """Measure time-to-first-token and steady decode speed with a streamed request."""
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "stream": True,
               "stream_options": {"include_usage": True}, **(extra or {})}
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    stamps = []
    usage = {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    now = time.time()
                    first = first or now
                    stamps.append(now)
    n = usage.get("completion_tokens") or len(stamps)
    decode_s = (stamps[-1] - first) if len(stamps) > 1 else 0
    return {"ttft_s": round(first - t0, 3) if first else None,
            "decode_tok_s": round((n - 1) / decode_s, 1) if decode_s > 0 else None,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": n}


# ---------------------------------------------------------------------------
# Safety guard: never start new work while the box is short on RAM or hot.
# The GPU also drives the desktop, and the RAM kit was never soak-tested, so
# this benchmark doubles as a long stress test -- pause rather than push it.
# ---------------------------------------------------------------------------
LIMITS = {
    # name: (pause at, resume below)
    "gpu_c": (83, 75),
    "cpu_c": (90, 80),     # Ryzen 9700X Tctl; TjMax is 95
    "dimm_c": (70, 60),    # DDR5 SPD sensors
}
MIN_FREE_RAM_GB = 6.0


def _hwmon(name):
    out = []
    for h in os.listdir("/sys/class/hwmon"):
        base = os.path.join("/sys/class/hwmon", h)
        try:
            if open(os.path.join(base, "name")).read().strip() != name:
                continue
            out.append(int(open(os.path.join(base, "temp1_input")).read()) / 1000)
        except OSError:
            pass
    return out


def health():
    h = {}
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip().split(",")
        h["gpu_c"], h["gpu_mem_mib"] = int(g[0]), int(g[1])
    except Exception:
        pass
    cpu = _hwmon("k10temp")
    if cpu:
        h["cpu_c"] = max(cpu)
    dimm = _hwmon("spd5118")
    if dimm:
        h["dimm_c"] = max(dimm)
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                h["ram_free_gb"] = int(line.split()[1]) / 1048576
    except OSError:
        pass
    return h


_guard_lock = threading.Lock()


def wait_until_safe(log=print, max_wait_s=3600):
    """Block until RAM and temperatures are within limits. Returns seconds waited."""
    with _guard_lock:
        t0 = time.time()
        paused = False
        while True:
            h = health()
            hot = [k for k, (hi, lo) in LIMITS.items() if k in h and h[k] >= (lo if paused else hi)]
            low_ram = h.get("ram_free_gb", 99) < MIN_FREE_RAM_GB
            if not hot and not low_ram:
                if paused:
                    log(f"  [guard] resumed after {time.time() - t0:.0f}s: {h}")
                return time.time() - t0
            if not paused:
                log(f"  [guard] pausing new work: hot={hot} low_ram={low_ram} {h}")
                paused = True
            if time.time() - t0 > max_wait_s:
                raise RuntimeError(f"guard: unsafe conditions for over {max_wait_s}s: {h}")
            time.sleep(15)


def stop_orphaned_bench_servers(port=30000, grace=60):
    """Stop SGLang servers left behind by a crashed benchmark runner: only
    processes running `sglang.launch_server` from the qwen35 venv on this port,
    stopped by SIGTERM to their whole session (never kill -9 on the parent alone)."""
    sessions = set()
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            argv = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
            stat = open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        a = [x.decode(errors="replace") for x in argv]
        try:
            # A server run as a systemd service (e.g. the daily model) is not an orphan:
            # leave it alone and let start() report "port in use" instead.
            if ".service" in open(f"/proc/{d}/cgroup").read().split("/")[-1]:
                continue
        except OSError:
            pass
        if "sglang.launch_server" in a and "/home/tate/qwen35/" in a[0] and \
                (("--port" in a and a[a.index("--port") + 1] == str(port)) or ("--port" not in a and port == 30000)):
            sessions.add(int(stat[3]))
    for sid in sessions:
        for pid in session_pids(sid):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    t0 = time.time()
    while time.time() - t0 < grace and any(session_pids(sid) for sid in sessions):
        time.sleep(1)
    return sorted(sessions)
