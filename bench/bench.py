"""Measure gateway overhead and failover cost against local mock providers.

Starts two mock providers and the gateway as separate processes on localhost,
then reports latency percentiles. Nothing here talks to a real LLM.

    python bench/bench.py --n 2000 --out bench/RESULTS.md
"""

from __future__ import annotations

import argparse
import asyncio
import os
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

CFG = """
require_auth: false
providers:
  - {{name: primary, type: openai, base_url: "http://127.0.0.1:{p1}/v1", timeout_s: {timeout}}}
  - {{name: secondary, type: openai, base_url: "http://127.0.0.1:{p2}/v1", timeout_s: {timeout}}}
routes:
  bench: {{targets: [{{provider: primary, model: mock-smart}}, {{provider: secondary, model: mock-smart}}]}}
breaker: {{failure_threshold: {threshold}, reset_timeout_s: 3600}}
retry: {{attempts_per_provider: 1, backoff_base_s: 0.0, backoff_max_s: 0.0}}
cache: {{enabled: false}}
rate_limit: {{default_rpm: 100000000, default_burst: 100000000}}
"""
BODY = {"model": "bench", "messages": [{"role": "user", "content": "Repeat exactly: benchmark payload"}],
        "temperature": 0.7}


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


def summary(xs_ms: list[float]) -> dict[str, float]:
    return {"p50": pct(xs_ms, 50), "p90": pct(xs_ms, 90), "p99": pct(xs_ms, 99), "mean": statistics.fmean(xs_ms)}


def spawn(args: list[str]) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)


async def wait_up(url: str) -> None:
    async with httpx.AsyncClient() as c:
        for _ in range(200):
            try:
                if (await c.get(url)).status_code < 500:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.05)
    raise RuntimeError(f"{url} did not come up")


async def timed(c: httpx.AsyncClient, url: str, body=BODY, n: int = 1) -> tuple[list[float], list[httpx.Response]]:
    lat, resps = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        r = await c.post(url, json=body)
        lat.append((time.perf_counter() - t0) * 1000)
        resps.append(r)
    return lat, resps


class Stack:
    def __init__(self, threshold: int, timeout: float) -> None:
        self.p1, self.p2, self.pg = 19101, 19102, 19100
        self.threshold, self.timeout = threshold, timeout
        self.procs: dict[str, subprocess.Popen] = {}

    async def __aenter__(self):
        cfg = Path(tempfile.mkdtemp()) / "bench.yaml"
        cfg.write_text(CFG.format(p1=self.p1, p2=self.p2, timeout=self.timeout, threshold=self.threshold))
        self.procs["m1"] = spawn(["-m", "llm_gateway.mock_server", "--host", "127.0.0.1", "--port", str(self.p1)])
        self.procs["m2"] = spawn(["-m", "llm_gateway.mock_server", "--host", "127.0.0.1", "--port", str(self.p2)])
        self.procs["gw"] = spawn(["-m", "llm_gateway", "--config", str(cfg), "--host", "127.0.0.1",
                                  "--port", str(self.pg), "--log-level", "WARNING"])
        for p in (self.p1, self.p2):
            await wait_up(f"http://127.0.0.1:{p}/health")
        await wait_up(f"http://127.0.0.1:{self.pg}/healthz")
        return self

    async def __aexit__(self, *exc):
        for p in self.procs.values():
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()

    def kill(self, name: str) -> None:
        os.killpg(self.procs[name].pid, signal.SIGKILL)
        self.procs[name].wait()

    @property
    def gw(self) -> str:
        return f"http://127.0.0.1:{self.pg}/v1/chat/completions"

    def mock(self, port: int) -> str:
        return f"http://127.0.0.1:{port}/v1/chat/completions"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="bench/RESULTS.md")
    a = ap.parse_args()
    res: dict[str, dict[str, float]] = {}

    async with httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_keepalive_connections=8)) as c:
        # 1) Overhead: direct-to-mock vs through gateway, same keep-alive client, sequential.
        async with Stack(threshold=3, timeout=2.0) as s:
            await timed(c, s.mock(s.p1), {**BODY, "model": "mock-smart"}, 200)  # warm-up
            await timed(c, s.gw, n=200)
            d, _ = await timed(c, s.mock(s.p1), {**BODY, "model": "mock-smart"}, a.n)
            g, rs = await timed(c, s.gw, n=a.n)
            assert all(r.status_code == 200 for r in rs)
            res["direct to mock"] = summary(d)
            res["through gateway"] = summary(g)
            res["gateway overhead (p50 delta)"] = {"p50": res["through gateway"]["p50"] - res["direct to mock"]["p50"]}

        # 2) Fresh stack per round (breaker disabled): HTTP 503, then process killed.
        f503, f_refused, f_open = [], [], []
        rounds = 50
        for _ in range(rounds):
            async with Stack(threshold=1_000_000, timeout=2.0) as s:
                await timed(c, s.gw, n=20)  # warm
                async with httpx.AsyncClient() as admin:
                    await admin.post(f"http://127.0.0.1:{s.p1}/_faults", json={"mode": "error", "status": 503})
                lat, rs = await timed(c, s.gw, n=20)
                assert all(r.json()["x_gateway"]["provider"] == "secondary" for r in rs)
                f503 += lat
                # Primary process killed (connection refused) -> failover.
                s.kill("m1")
                lat, rs = await timed(c, s.gw, n=20)
                assert all(r.status_code == 200 and r.json()["x_gateway"]["failover"] for r in rs)
                f_refused += lat
            if len(f503) >= a.n // 4:
                break
        res["failover: primary HTTP 503"] = summary(f503)
        res["failover: primary killed (conn refused)"] = summary(f_refused)

        # 3) Breaker open: primary skipped without a network attempt.
        async with Stack(threshold=3, timeout=2.0) as s:
            await timed(c, s.gw, n=20)
            s.kill("m1")
            await timed(c, s.gw, n=3)  # trips breaker
            lat, rs = await timed(c, s.gw, n=a.n)
            assert all(r.json()["x_gateway"]["attempts"] == 1 for r in rs)
            f_open = lat
        res["breaker open (primary skipped)"] = summary(f_open)

        # 4) Primary hangs: bounded by the per-provider timeout (set 0.25 s here).
        async with Stack(threshold=1_000_000, timeout=0.25) as s:
            await timed(c, s.gw, n=20)
            async with httpx.AsyncClient() as admin:
                await admin.post(f"http://127.0.0.1:{s.p1}/_faults", json={"mode": "hang"})
            lat, rs = await timed(c, s.gw, n=20)
            assert all(r.json()["x_gateway"]["provider"] == "secondary" for r in rs)
        res["failover: primary hangs (timeout_s=0.25)"] = summary(lat)

    hw = _hardware()
    lines = ["# Benchmark results", "",
             f"Measured {time.strftime('%Y-%m-%d %H:%M %Z')} with `python bench/bench.py --n {a.n}`.", "",
             f"**Hardware:** {hw}", "",
             ("Setup: gateway, primary mock and secondary mock run as three separate local processes "
             "(uvicorn, 1 worker each) on 127.0.0.1. Requests are sequential over a keep-alive client, "
             "cache disabled, retries=1 per provider. Mock providers answer instantly, so these numbers "
             "isolate gateway cost; real LLM latency (hundreds of ms+) dominates in practice."), "",
             "| Scenario | p50 ms | p90 ms | p99 ms | mean ms | samples |", "|---|---|---|---|---|---|"]
    counts = {"direct to mock": a.n, "through gateway": a.n, "failover: primary HTTP 503": len(f503),
              "failover: primary killed (conn refused)": len(f_refused),
              "breaker open (primary skipped)": len(f_open), "failover: primary hangs (timeout_s=0.25)": 20}
    for k, v in res.items():
        if k.startswith("gateway overhead"):
            continue
        lines.append(f"| {k} | {v['p50']:.2f} | {v['p90']:.2f} | {v['p99']:.2f} | {v['mean']:.2f} | {counts[k]} |")
    base = res["through gateway"]["p50"]
    lines += ["", "Derived (p50):", "",
              f"- Gateway overhead vs direct: **{res['gateway overhead (p50 delta)']['p50']:.2f} ms**",
              f"- Added by failover on HTTP 503: **{res['failover: primary HTTP 503']['p50'] - base:.2f} ms**",
              (f"- Added by failover on connection refused: "
              f"**{res['failover: primary killed (conn refused)']['p50'] - base:.2f} ms**"),
              f"- Added once breaker is open: **{res['breaker open (primary skipped)']['p50'] - base:.2f} ms**",
              ("- Hung upstream: failover time is bounded by `timeout_s` "
              f"(measured p50 {res['failover: primary hangs (timeout_s=0.25)']['p50']:.0f} ms at 250 ms timeout)."),
              ""]
    Path(a.out).write_text("\n".join(lines))
    print("\n".join(lines))


def _hardware() -> str:
    cpu = platform.processor() or "unknown CPU"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem = ""
    try:
        kb = int(Path("/proc/meminfo").read_text().split()[1])
        mem = f", {kb / 1024 / 1024:.0f} GiB RAM"
    except (OSError, ValueError, IndexError):
        pass
    return (f"{cpu}, {os.cpu_count()} vCPU{mem}, {platform.system()} {platform.release()}, "
            f"Python {platform.python_version()}")


if __name__ == "__main__":
    asyncio.run(main())
