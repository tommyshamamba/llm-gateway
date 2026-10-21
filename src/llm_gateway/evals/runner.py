"""Run a prompt suite against one or more models through any OpenAI-compatible endpoint."""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from .scorers import SCORERS


@dataclass
class Case:
    id: str
    messages: list[dict[str, Any]]
    scorer: str
    expected: Any = None
    tags: list[str] = field(default_factory=list)


@dataclass
class Suite:
    name: str
    cases: list[Case]
    defaults: dict[str, Any] = field(default_factory=dict)


def load_suite(path: str | Path) -> Suite:
    d = yaml.safe_load(Path(path).read_text())
    cases = []
    for c in d["cases"]:
        msgs = c.get("messages") or [{"role": "user", "content": c["prompt"]}]
        if d.get("system"):
            msgs = [{"role": "system", "content": d["system"]}, *msgs]
        if c["scorer"] not in SCORERS:
            raise ValueError(f"unknown scorer {c['scorer']!r} in case {c['id']}")
        cases.append(Case(c["id"], msgs, c["scorer"], c.get("expected"), c.get("tags", [])))
    return Suite(d["name"], cases, d.get("defaults", {"temperature": 0}))


@dataclass
class CaseResult:
    case_id: str
    model: str
    score: float
    output: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


async def _run_case(client: httpx.AsyncClient, url: str, model: str, case: Case, defaults: dict,
                    headers: dict) -> CaseResult:
    body = {"model": model, "messages": case.messages, **defaults, "cache": False}
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=body, headers=headers)
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            return CaseResult(case.id, model, 0.0, "", ms, error=f"HTTP {r.status_code}")
        d = r.json()
        out = d["choices"][0]["message"]["content"] or ""
        u = d.get("usage", {})
        return CaseResult(case.id, model, SCORERS[case.scorer](out, case.expected), out, ms,
                          u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("cost_usd", 0.0))
    except httpx.HTTPError as e:
        return CaseResult(case.id, model, 0.0, "", (time.perf_counter() - t0) * 1000, error=type(e).__name__)


async def run_suite(suite: Suite, models: list[str], base_url: str, api_key: str | None = None,
                    concurrency: int = 8, transport: httpx.AsyncBaseTransport | None = None) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=120, transport=transport) as client:
        async def one(m, c):
            async with sem:
                return await _run_case(client, url, m, c, suite.defaults, headers)
        results = await asyncio.gather(*(one(m, c) for m in models for c in suite.cases))
    return summarize(suite, models, results)


def summarize(suite: Suite, models: list[str], results: list[CaseResult]) -> dict[str, Any]:
    per_model = {}
    for m in models:
        rs = [r for r in results if r.model == m]
        lat = sorted(r.latency_ms for r in rs)
        per_model[m] = {
            "score": round(sum(r.score for r in rs) / len(rs), 4) if rs else 0.0,
            "passed": sum(r.score >= 1.0 for r in rs),
            "cases": len(rs),
            "errors": sum(r.error is not None for r in rs),
            "latency_ms_p50": round(statistics.median(lat), 2) if lat else 0.0,
            "latency_ms_p95": round(lat[max(0, int(len(lat) * 0.95) - 1)], 2) if lat else 0.0,
            "prompt_tokens": sum(r.prompt_tokens for r in rs),
            "completion_tokens": sum(r.completion_tokens for r in rs),
            "cost_usd": round(sum(r.cost_usd for r in rs), 6),
        }
    return {
        "suite": suite.name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": per_model,
        "results": [asdict(r) for r in results],
    }
