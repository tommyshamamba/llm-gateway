"""Prometheus metrics (own registry so tests can create many apps)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            "llm_gateway_requests_total", "Client requests", ["route", "status"], registry=r
        )
        self.upstream = Counter(
            "llm_gateway_upstream_requests_total",
            "Upstream attempts",
            ["provider", "outcome"],
            registry=r,
        )
        self.latency = Histogram(
            "llm_gateway_request_seconds",
            "End-to-end request latency",
            ["route", "stream"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
            registry=r,
        )
        self.upstream_latency = Histogram(
            "llm_gateway_upstream_seconds",
            "Upstream attempt latency",
            ["provider"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
            registry=r,
        )
        self.failovers = Counter(
            "llm_gateway_failovers_total", "Requests served by a non-primary target", ["route"], registry=r
        )
        self.tokens = Counter(
            "llm_gateway_tokens_total", "Tokens", ["provider", "model", "kind"], registry=r
        )
        self.cost = Counter("llm_gateway_cost_usd_total", "Estimated cost (USD)", ["provider", "model"], registry=r)
        self.cache = Counter("llm_gateway_cache_total", "Cache lookups", ["result"], registry=r)
        self.rate_limited = Counter("llm_gateway_rate_limited_total", "429s", ["key"], registry=r)
        self.breaker_state = Gauge(
            "llm_gateway_breaker_state", "0=closed 1=half_open 2=open", ["provider"], registry=r
        )
