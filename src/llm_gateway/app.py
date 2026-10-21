"""FastAPI application: OpenAI-compatible /v1/chat/completions and /v1/models."""

from __future__ import annotations

import hmac
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import __version__
from .accounting import Ledger, cost_usd, estimate_tokens
from .cache import ResponseCache, cache_key, cacheable
from .config import Settings, load_settings
from .logging import log
from .metrics import Metrics
from .providers import ProviderError, build_provider
from .ratelimit import RateLimiter
from .router import ClientRequestError, NoHealthyProvider, Router


def _err(status: int, message: str, typ: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": typ}}, status_code=status, headers=headers)


class Gateway:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        self.metrics = Metrics()
        self.client = httpx.AsyncClient(
            transport=transport, limits=httpx.Limits(max_connections=512, max_keepalive_connections=128)
        )
        providers = {p.name: build_provider(p, self.client) for p in settings.providers}
        self.router = Router(settings, providers, self.metrics)
        self.cache = ResponseCache(settings.cache.ttl_s, settings.cache.max_entries)
        self.limiter = RateLimiter()
        self.ledger = Ledger()
        self._keys = []  # (name, secret, rpm, burst)
        rl = settings.rate_limit
        import os

        for k in settings.api_keys:
            secret = os.environ.get(k.key_env)
            if secret:
                self._keys.append((k.name, secret, k.rpm or rl.default_rpm, k.burst or rl.default_burst))

    def authenticate(self, request: Request) -> tuple[str, int, int] | None:
        rl = self.settings.rate_limit
        auth = request.headers.get("authorization", "")
        token = auth[7:] if auth.lower().startswith("bearer ") else ""
        for name, secret, rpm, burst in self._keys:
            if token and hmac.compare_digest(token, secret):
                return name, rpm, burst
        if not self.settings.require_auth:
            return "anonymous", rl.default_rpm, rl.default_burst
        return None

    def account(self, key: str, provider: str, model: str, usage: dict | None, prompt_text: str,
                completion_text: str, cache_hit: bool = False) -> dict[str, Any]:
        usage = usage or {}
        p = usage.get("prompt_tokens") or estimate_tokens(prompt_text)
        c = usage.get("completion_tokens")
        if c is None:
            c = estimate_tokens(completion_text)
        pricing = self.settings.provider(provider).pricing.get(model)
        cost = 0.0 if cache_hit else cost_usd(pricing, p, c)
        self.ledger.record(key, p, c, cost, cache_hit)
        if not cache_hit:
            self.metrics.tokens.labels(provider, model, "prompt").inc(p)
            self.metrics.tokens.labels(provider, model, "completion").inc(c)
            self.metrics.cost.labels(provider, model).inc(cost)
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c, "cost_usd": round(cost, 8)}


def _prompt_text(body: dict[str, Any]) -> str:
    out = []
    for m in body.get("messages", []):
        c = m.get("content")
        out.append(c if isinstance(c, str) else json.dumps(c))
    return "\n".join(out)


def create_app(settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    settings = settings or load_settings()
    gw = Gateway(settings, transport)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await gw.client.aclose()

    app = FastAPI(title="llm-gateway", version=__version__, lifespan=lifespan)
    app.state.gateway = gw

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        states = {n: b.state.name.lower() for n, b in gw.router.breakers.items()}
        ready = any(s != "open" for s in states.values())
        return JSONResponse({"ready": ready, "breakers": states}, status_code=200 if ready else 503)

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(generate_latest(gw.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    async def models():
        return {"object": "list",
                "data": [{"id": r, "object": "model", "owned_by": "llm-gateway"} for r in settings.routes]}

    @app.get("/v1/usage")
    async def usage(request: Request):
        ident = gw.authenticate(request)
        if ident is None:
            return _err(401, "invalid API key", "authentication_error")
        return {"key": ident[0], "usage": gw.ledger.snapshot(ident[0])}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        t0 = time.perf_counter()
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        ident = gw.authenticate(request)
        if ident is None:
            return _err(401, "invalid API key", "authentication_error")
        key, rpm, burst = ident
        allowed, retry_after = gw.limiter.check(key, rpm, burst)
        if not allowed:
            gw.metrics.rate_limited.labels(key).inc()
            return _err(429, "rate limit exceeded", "rate_limit_error",
                        {"Retry-After": str(max(1, int(retry_after + 0.999)))})
        try:
            body = await request.json()
        except ValueError:
            return _err(400, "body must be JSON", "invalid_request_error")
        route = body.get("model")
        if route not in settings.routes:
            return _err(404, f"unknown model {route!r}", "invalid_request_error")
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            return _err(400, "messages must be a non-empty list", "invalid_request_error")

        stream = bool(body.get("stream"))
        hdrs = {"x-request-id": request_id}
        logx = {"request_id": request_id, "key": key, "route": route, "stream": stream}

        if not stream and settings.cache.enabled and cacheable(body):
            ck = cache_key(route, body)
            hit = gw.cache.get(ck)
            if hit is not None:
                gw.metrics.cache.labels("hit").inc()
                resp = dict(hit)
                resp["usage"] = gw.account(key, hit["x_gateway"]["provider"], hit["x_gateway"]["model"],
                                           hit.get("usage"), "", "", cache_hit=True)
                resp["x_gateway"] = {**hit["x_gateway"], "cache": "hit"}
                gw.metrics.requests.labels(route, "200").inc()
                gw.metrics.latency.labels(route, "false").observe(time.perf_counter() - t0)
                log.info("request", extra={**logx, "status": 200, "cache": "hit",
                                           "ms": round((time.perf_counter() - t0) * 1000, 3)})
                return JSONResponse(resp, headers={**hdrs, "x-cache": "hit"})
            gw.metrics.cache.labels("miss").inc()
        else:
            ck = None

        try:
            if stream:
                gen, out = await gw.router.stream(route, body)
            else:
                resp, out = await gw.router.complete(route, body)
        except ClientRequestError as e:
            gw.metrics.requests.labels(route, str(e.status)).inc()
            return _err(e.status, str(e), "invalid_request_error", hdrs)
        except NoHealthyProvider as e:
            gw.metrics.requests.labels(route, "503").inc()
            log.error("no_healthy_provider", extra={**logx, "errors": e.errors})
            return _err(503, "all upstream providers failed", "upstream_unavailable", hdrs)

        if out.failover:
            gw.metrics.failovers.labels(route).inc()
        meta = {"provider": out.target.provider, "model": out.target.model, "attempts": out.attempts,
                "failover": out.failover}
        hdrs.update({"x-provider": out.target.provider, "x-attempts": str(out.attempts)})

        if not stream:
            text = (resp.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            resp["usage"] = gw.account(key, out.target.provider, out.target.model, resp.get("usage"),
                                       _prompt_text(body), text)
            resp["model"] = route
            resp["x_gateway"] = meta
            if ck is not None:
                gw.cache.put(ck, resp)
            gw.metrics.requests.labels(route, "200").inc()
            gw.metrics.latency.labels(route, "false").observe(time.perf_counter() - t0)
            log.info("request", extra={**logx, **meta, "status": 200, "usage": resp["usage"],
                                       "ms": round((time.perf_counter() - t0) * 1000, 3)})
            return JSONResponse(resp, headers=hdrs)

        async def sse():
            parts: list[str] = []
            usage = None
            status = "200"
            try:
                async for c in gen:
                    c["model"] = route
                    if c.get("usage"):
                        usage = c["usage"]
                    for ch in c.get("choices") or []:
                        parts.append((ch.get("delta") or {}).get("content") or "")
                    if not c.get("choices") and c.get("usage"):
                        continue  # emitted once below with gateway accounting
                    yield f"data: {json.dumps(c)}\n\n"
            except ProviderError as e:
                status = "502"
                yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'upstream_stream_error'}})}\n\n"
            acct = gw.account(key, out.target.provider, out.target.model, usage, _prompt_text(body), "".join(parts))
            yield f"data: {json.dumps({'object': 'chat.completion.chunk', 'model': route, 'choices': [], 'usage': acct, 'x_gateway': meta})}\n\n"
            yield "data: [DONE]\n\n"
            gw.metrics.requests.labels(route, status).inc()
            gw.metrics.latency.labels(route, "true").observe(time.perf_counter() - t0)
            log.info("request", extra={**logx, **meta, "status": int(status), "usage": acct,
                                       "ms": round((time.perf_counter() - t0) * 1000, 3)})

        return StreamingResponse(sse(), media_type="text/event-stream", headers=hdrs)

    return app
