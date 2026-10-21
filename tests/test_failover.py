"""Fault-injection tests: upstream errors, refused connections, timeouts,
circuit breaker open/half-open/close, and streaming failover."""

import asyncio
import json

from conftest import chat


async def faults(gw, name, **f):
    gw.mocks[name].state.faults.update(f)


async def test_failover_on_5xx(gw):
    await faults(gw, "primary", mode="error", status=503)
    r = await gw.post("/v1/chat/completions", json=chat("What is 5 + 5?"))
    assert r.status_code == 200
    g = r.json()["x_gateway"]
    assert g["provider"] == "secondary" and g["failover"] is True
    assert g["attempts"] == 3  # 2 retries on primary, then secondary
    assert r.json()["choices"][0]["message"]["content"] == "10"


async def test_failover_on_connection_refused(gw):
    gw.upstream.down.add("primary")
    r = await gw.post("/v1/chat/completions", json=chat("hi"))
    assert r.status_code == 200 and r.json()["x_gateway"]["provider"] == "secondary"


async def test_failover_on_timeout(gw):
    gw.upstream.timeouts.add("primary")
    r = await gw.post("/v1/chat/completions", json=chat("hi"))
    assert r.status_code == 200 and r.json()["x_gateway"]["provider"] == "secondary"


async def test_failover_on_429(gw):
    await faults(gw, "primary", mode="error", status=429)
    r = await gw.post("/v1/chat/completions", json=chat("hi"))
    assert r.json()["x_gateway"]["provider"] == "secondary"


async def test_client_error_is_not_failed_over(gw):
    await faults(gw, "primary", mode="error", status=400)
    r = await gw.post("/v1/chat/completions", json=chat("hi"))
    assert r.status_code == 400
    assert gw.upstream.hits.get("secondary", 0) == 0


async def test_all_targets_down_returns_503(gw):
    gw.upstream.down |= {"primary", "secondary"}
    r = await gw.post("/v1/chat/completions", json=chat("hi"))
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "upstream_unavailable"


async def test_breaker_opens_then_skips_then_recovers(gw):
    await faults(gw, "primary", mode="error", status=500)
    await gw.post("/v1/chat/completions", json=chat("hi"))  # 2 failures -> threshold 2 -> open
    calls_after_trip = gw.mocks["primary"].state.calls
    assert calls_after_trip == 2
    for _ in range(5):
        r = await gw.post("/v1/chat/completions", json=chat("hi"))
        assert r.json()["x_gateway"] == {"provider": "secondary", "model": "mock-smart",
                                         "attempts": 1, "failover": True}
    assert gw.mocks["primary"].state.calls == calls_after_trip  # open breaker: no traffic
    ready = (await gw.get("/readyz")).json()
    assert ready["breakers"]["primary"] == "open"

    await faults(gw, "primary", mode="ok")
    await asyncio.sleep(0.35)  # reset_timeout_s=0.3 -> half-open, one probe
    r = await gw.post("/v1/chat/completions", json=chat("hi"))
    assert r.json()["x_gateway"]["provider"] == "primary"
    assert (await gw.get("/readyz")).json()["breakers"]["primary"] == "closed"


async def test_half_open_probe_failure_reopens(gw):
    await faults(gw, "primary", mode="error", status=500)
    await gw.post("/v1/chat/completions", json=chat("hi"))
    await asyncio.sleep(0.35)
    before = gw.mocks["primary"].state.calls
    await gw.post("/v1/chat/completions", json=chat("hi"))
    assert gw.mocks["primary"].state.calls == before + 1  # exactly one probe
    assert (await gw.get("/readyz")).json()["breakers"]["primary"] == "open"


async def test_intermittent_errors_still_served(gw):
    await faults(gw, "primary", error_rate=0.5, status=502)
    codes = [(await gw.post("/v1/chat/completions", json=chat(f"n{i}"))).status_code for i in range(40)]
    assert codes == [200] * 40


async def test_streaming_failover_before_first_chunk(gw):
    await faults(gw, "primary", mode="error", status=503)
    async with gw.stream("POST", "/v1/chat/completions", json=chat("Repeat exactly: a b", stream=True)) as r:
        assert r.status_code == 200
        assert r.headers["x-provider"] == "secondary"
        events = [line[6:] async for line in r.aiter_lines() if line.startswith("data: ")]
    final = json.loads(events[-2])
    assert final["x_gateway"]["failover"] is True


async def test_failover_metric_counted(gw):
    gw.upstream.down.add("primary")
    await gw.post("/v1/chat/completions", json=chat("hi"))
    m = (await gw.get("/metrics")).text
    assert 'llm_gateway_failovers_total{route="smart"} 1.0' in m
    assert 'llm_gateway_upstream_requests_total{outcome="error",provider="primary"} 2.0' in m
