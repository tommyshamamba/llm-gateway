import json

from conftest import chat


async def test_completion_openai_shape(gw):
    r = await gw.post("/v1/chat/completions", json=chat("What is 2 + 2?"))
    assert r.status_code == 200
    d = r.json()
    assert d["object"] == "chat.completion"
    assert d["model"] == "smart"
    assert d["choices"][0]["message"]["content"] == "4"
    assert d["x_gateway"] == {"provider": "primary", "model": "mock-smart", "attempts": 1, "failover": False}
    assert r.headers["x-provider"] == "primary"


async def test_models_list(gw):
    r = await gw.get("/v1/models")
    assert {m["id"] for m in r.json()["data"]} == {"smart", "sloppy"}


async def test_auth_required(gw):
    r = await gw.post("/v1/chat/completions", json=chat("hi"), headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


async def test_unknown_model_and_bad_body(gw):
    assert (await gw.post("/v1/chat/completions", json=chat("hi", model="nope"))).status_code == 404
    assert (await gw.post("/v1/chat/completions", json={"model": "smart", "messages": []})).status_code == 400


async def test_token_and_cost_accounting(gw):
    r = await gw.post("/v1/chat/completions", json=chat("Repeat exactly: hello world"))
    u = r.json()["usage"]
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
    assert u["cost_usd"] == round((u["prompt_tokens"] * 1.0 + u["completion_tokens"] * 2.0) / 1e6, 8)
    led = (await gw.get("/v1/usage")).json()
    assert led["key"] == "alice"
    assert led["usage"]["requests"] == 1
    assert led["usage"]["prompt_tokens"] == u["prompt_tokens"]


async def test_cache_hit_on_deterministic_request(gw):
    body = chat("What is 3 + 4?", temperature=0)
    a = await gw.post("/v1/chat/completions", json=body)
    b = await gw.post("/v1/chat/completions", json=body)
    assert a.headers.get("x-cache") is None
    assert b.headers["x-cache"] == "hit"
    assert b.json()["choices"] == a.json()["choices"]
    assert b.json()["usage"]["cost_usd"] == 0.0
    assert gw.upstream.hits["primary"] == 1


async def test_no_cache_for_sampled_request(gw):
    body = chat("What is 3 + 4?", temperature=0.7)
    await gw.post("/v1/chat/completions", json=body)
    await gw.post("/v1/chat/completions", json=body)
    assert gw.upstream.hits["primary"] == 2


async def test_rate_limit_per_key(gw):
    h = {"Authorization": "Bearer test-slow-not-a-secret"}
    codes = [(await gw.post("/v1/chat/completions", json=chat("hi"), headers=h)).status_code for _ in range(4)]
    assert codes[:2] == [200, 200]
    assert codes[2] == 429
    # other keys are unaffected
    assert (await gw.post("/v1/chat/completions", json=chat("hi"))).status_code == 200


async def test_streaming_sse(gw):
    async with gw.stream("POST", "/v1/chat/completions",
                         json=chat("Repeat exactly: one two three", stream=True)) as r:
        assert r.status_code == 200
        events = [line[6:] async for line in r.aiter_lines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join((c["choices"][0]["delta"].get("content") or "") for c in chunks if c["choices"])
    assert text == "one two three"
    final = chunks[-1]
    assert final["usage"]["completion_tokens"] > 0
    assert final["x_gateway"]["provider"] == "primary"


async def test_metrics_exposed(gw):
    await gw.post("/v1/chat/completions", json=chat("hi"))
    m = (await gw.get("/metrics")).text
    assert 'llm_gateway_requests_total{route="smart",status="200"} 1.0' in m
    assert "llm_gateway_tokens_total" in m
    assert "llm_gateway_cost_usd_total" in m
