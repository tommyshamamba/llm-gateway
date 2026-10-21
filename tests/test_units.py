import json

import httpx

from llm_gateway.breaker import CircuitBreaker, State
from llm_gateway.cache import ResponseCache, cacheable
from llm_gateway.config import ProviderConfig
from llm_gateway.providers.anthropic import AnthropicProvider, to_anthropic
from llm_gateway.providers.ollama import OllamaProvider
from llm_gateway.ratelimit import RateLimiter


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_breaker_state_machine():
    c = Clock()
    b = CircuitBreaker(failure_threshold=2, reset_timeout_s=10, clock=c)
    b.record_failure()
    assert b.state is State.CLOSED
    b.record_failure()
    assert b.state is State.OPEN and not b.allow()
    c.t = 10
    assert b.state is State.HALF_OPEN
    assert b.allow() and not b.allow()  # single probe
    b.record_success()
    assert b.state is State.CLOSED


def test_rate_limiter_refills():
    c = Clock()
    rl = RateLimiter(clock=c)
    assert rl.check("k", rpm=60, burst=1) == (True, 0.0)
    ok, retry = rl.check("k", rpm=60, burst=1)
    assert not ok and abs(retry - 1.0) < 1e-9
    c.t = 1.0
    assert rl.check("k", rpm=60, burst=1)[0]


def test_cache_ttl_and_lru():
    c = Clock()
    cache = ResponseCache(ttl_s=5, max_entries=2, clock=c)
    cache.put("a", {"v": 1})
    cache.put("b", {"v": 2})
    cache.get("a")
    cache.put("c", {"v": 3})  # evicts b (least recently used)
    assert cache.get("b") is None and cache.get("a") == {"v": 1}
    c.t = 6
    assert cache.get("a") is None
    assert cacheable({"temperature": 0}) and not cacheable({"temperature": 0, "stream": True})
    assert not cacheable({"temperature": 0.5}) and cacheable({"temperature": 0.5, "cache": True})


def test_anthropic_request_translation():
    p = to_anthropic("claude", {"messages": [{"role": "system", "content": "be brief"},
                                             {"role": "user", "content": "hi"}], "stop": "END"}, False)
    assert p["system"] == "be brief" and p["messages"] == [{"role": "user", "content": "hi"}]
    assert p["max_tokens"] == 1024 and p["stop_sequences"] == ["END"]


async def test_anthropic_adapter_completion_and_stream():
    def handler(req: httpx.Request):
        assert req.headers["x-api-key"] == "dummy"
        body = json.loads(req.content)
        if body.get("stream"):
            ev = [
                {"type": "message_start", "message": {"usage": {"input_tokens": 7}}},
                {"type": "content_block_delta", "delta": {"text": "Hel"}},
                {"type": "content_block_delta", "delta": {"text": "lo"}},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
            ]
            return httpx.Response(200, text="".join(f"event: x\ndata: {json.dumps(e)}\n\n" for e in ev))
        return httpx.Response(200, json={"content": [{"type": "text", "text": "Hello"}], "stop_reason": "end_turn",
                                         "usage": {"input_tokens": 7, "output_tokens": 2}})

    import os
    os.environ["DUMMY_ANTHROPIC"] = "dummy"
    cfg = ProviderConfig(name="anthropic", type="anthropic", base_url="http://a", api_key_env="DUMMY_ANTHROPIC")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        p = AnthropicProvider(cfg, c)
        body = {"messages": [{"role": "user", "content": "hi"}]}
        d = await p.complete("claude", body)
        assert d["choices"][0]["message"]["content"] == "Hello" and d["usage"]["total_tokens"] == 9
        chunks = [x async for x in p.stream("claude", body)]
        assert "".join(x["choices"][0]["delta"].get("content", "") for x in chunks) == "Hello"
        assert chunks[-1]["usage"] == {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}


async def test_ollama_adapter_completion_and_stream():
    def handler(req: httpx.Request):
        body = json.loads(req.content)
        assert body["options"]["temperature"] == 0
        if body["stream"]:
            lines = [{"message": {"content": "Hi"}, "done": False},
                     {"message": {"content": "!"}, "done": True, "prompt_eval_count": 3, "eval_count": 2}]
            return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))
        return httpx.Response(200, json={"message": {"content": "Hi!"}, "prompt_eval_count": 3, "eval_count": 2})

    cfg = ProviderConfig(name="ollama", type="ollama", base_url="http://o")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        p = OllamaProvider(cfg, c)
        body = {"messages": [{"role": "user", "content": "hi"}], "temperature": 0}
        assert (await p.complete("llama", body))["choices"][0]["message"]["content"] == "Hi!"
        chunks = [x async for x in p.stream("llama", body)]
        assert "".join(x["choices"][0]["delta"]["content"] for x in chunks) == "Hi!"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
