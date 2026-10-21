from __future__ import annotations

import httpx
import pytest

from llm_gateway.app import create_app
from llm_gateway.config import Settings
from llm_gateway.mock_server import create_mock_app


class Router(httpx.AsyncBaseTransport):
    """Dispatch upstream calls by host to in-process mock apps; 'down' hosts refuse."""

    def __init__(self, apps: dict[str, object]) -> None:
        self.apps = {h: httpx.ASGITransport(app=a) for h, a in apps.items()}
        self.down: set[str] = set()
        self.timeouts: set[str] = set()
        self.hits: dict[str, int] = {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        self.hits[host] = self.hits.get(host, 0) + 1
        if host in self.down or host not in self.apps:
            raise httpx.ConnectError("connection refused", request=request)
        if host in self.timeouts:
            raise httpx.ReadTimeout("read timeout", request=request)
        return await self.apps[host].handle_async_request(request)


def make_settings(**over) -> Settings:
    base = {
        "require_auth": True,
        "api_keys": [{"name": "alice", "key_env": "TEST_KEY_ALICE", "rpm": 6000, "burst": 1000},
                     {"name": "slow", "key_env": "TEST_KEY_SLOW", "rpm": 60, "burst": 2}],
        "providers": [
            {"name": "primary", "type": "openai", "base_url": "http://primary/v1", "timeout_s": 0.5,
             "pricing": {"mock-smart": {"input": 1.0, "output": 2.0}}},
            {"name": "secondary", "type": "openai", "base_url": "http://secondary/v1", "timeout_s": 0.5,
             "pricing": {"mock-smart": {"input": 3.0, "output": 6.0}}},
        ],
        "routes": {
            "smart": {"targets": [{"provider": "primary", "model": "mock-smart"},
                                  {"provider": "secondary", "model": "mock-smart"}]},
            "sloppy": {"targets": [{"provider": "primary", "model": "mock-sloppy"}]},
        },
        "breaker": {"failure_threshold": 2, "reset_timeout_s": 0.3},
        "retry": {"attempts_per_provider": 2, "backoff_base_s": 0.001, "backoff_max_s": 0.002},
        "cache": {"enabled": True, "ttl_s": 60, "max_entries": 100},
    }
    base.update(over)
    return Settings.model_validate(base)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("TEST_KEY_ALICE", "test-alice-not-a-secret")
    monkeypatch.setenv("TEST_KEY_SLOW", "test-slow-not-a-secret")


@pytest.fixture
def mocks():
    return {"primary": create_mock_app(), "secondary": create_mock_app()}


@pytest.fixture
async def gw(env, mocks):
    transport = Router(mocks)
    app = create_app(make_settings(), transport=transport)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        c.headers["Authorization"] = "Bearer test-alice-not-a-secret"
        c.upstream = transport
        c.mocks = mocks
        c.app = app
        yield c
    await app.state.gateway.client.aclose()


def chat(content: str, model: str = "smart", **kw) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": content}], **kw}
