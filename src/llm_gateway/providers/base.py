"""Provider interface. Every adapter speaks OpenAI chat-completions shapes
internally: ``complete`` returns an OpenAI ``chat.completion`` dict and
``stream`` yields OpenAI ``chat.completion.chunk`` dicts."""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..config import ProviderConfig


class ProviderError(Exception):
    def __init__(self, message: str, *, status: int | None = None, retryable: bool = True) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def raise_for_status(provider: str, resp: httpx.Response) -> None:
    if resp.status_code < 400:
        return
    retryable = resp.status_code in RETRYABLE_STATUS
    raise ProviderError(
        f"{provider}: upstream HTTP {resp.status_code}", status=resp.status_code, retryable=retryable
    )


def new_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def completion(model: str, content: str, finish: str, prompt_t: int, completion_t: int) -> dict[str, Any]:
    return {
        "id": new_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_t,
            "completion_tokens": completion_t,
            "total_tokens": prompt_t + completion_t,
        },
    }


def chunk(cid: str, model: str, delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


class Provider(ABC):
    def __init__(self, cfg: ProviderConfig, client: httpx.AsyncClient) -> None:
        self.cfg = cfg
        self.name = cfg.name
        self.client = client

    @abstractmethod
    async def complete(self, model: str, body: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def stream(self, model: str, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]: ...

    async def _post(self, url: str, **kw: Any) -> httpx.Response:
        try:
            resp = await self.client.post(url, timeout=self.cfg.timeout_s, **kw)
        except httpx.TimeoutException as e:
            raise ProviderError(f"{self.name}: timeout", retryable=True) from e
        except httpx.TransportError as e:
            raise ProviderError(f"{self.name}: transport error {type(e).__name__}", retryable=True) from e
        raise_for_status(self.name, resp)
        return resp


def build_provider(cfg: ProviderConfig, client: httpx.AsyncClient) -> Provider:
    from .anthropic import AnthropicProvider
    from .ollama import OllamaProvider
    from .openai_compat import OpenAICompatProvider

    return {"openai": OpenAICompatProvider, "anthropic": AnthropicProvider, "ollama": OllamaProvider}[cfg.type](
        cfg, client
    )
