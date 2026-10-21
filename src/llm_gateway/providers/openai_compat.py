"""OpenAI-compatible upstreams: OpenAI, DeepSeek, vLLM, the mock provider."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .base import Provider, ProviderError, raise_for_status

_PASSTHROUGH = ("messages", "temperature", "top_p", "max_tokens", "stop", "response_format", "tools",
                "tool_choice", "seed", "presence_penalty", "frequency_penalty", "n")


class OpenAICompatProvider(Provider):
    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    def _payload(self, model: str, body: dict[str, Any], stream: bool) -> dict[str, Any]:
        p = {k: body[k] for k in _PASSTHROUGH if k in body}
        p["model"] = model
        if stream:
            p["stream"] = True
            p["stream_options"] = {"include_usage": True}
        return p

    async def complete(self, model: str, body: dict[str, Any]) -> dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        resp = await self._post(url, headers=self._headers(), json=self._payload(model, body, False))
        try:
            return resp.json()
        except ValueError as e:
            raise ProviderError(f"{self.name}: invalid JSON", retryable=True) from e

    async def stream(self, model: str, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        req = self.client.build_request(
            "POST", url, headers=self._headers(), json=self._payload(model, body, True), timeout=self.cfg.timeout_s
        )
        try:
            resp = await self.client.send(req, stream=True)
        except httpx.TimeoutException as e:
            raise ProviderError(f"{self.name}: timeout") from e
        except httpx.TransportError as e:
            raise ProviderError(f"{self.name}: transport error {type(e).__name__}") from e
        try:
            if resp.status_code >= 400:
                await resp.aread()
                raise_for_status(self.name, resp)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                yield json.loads(data)
        except httpx.TransportError as e:
            raise ProviderError(f"{self.name}: stream interrupted") from e
        finally:
            await resp.aclose()
