"""Ollama native /api/chat adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .base import Provider, ProviderError, chunk, completion, new_id, raise_for_status


def to_ollama(model: str, body: dict[str, Any], stream: bool) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    if "temperature" in body:
        opts["temperature"] = body["temperature"]
    if "top_p" in body:
        opts["top_p"] = body["top_p"]
    if body.get("max_tokens"):
        opts["num_predict"] = body["max_tokens"]
    if body.get("stop"):
        opts["stop"] = [body["stop"]] if isinstance(body["stop"], str) else body["stop"]
    p: dict[str, Any] = {"model": model, "messages": body["messages"], "stream": stream, "options": opts}
    if (body.get("response_format") or {}).get("type") == "json_object":
        p["format"] = "json"
    return p


class OllamaProvider(Provider):
    async def complete(self, model: str, body: dict[str, Any]) -> dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + "/api/chat"
        d = (await self._post(url, json=to_ollama(model, body, False))).json()
        return completion(
            model,
            d.get("message", {}).get("content", ""),
            "length" if d.get("done_reason") == "length" else "stop",
            d.get("prompt_eval_count", 0),
            d.get("eval_count", 0),
        )

    async def stream(self, model: str, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        url = self.cfg.base_url.rstrip("/") + "/api/chat"
        req = self.client.build_request("POST", url, json=to_ollama(model, body, True), timeout=self.cfg.timeout_s)
        try:
            resp = await self.client.send(req, stream=True)
        except httpx.TransportError as e:
            raise ProviderError(f"{self.name}: transport error {type(e).__name__}") from e
        cid, first = new_id(), True
        try:
            if resp.status_code >= 400:
                await resp.aread()
                raise_for_status(self.name, resp)
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                d = json.loads(line)
                delta = {"content": d.get("message", {}).get("content", "")}
                if first:
                    delta["role"] = "assistant"
                    first = False
                if d.get("done"):
                    c = chunk(cid, model, delta, "length" if d.get("done_reason") == "length" else "stop")
                    p, o = d.get("prompt_eval_count", 0), d.get("eval_count", 0)
                    c["usage"] = {"prompt_tokens": p, "completion_tokens": o, "total_tokens": p + o}
                    yield c
                else:
                    yield chunk(cid, model, delta)
        except httpx.TransportError as e:
            raise ProviderError(f"{self.name}: stream interrupted") from e
        finally:
            await resp.aclose()
