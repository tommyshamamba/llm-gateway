"""Anthropic Messages API adapter (translated to/from OpenAI shapes)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .base import Provider, ProviderError, chunk, completion, new_id, raise_for_status

_FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls"}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if isinstance(p, dict))


def to_anthropic(model: str, body: dict[str, Any], stream: bool) -> dict[str, Any]:
    system = "\n".join(_text(m["content"]) for m in body["messages"] if m["role"] == "system")
    msgs = [
        {"role": m["role"], "content": _text(m["content"])}
        for m in body["messages"]
        if m["role"] in ("user", "assistant")
    ]
    p: dict[str, Any] = {"model": model, "messages": msgs, "max_tokens": body.get("max_tokens") or 1024}
    if system:
        p["system"] = system
    for k in ("temperature", "top_p"):
        if k in body:
            p[k] = body[k]
    if body.get("stop"):
        stop = body["stop"]
        p["stop_sequences"] = [stop] if isinstance(stop, str) else stop
    if stream:
        p["stream"] = True
    return p


class AnthropicProvider(Provider):
    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.cfg.api_key or "",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    async def complete(self, model: str, body: dict[str, Any]) -> dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + "/v1/messages"
        resp = await self._post(url, headers=self._headers(), json=to_anthropic(model, body, False))
        d = resp.json()
        u = d.get("usage", {})
        return completion(
            model,
            _text(d.get("content")),
            _FINISH.get(d.get("stop_reason"), "stop"),
            u.get("input_tokens", 0),
            u.get("output_tokens", 0),
        )

    async def stream(self, model: str, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        url = self.cfg.base_url.rstrip("/") + "/v1/messages"
        req = self.client.build_request(
            "POST", url, headers=self._headers(), json=to_anthropic(model, body, True), timeout=self.cfg.timeout_s
        )
        try:
            resp = await self.client.send(req, stream=True)
        except httpx.TransportError as e:
            raise ProviderError(f"{self.name}: transport error {type(e).__name__}") from e
        cid, in_t, out_t = new_id(), 0, 0
        try:
            if resp.status_code >= 400:
                await resp.aread()
                raise_for_status(self.name, resp)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                ev = json.loads(line[5:].strip())
                t = ev.get("type")
                if t == "message_start":
                    in_t = ev["message"].get("usage", {}).get("input_tokens", 0)
                    yield chunk(cid, model, {"role": "assistant", "content": ""})
                elif t == "content_block_delta":
                    yield chunk(cid, model, {"content": ev["delta"].get("text", "")})
                elif t == "message_delta":
                    out_t = ev.get("usage", {}).get("output_tokens", out_t)
                    c = chunk(cid, model, {}, _FINISH.get(ev["delta"].get("stop_reason"), "stop"))
                    c["usage"] = {"prompt_tokens": in_t, "completion_tokens": out_t,
                                  "total_tokens": in_t + out_t}
                    yield c
        except httpx.TransportError as e:
            raise ProviderError(f"{self.name}: stream interrupted") from e
        finally:
            await resp.aclose()
