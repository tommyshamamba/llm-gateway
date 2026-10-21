"""OpenAI-compatible mock provider with fault injection.

Used by tests, benchmarks, docker-compose demo and CI. Faults are set per
instance via ``POST /_faults``:

    {"mode": "ok" | "error" | "hang" | "reset", "status": 503,
     "latency_ms": 0, "error_rate": 0.0}

Model behaviour (deterministic, for the eval harness):
  * ``mock-smart``: answers simple arithmetic and echo prompts correctly
  * ``mock-sloppy``: same, but wrong on every third arithmetic question
  * anything else: echoes the last user message
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

_ARITH = re.compile(r"(-?\d+)\s*([+\-*x×])\s*(-?\d+)")
_REPEAT = re.compile(r"repeat(?: back)?(?: exactly)?:\s*(.+)", re.IGNORECASE | re.DOTALL)


def answer(model: str, prompt: str) -> str:
    m = _ARITH.search(prompt)
    if m and model.startswith("mock-"):
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        v = a + b if op == "+" else a - b if op == "-" else a * b
        if model == "mock-sloppy" and (a + b) % 3 == 0:
            v += 1
        return str(v)
    r = _REPEAT.search(prompt)
    if r and model.startswith("mock-"):
        return r.group(1).strip()
    if "json" in prompt.lower() and model.startswith("mock-"):
        return json.dumps({"ok": True, "model": model}) if model == "mock-smart" else "ok: true"
    return prompt


def create_mock_app() -> FastAPI:
    app = FastAPI(title="mock-llm-provider")
    app.state.faults = {"mode": "ok", "status": 503, "latency_ms": 0.0, "error_rate": 0.0}
    app.state.calls = 0

    @app.post("/_faults")
    async def set_faults(req: Request):
        app.state.faults.update(await req.json())
        return app.state.faults

    @app.get("/_stats")
    async def stats():
        return {"calls": app.state.calls, "faults": app.state.faults}

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        app.state.calls += 1
        f = app.state.faults
        if f["latency_ms"]:
            await asyncio.sleep(f["latency_ms"] / 1000)
        if f["mode"] == "hang":
            await asyncio.sleep(3600)
        failing = f["mode"] == "error" or (f["error_rate"] and random.random() < f["error_rate"])
        if failing:
            return JSONResponse({"error": {"message": "injected fault"}}, status_code=int(f["status"]))
        body = await req.json()
        model = body.get("model", "mock")
        prompt = next((m["content"] for m in reversed(body["messages"]) if m["role"] == "user"), "")
        text = answer(model, prompt if isinstance(prompt, str) else json.dumps(prompt))
        pt = sum(len(str(m.get("content", ""))) for m in body["messages"]) // 4 + 1
        ct = len(text) // 4 + 1
        cid = "mock-" + uuid.uuid4().hex[:12]
        if body.get("stream"):
            async def gen():
                words = text.split(" ")
                for i, w in enumerate(words):
                    d = {"content": (w if i == 0 else " " + w)}
                    if i == 0:
                        d["role"] = "assistant"
                    yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk",
                                                 "created": int(time.time()), "model": model,
                                                 "choices": [{"index": 0, "delta": d, "finish_reason": None}]}) + "\n\n"
                    if f["mode"] == "reset" and i == 0:
                        raise RuntimeError("injected mid-stream reset")
                yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "model": model,
                                             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
                yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "model": model,
                                             "choices": [], "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                                                                      "total_tokens": pt + ct}}) + "\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return {
            "id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
        }

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser("llm-mock-provider")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    a = ap.parse_args()
    uvicorn.run(create_mock_app(), host=a.host, port=a.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
