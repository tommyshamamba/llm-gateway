"""Run the gateway: ``python -m llm_gateway`` (or ``llm-gateway``)."""

from __future__ import annotations

import argparse
import os

import uvicorn

from .app import create_app
from .config import load_settings
from .logging import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser("llm-gateway")
    ap.add_argument("--config", default=os.environ.get("LLM_GATEWAY_CONFIG", "config/gateway.yaml"))
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    ap.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    a = ap.parse_args()
    setup_logging(a.log_level)
    app = create_app(load_settings(a.config))
    uvicorn.run(app, host=a.host, port=a.port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
