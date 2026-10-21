"""Gateway configuration (YAML file + environment variables).

API keys are never stored in the config file: each provider names the
environment variable that holds its key (``api_key_env``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class Pricing(BaseModel):
    """USD per 1M tokens."""

    input: float = 0.0
    output: float = 0.0


class ProviderConfig(BaseModel):
    name: str
    type: Literal["openai", "anthropic", "ollama"]
    base_url: str
    api_key_env: str | None = None
    timeout_s: float = 30.0
    # upstream model name -> price
    pricing: dict[str, Pricing] = Field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


class Target(BaseModel):
    provider: str
    model: str


class Route(BaseModel):
    """A public model alias mapped to an ordered fallback chain."""

    targets: list[Target]


class BreakerConfig(BaseModel):
    failure_threshold: int = 3
    reset_timeout_s: float = 15.0


class RetryConfig(BaseModel):
    attempts_per_provider: int = 2
    backoff_base_s: float = 0.05
    backoff_max_s: float = 1.0


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl_s: float = 300.0
    max_entries: int = 10_000


class RateLimitConfig(BaseModel):
    default_rpm: int = 600
    default_burst: int = 60


class ApiKeyConfig(BaseModel):
    """Gateway client key. ``key_env`` names the env var holding the secret."""

    name: str
    key_env: str
    rpm: int | None = None
    burst: int | None = None


class Settings(BaseModel):
    providers: list[ProviderConfig]
    routes: dict[str, Route]
    breaker: BreakerConfig = BreakerConfig()
    retry: RetryConfig = RetryConfig()
    cache: CacheConfig = CacheConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    api_keys: list[ApiKeyConfig] = Field(default_factory=list)
    require_auth: bool = True

    def provider(self, name: str) -> ProviderConfig:
        for p in self.providers:
            if p.name == name:
                return p
        raise KeyError(name)


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    path = path or os.environ.get("LLM_GATEWAY_CONFIG", "config/gateway.yaml")
    data = yaml.safe_load(Path(path).read_text())
    return Settings.model_validate(data)
