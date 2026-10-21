# Design

## Goals

1. Clients use one OpenAI-compatible API regardless of which provider serves them.
2. A provider outage, rate limit or slowdown degrades to a fallback in milliseconds, not to an error.
3. Every request is attributable: tokens, cost, provider, attempts and latency, per client key.
4. Model changes are tested before rollout with repeatable eval suites.

Non-goals: tool-call translation across providers, embeddings, multi-tenant billing storage.

## Request path

1. **Auth.** `Authorization: Bearer <key>` is compared in constant time against keys loaded from env
   vars named in config. Each key has its own `rpm`/`burst`.
2. **Rate limit.** In-memory token bucket per key. Rejections return `429` with `Retry-After`.
3. **Cache.** Only for non-streaming requests that are deterministic (`temperature: 0`) or explicitly
   opt in (`cache: true`). The key is a SHA-256 of model alias + messages + sampling params. LRU + TTL.
4. **Route.** A public alias (`chat-default`) maps to an ordered list of `(provider, model)` targets.
5. **Attempt loop.** For each target: skip if its breaker is open; otherwise try up to
   `attempts_per_provider` times with jittered exponential backoff on retryable failures; then move on.
6. **Normalise + account.** Adapters return OpenAI shapes. Usage comes from the provider, or is
   estimated (~4 chars/token) when missing. Cost = tokens × per-model price from config.

## Failure model

| Failure | Classified as | Behaviour |
|---|---|---|
| Connect error / reset | retryable | retry, then next target |
| Timeout (`timeout_s`) | retryable | same; a hang costs exactly `timeout_s` |
| 408, 409, 425, 429, 5xx, 529 | retryable | same |
| 401, 403, 404 | non-retryable | no retry on this provider, fail over (likely misconfigured key/model) |
| 400, 413, 422 | client error | returned to caller, no failover (another provider would reject it too) |
| All targets fail | — | `503 upstream_unavailable`, errors logged |

**Circuit breaker** per provider: `failure_threshold` consecutive failures open it; while open, the
provider gets no traffic. After `reset_timeout_s` it goes half-open and admits **exactly one** probe;
success closes it, failure re-opens it. `/readyz` reports breaker states and returns 503 only if every
breaker is open. This is what keeps added latency near zero once a provider is known bad
(measured ~0.1 ms vs ~1–1.4 ms for a failover that still has to hit the dead upstream).

**Streaming.** The gateway fails over until the first chunk arrives. After that, the response is
committed to that provider; a mid-stream failure ends the stream with an error event rather than
silently splicing output from a different model. Accounting runs at the end of every stream.

## State and scaling

Rate-limit buckets, breaker state, cache and ledger are in-process. That is deliberate for a single
binary with no dependencies; with N replicas the effective per-key limit is up to N× the configured
value and each replica learns provider health independently (which is fine: breakers converge within
`failure_threshold` requests). The swap-in points for shared state are `RateLimiter`, `ResponseCache`
and `Ledger` (Redis / Postgres) without touching the router.

## Evals

Suites are YAML: cases with a prompt or messages, a scorer and an expected value. The runner calls
the gateway like any client (so it measures what users get, including failover), with `cache: false`.
Reports include score, pass count, errors, p50/p95 latency, tokens and cost per model, plus a
per-case disagreement table. `llm-eval compare` diffs two runs and exits non-zero when a model's score
drops more than `--max-drop`, so it can gate a model or prompt change in CI.

## Testing

All tests run against in-process mock providers (`llm_gateway.mock_server`) behind a transport that
can also simulate refused connections and timeouts. Fault-injection tests cover 5xx, 429, refused
connections, timeouts, intermittent 50% errors, breaker open/half-open/close, single-probe re-open,
client-error passthrough, total outage and streaming failover. No API keys are needed anywhere in CI.

## Benchmark method

`bench/bench.py` starts the gateway and two mocks as separate processes and sends sequential
requests over a keep-alive client: baseline direct to the mock, then through the gateway, then with
the primary returning 503, killed, behind an open breaker, and hung. Results are in
`bench/RESULTS.md` with the hardware they were measured on.

## Origin

The provider clients started as a small multi-provider "audit" helper in an older personal project.
This repository keeps only the general-purpose parts (provider adapters, caching, rate limiting) and
rebuilds them as a standalone gateway.
