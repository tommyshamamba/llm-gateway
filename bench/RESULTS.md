# Benchmark results

Measured 2026-10-21 02:21 CDT with `python bench/bench.py --n 2000`.

**Hardware:** Intel(R) Xeon(R) Processor, 8 vCPU, 16 GiB RAM, Linux 6.12.94+, Python 3.12.15

Setup: gateway, primary mock and secondary mock run as three separate local processes (uvicorn, 1 worker each) on 127.0.0.1. Requests are sequential over a keep-alive client, cache disabled, retries=1 per provider. Mock providers answer instantly, so these numbers isolate gateway cost; real LLM latency (hundreds of ms+) dominates in practice.

| Scenario | p50 ms | p90 ms | p99 ms | mean ms | samples |
|---|---|---|---|---|---|
| direct to mock | 0.90 | 1.23 | 1.76 | 0.97 | 2000 |
| through gateway | 2.44 | 2.91 | 3.33 | 2.50 | 2000 |
| failover: primary HTTP 503 | 3.85 | 4.43 | 6.91 | 3.99 | 500 |
| failover: primary killed (conn refused) | 3.38 | 3.88 | 5.00 | 3.48 | 500 |
| breaker open (primary skipped) | 2.52 | 2.97 | 3.42 | 2.59 | 2000 |
| failover: primary hangs (timeout_s=0.25) | 256.37 | 257.29 | 259.61 | 256.53 | 20 |

Derived (p50):

- Gateway overhead vs direct: **1.54 ms**
- Added by failover on HTTP 503: **1.41 ms**
- Added by failover on connection refused: **0.94 ms**
- Added once breaker is open: **0.08 ms**
- Hung upstream: failover time is bounded by `timeout_s` (measured p50 256 ms at 250 ms timeout).
