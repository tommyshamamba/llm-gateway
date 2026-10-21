"""Markdown reports and regression comparison between two eval runs."""

from __future__ import annotations

from typing import Any


def to_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Eval report: {report['suite']}", "", f"Generated {report['generated_at']}", "",
             "| Model | Score | Passed | Errors | p50 ms | p95 ms | Tokens (in/out) | Cost USD |",
             "|---|---|---|---|---|---|---|---|"]
    for m, s in sorted(report["models"].items(), key=lambda kv: -kv[1]["score"]):
        lines.append(f"| {m} | {s['score']:.2%} | {s['passed']}/{s['cases']} | {s['errors']} | "
                     f"{s['latency_ms_p50']} | {s['latency_ms_p95']} | {s['prompt_tokens']}/"
                     f"{s['completion_tokens']} | {s['cost_usd']:.6f} |")
    models = list(report["models"])
    if len(models) > 1:
        lines += ["", "## Per-case disagreements", "", "| Case | " + " | ".join(models) + " |",
                  "|---|" + "---|" * len(models)]
        by_case: dict[str, dict[str, float]] = {}
        for r in report["results"]:
            by_case.setdefault(r["case_id"], {})[r["model"]] = r["score"]
        for cid, scores in sorted(by_case.items()):
            if len(set(scores.values())) > 1:
                lines.append(f"| {cid} | " + " | ".join(f"{scores.get(m, 0):.2f}" for m in models) + " |")
    return "\n".join(lines) + "\n"


def compare(baseline: dict[str, Any], candidate: dict[str, Any], max_drop: float = 0.0) -> dict[str, Any]:
    """Per-model score delta and per-case regressions (pass in baseline, fail in candidate)."""
    base_cases = {(r["model"], r["case_id"]): r["score"] for r in baseline["results"]}
    out: dict[str, Any] = {"suite": candidate["suite"], "models": {}, "regressed": False}
    for m, s in candidate["models"].items():
        b = baseline["models"].get(m)
        if b is None:
            continue
        regressions = sorted(
            r["case_id"] for r in candidate["results"]
            if r["model"] == m and base_cases.get((m, r["case_id"]), 0) > r["score"]
        )
        delta = round(s["score"] - b["score"], 4)
        bad = delta < -max_drop
        out["models"][m] = {"baseline": b["score"], "candidate": s["score"], "delta": delta,
                            "regressed_cases": regressions, "regressed": bad}
        out["regressed"] |= bad
    return out


def compare_markdown(cmp: dict[str, Any]) -> str:
    lines = [f"# Regression report: {cmp['suite']}", "",
             "| Model | Baseline | Candidate | Delta | Regressed cases | Status |", "|---|---|---|---|---|---|"]
    for m, s in cmp["models"].items():
        lines.append(f"| {m} | {s['baseline']:.2%} | {s['candidate']:.2%} | {s['delta']:+.2%} | "
                     f"{', '.join(s['regressed_cases']) or '-'} | {'FAIL' if s['regressed'] else 'ok'} |")
    return "\n".join(lines) + "\n"
