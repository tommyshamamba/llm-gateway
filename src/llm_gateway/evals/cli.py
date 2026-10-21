"""``llm-eval run`` / ``llm-eval compare``."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .report import compare, compare_markdown, to_markdown
from .runner import load_suite, run_suite


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("llm-eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a suite against one or more models")
    r.add_argument("--suite", required=True)
    r.add_argument("--models", required=True, help="comma-separated gateway model aliases")
    r.add_argument("--base-url", default=os.environ.get("LLM_GATEWAY_URL", "http://localhost:8080/v1"))
    r.add_argument("--api-key-env", default="LLM_GATEWAY_KEY")
    r.add_argument("--out", default="reports")
    r.add_argument("--concurrency", type=int, default=8)
    c = sub.add_parser("compare", help="regression report between two runs")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--max-drop", type=float, default=0.0, help="allowed score drop (0.02 = 2 points)")
    a = ap.parse_args(argv)

    if a.cmd == "run":
        suite = load_suite(a.suite)
        report = asyncio.run(run_suite(suite, a.models.split(","), a.base_url, os.environ.get(a.api_key_env),
                                       a.concurrency))
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        stem = out / suite.name
        stem.with_suffix(".json").write_text(json.dumps(report, indent=2))
        md = to_markdown(report)
        stem.with_suffix(".md").write_text(md)
        print(md)
        return 0

    cmp = compare(json.loads(Path(a.baseline).read_text()), json.loads(Path(a.candidate).read_text()),
                  a.max_drop)
    print(compare_markdown(cmp))
    return 1 if cmp["regressed"] else 0


if __name__ == "__main__":
    sys.exit(main())
