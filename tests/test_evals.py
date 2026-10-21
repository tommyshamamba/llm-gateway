import httpx

from llm_gateway.evals.cli import main as eval_main
from llm_gateway.evals.report import compare, to_markdown
from llm_gateway.evals.runner import load_suite, run_suite
from llm_gateway.evals.scorers import contains, exact, json_valid, numeric, regex


def test_scorers():
    assert numeric("The answer is 1,203.", 1203) == 1.0 and numeric("none", 1) == 0.0
    assert exact(" Hello  World ", "hello world") == 1.0
    assert contains("readiness probe ok", ["readiness", "liveness"]) == 0.5
    assert regex('{"ok": true}', r'"ok"\s*:\s*true') == 1.0
    assert json_valid('{"ok": 1, "model": "x"}', ["ok", "model"]) == 1.0 and json_valid("nope") == 0.0


async def test_suite_compares_models_through_gateway(gw):
    suite = load_suite("evals/suites/basic.yaml")
    transport = httpx.ASGITransport(app=gw.app)
    rep = await run_suite(suite, ["smart", "sloppy"], "http://gw/v1", "test-alice-not-a-secret",
                          transport=transport)
    assert rep["models"]["smart"]["score"] == 1.0
    assert rep["models"]["sloppy"]["score"] < 1.0
    md = to_markdown(rep)
    assert "| smart | 100.00%" in md and "Per-case disagreements" in md

    # regression: baseline = smart, candidate claims the same model name but sloppy results
    cand = {**rep, "models": {"smart": rep["models"]["sloppy"]},
            "results": [{**r, "model": "smart"} for r in rep["results"] if r["model"] == "sloppy"]}
    cmp = compare(rep, cand)
    assert cmp["regressed"] and cmp["models"]["smart"]["regressed_cases"]


def test_compare_cli_exit_code(tmp_path):
    import json
    base = {"suite": "s", "models": {"m": {"score": 1.0}}, "results": [{"model": "m", "case_id": "a", "score": 1.0}]}
    cand = {"suite": "s", "models": {"m": {"score": 0.5}}, "results": [{"model": "m", "case_id": "a", "score": 0.0}]}
    (tmp_path / "b.json").write_text(json.dumps(base))
    (tmp_path / "c.json").write_text(json.dumps(cand))
    assert eval_main(["compare", str(tmp_path / "b.json"), str(tmp_path / "c.json")]) == 1
    assert eval_main(["compare", str(tmp_path / "b.json"), str(tmp_path / "b.json")]) == 0
