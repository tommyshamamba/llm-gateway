"""Scorers return a float in [0, 1]."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def exact(output: str, expected: Any) -> float:
    return float(_norm(output) == _norm(str(expected)))


def contains(output: str, expected: Any) -> float:
    items = expected if isinstance(expected, list) else [expected]
    return sum(_norm(str(i)) in _norm(output) for i in items) / len(items)


def regex(output: str, expected: Any) -> float:
    return float(re.search(str(expected), output, re.DOTALL) is not None)


def numeric(output: str, expected: Any, tol: float = 1e-6) -> float:
    m = re.search(r"-?\d+(?:\.\d+)?", output.replace(",", ""))
    return float(m is not None and abs(float(m.group()) - float(expected)) <= tol)


def json_valid(output: str, expected: Any = None) -> float:
    try:
        obj = json.loads(output)
    except ValueError:
        return 0.0
    if isinstance(expected, list):  # required keys
        return float(isinstance(obj, dict) and all(k in obj for k in expected))
    return 1.0


SCORERS: dict[str, Callable[[str, Any], float]] = {
    "exact": exact,
    "contains": contains,
    "regex": regex,
    "numeric": numeric,
    "json_valid": json_valid,
}
