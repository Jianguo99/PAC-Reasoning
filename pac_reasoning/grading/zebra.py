"""ZebraLogic (grid mode) grading.

Adapted from ZeroEval (Apache-2.0):
  https://github.com/WildEval/ZeroEval/blob/main/src/evaluation/zebra_grid_eval.py
A puzzle counts as solved only if every cell of the predicted grid matches the
reference solution (case-insensitive).
"""

from __future__ import annotations

import json


def extract_last_complete_json(s: str):
    """Parse the last top-level ``{...}`` block of ``s``; ``None`` if it is not valid JSON."""
    stack = []
    last_json_start = None
    last_json_str = None
    for i, char in enumerate(s):
        if char == "{":
            stack.append(i)
            if last_json_start is None:
                last_json_start = i
        elif char == "}":
            if stack:
                stack.pop()
                if not stack:
                    last_json_str = s[last_json_start : i + 1]
                    last_json_start = None
    if last_json_str:
        try:
            return json.loads(last_json_str.replace("\n", ""))
        except json.JSONDecodeError:
            pass
    return None


def solution_table(solution: dict) -> dict:
    """``{"House 1": {"Name": "Arnold", ...}, ...}`` from the benchmark's header/rows format."""
    header = solution["header"]
    assert header[0] == "House"
    return {
        f"House {i + 1}": {header[j]: row[j] for j in range(1, len(header))}
        for i, row in enumerate(solution["rows"])
    }


def _cell(value):
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    return str(value).lower().strip()


def grade_zebra(response: str, solution: dict) -> bool:
    """``True`` if the JSON grid in ``response`` matches ``solution`` in every cell."""
    prediction = extract_last_complete_json(response)
    if not isinstance(prediction, dict) or not isinstance(prediction.get("solution"), dict):
        return False
    predicted = prediction["solution"]
    for house, columns in solution_table(solution).items():
        row = predicted.get(house)
        if not isinstance(row, dict):
            return False
        for column, truth in columns.items():
            if _cell(row.get(column)) != truth.lower().strip():
                return False
    return True
