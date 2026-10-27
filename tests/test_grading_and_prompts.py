import json

import pytest

from pac_reasoning.benchmarks import ZEBRA_GRID_TEMPLATE, apply_prompt_style, shuffle_gpqa, zebra_prompt
from pac_reasoning.grading.gpqa import grade_gpqa, parse_choice
from pac_reasoning.grading.zebra import extract_last_complete_json, grade_zebra

SOLUTION = {"header": ["House", "Name", "Drink"], "rows": [["1", "Arnold", "tea"], ["2", "Peter", "water"]]}


def test_gpqa_grading():
    assert parse_choice("blah\nANSWER: B") == "B"
    assert parse_choice("ANSWER: A ... ANSWER: C") == "C"
    assert grade_gpqa("ANSWER: C", 2)
    assert not grade_gpqa("ANSWER: C", 1)
    assert not grade_gpqa("no final answer", 0)  # unparseable counts as wrong


def test_zebra_grading():
    good = {"reasoning": "...", "solution": {"House 1": {"Name": "arnold", "Drink": "Tea"},
                                             "House 2": {"Name": "Peter", "Drink": ["water"]}}}
    bad = json.loads(json.dumps(good))
    bad["solution"]["House 2"]["Drink"] = "milk"
    assert grade_zebra("thinking {not json} " + json.dumps(good, indent=2), SOLUTION)
    assert not grade_zebra(json.dumps(bad), SOLUTION)
    assert not grade_zebra("no json here", SOLUTION)
    assert extract_last_complete_json('{"a": 1} text {"b": {"c": 2}}') == {"b": {"c": 2}}


def test_zebra_prompt_matches_zeroeval_format():
    # ZeroEval's template has trailing spaces after two headings.
    assert "# Example Puzzle \n" in ZEBRA_GRID_TEMPLATE and "# Puzzle to Solve \n" in ZEBRA_GRID_TEMPLATE
    prompt = zebra_prompt({"puzzle": "PUZZLE", "solution": SOLUTION})
    assert "PUZZLE" in prompt and '"House 2": {' in prompt and '"Drink": "___"' in prompt


def test_prompt_styles():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    assert apply_prompt_style(messages, "default", "math500") == messages
    assert apply_prompt_style(messages, "nothinking", "math500")[-1]["content"] == "q\n</think>"
    cod = apply_prompt_style(messages, "cod", "math500")
    assert cod[0]["role"] == "system" and "minimum draft" in cod[0]["content"] and cod[1]["content"] == "q"
    assert messages[-1]["content"] == "q"  # input is not modified


def test_gpqa_shuffle_is_deterministic():
    rows = [{"Question": f"q{i}", "Correct Answer": "right", "Incorrect Answer 1": "a",
             "Incorrect Answer 2": "b", "Incorrect Answer 3": "c"} for i in range(20)]
    first, second = shuffle_gpqa(rows), shuffle_gpqa(rows)
    assert first == second
    assert all(x["choices"][x["correct_index"]] == "right" for x in first)


def test_math_grader():
    pytest.importorskip("latex2sympy2")
    from pac_reasoning.grading.math_grader import grade_math

    assert grade_math("... so the answer is \\boxed{\\frac{1}{2}}.", "\\frac12")
    assert grade_math("\\boxed{0.5}", "\\frac{1}{2}")
    assert not grade_math("\\boxed{3}", "4")
