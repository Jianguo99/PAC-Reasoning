"""GPQA grading: the response must end with a line ``ANSWER: X`` (X in A-D)."""

from __future__ import annotations

import re

LETTERS = "ABCD"


def parse_choice(response: str) -> str | None:
    """Letter of the last ``ANSWER: X`` in the response, or ``None``."""
    matches = re.findall(r"ANSWER:\s*([A-D])", response)
    return matches[-1] if matches else None


def grade_gpqa(response: str, correct_index: int) -> bool:
    """Unparseable responses count as wrong."""
    letter = parse_choice(response)
    return letter is not None and LETTERS.index(letter) == int(correct_index)
