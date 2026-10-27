"""Correctness checkers for the verifiable benchmarks."""

from __future__ import annotations


def grade(dataset: str, response: str, reference) -> bool | None:
    """Return whether ``response`` is correct, or ``None`` for open-ended benchmarks.

    ``reference`` is the ``reference`` field produced by
    :func:`pac_reasoning.benchmarks.load_benchmark`.
    """
    if dataset in ("math500", "aime"):
        from .math_grader import grade_math  # needs latex2sympy2

        return grade_math(response, reference)
    if dataset == "gpqa":
        from .gpqa import grade_gpqa

        return grade_gpqa(response, reference)
    if dataset == "zebralogic":
        from .zebra import grade_zebra

        return grade_zebra(response, reference)
    return None
