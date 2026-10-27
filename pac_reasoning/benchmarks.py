"""Benchmarks and prompts.

``load_benchmark`` returns a list of items

    {"id", "index", "question", "messages", "reference"}

where ``messages`` is the chat sent to the model, ``question`` is the text shown
to the model when it rates its own answer (verbalized confidence), and
``reference`` is what :func:`pac_reasoning.grading.grade` compares against
(``None`` for open-ended tasks).
"""

from __future__ import annotations

import json
import random
import warnings

BENCHMARKS = ("math500", "aime", "gpqa", "zebralogic", "arena-hard")
PROMPT_STYLES = ("default", "nothinking", "cod")

MATH_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
GPQA_SYSTEM = "You are a very intelligent assistant, who follows instructions directly."

# Chain of Draft (Xu et al., 2025) baseline prompts.
COD_SYSTEM_MATH = (
    "Think step by step, but only keep a minimum draft for each thinking step, "
    "with at most five words. Finally, put your final answer within \\boxed{}."
)
COD_SYSTEM = (
    "Think step by step, but only keep a minimum draft for each thinking step, "
    "with at most five words. Follow the requested final answer format exactly."
)

# ZeroEval prompt for ZebraLogic grid puzzles (Apache-2.0,
# https://github.com/WildEval/ZeroEval/blob/main/src/templates/ZEBRA_GRID.py).
# NOTE: two headings end with a trailing space; keep it to reproduce the paper exactly.
ZEBRA_GRID_TEMPLATE = """
# Example Puzzle 

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {
        "House 1": {
            "Name": "Arnold",
            "Drink": "tea"
        },
        "House 2": {
            "Name": "Peter",
            "Drink": "water"
        },
        "House 3": {
            "Name": "Eric",
            "Drink": "milk"
        }
    }
}

# Puzzle to Solve 

{puzzle}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{json_template}

"""


def zebra_prompt(item: dict) -> str:
    header = item["solution"]["header"]
    assert header[0] == "House"
    template = {"reasoning": "___", "solution": {}}
    for i in range(len(item["solution"]["rows"])):
        template["solution"][f"House {i + 1}"] = {col: "___" for col in header[1:]}
    prompt = ZEBRA_GRID_TEMPLATE.replace("{puzzle}", item["puzzle"])
    return prompt.replace("{json_template}", json.dumps(template, indent=4))


def gpqa_prompt(question: str, choices: list[str]) -> str:
    text = f"What is the correct answer to this question: {question}"
    text += f"\n\nChoices:\n(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}"
    text += "\n\nRemember: your final answer MUST be on the last line with a strictly formatted answer:\n"
    text += "\nANSWER: X\n"
    text += "Where X must be exactly one of A, B, C, or D.\n"
    return text


def shuffle_gpqa(rows, seed: int = 2025) -> list[dict]:
    """Shuffle the four options of every GPQA question with one shared RNG."""
    rng = random.Random(seed)
    out = []
    for row in rows:
        choices = [row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"], row["Correct Answer"]]
        rng.shuffle(choices)
        out.append({"question": row["Question"], "choices": choices, "correct_index": choices.index(row["Correct Answer"])})
    return out


def _item(benchmark, index, item_id, question, user, reference, system=None):
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return {"id": str(item_id), "index": index, "question": question, "messages": messages, "reference": reference}


def _zebra_solutions() -> dict | None:
    from datasets import load_dataset

    try:
        private = load_dataset("allenai/ZebraLogicBench-private", "grid_mode", split="test")
    except Exception as e:  # gated dataset
        warnings.warn(
            "Could not load allenai/ZebraLogicBench-private (request access on the Hub and run "
            f"`huggingface-cli login`); ZebraLogic answers will not be graded. ({type(e).__name__})"
        )
        return None
    return {x["id"]: x["solution"] for x in private}


def load_benchmark(name: str, data_file: str | None = None, with_references: bool = True) -> list[dict]:
    """Load a benchmark in benchmark order."""
    from datasets import load_dataset

    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [_item(name, i, x["unique_id"], x["problem"], x["problem"], x["answer"], MATH_SYSTEM) for i, x in enumerate(ds)]

    if name == "aime":
        ds = load_dataset("GY2233/AIME-2024-2025", split="train")
        return [_item(name, i, x["ID"], x["Problem"], x["Problem"], x["Answer"], MATH_SYSTEM) for i, x in enumerate(ds)]

    if name == "gpqa":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
        items = []
        for i, (row, x) in enumerate(zip(ds, shuffle_gpqa(ds))):
            user = gpqa_prompt(x["question"], x["choices"])
            items.append(_item(name, i, row["Record ID"], x["question"], user, x["correct_index"], GPQA_SYSTEM))
        return items

    if name == "zebralogic":
        ds = load_dataset("allenai/ZebraLogicBench", "grid_mode", split="test")
        solutions = _zebra_solutions() if with_references else None
        items = []
        for i, x in enumerate(ds):
            prompt = zebra_prompt(x)
            reference = solutions[x["id"]] if solutions else None
            items.append(_item(name, i, x["id"], prompt, prompt, reference))
        return items

    if name == "arena-hard":
        if data_file is None:
            raise ValueError(
                "arena-hard needs --data-file pointing to arena-hard-v2.0/question.jsonl "
                "(see https://github.com/lmarena-ai/arena-hard-auto)"
            )
        with open(data_file, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return [_item(name, i, x["uid"], x["prompt"], x["prompt"], None) for i, x in enumerate(rows)]

    raise ValueError(f"unknown benchmark {name!r}; choose from {BENCHMARKS}")


def apply_prompt_style(messages: list[dict], style: str, benchmark: str) -> list[dict]:
    """Prompt variants for the efficient-reasoning baselines in Table 1.

    ``nothinking``: close the thinking block right away by appending
    ``</think>`` to the query.  ``cod``: Chain-of-Draft system prompt.
    """
    messages = [dict(m) for m in messages]
    if style == "default":
        return messages
    if style == "nothinking":
        messages[-1]["content"] += "\n</think>"
        return messages
    if style == "cod":
        system = COD_SYSTEM_MATH if benchmark in ("math500", "aime") else COD_SYSTEM
        return [{"role": "system", "content": system}] + [m for m in messages if m["role"] != "system"]
    raise ValueError(f"unknown prompt style {style!r}; choose from {PROMPT_STYLES}")
