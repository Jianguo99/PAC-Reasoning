"""Verbalized (black-box) uncertainty score.

The non-thinking model is shown each question with its own answer and asked for
the probability that the answer is correct.  The score is one minus the mean of
``--n-samples`` sampled probabilities (Table 2 of the paper):

    python scripts/verbalized_confidence.py \
        --input results/math500/nonthinking.jsonl --model Qwen/Qwen3-4B-Instruct-2507

writes ``results/math500/nonthinking.verbalized.jsonl``.
"""

from __future__ import annotations

import argparse
import re

from pac_reasoning.records import read_jsonl, write_jsonl

SYSTEM_PROMPT = (
    "You are a reasoning assistant. For each question and proposed answer, "
    "you must estimate how likely the proposed answer is correct."
)
USER_TEMPLATE = """Question: {question}
Answer: {answer}
Provide a probability (between 0.0 and 1.0) that your answer is correct. Only output the probability.
"""


def parse_probability(text: str) -> float:
    """First number in the reply; values above 1 are read as percentages, no number gives 0.5."""
    match = re.search(r"(\d+\.?\d*)", text)
    if not match:
        return 0.5
    value = float(match.group(1))
    return value / 100 if value > 1 else value


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="generation records of the non-thinking model")
    p.add_argument("--model", required=True, help="the non-thinking model itself")
    p.add_argument("--output", help="default: <input>.verbalized.jsonl")
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-tokens", type=int, default=20)
    p.add_argument("--max-model-len", type=int, default=65536)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    args = p.parse_args()
    output = args.output or args.input.removesuffix(".jsonl") + ".verbalized.jsonl"

    from vllm import LLM, SamplingParams

    records = read_jsonl(args.input)
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              tensor_parallel_size=args.tensor_parallel_size, disable_log_stats=True)
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(question=r["question"], answer=r["response"])},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        for r in records
    ]
    params = SamplingParams(n=args.n_samples, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    outputs = llm.generate(prompts, params)

    rows = []
    for record, out in zip(records, outputs):
        confidences = [parse_probability(o.text) for o in out.outputs]
        rows.append({
            "id": record["id"],
            "confidences": confidences,
            "verbalized_uncertainty": 1.0 - sum(confidences) / len(confidences),
        })
    write_jsonl(output, rows)
    print(f"wrote {len(rows)} scores to {output}")


if __name__ == "__main__":
    main()
