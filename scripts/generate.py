"""Generate answers with an OpenAI-compatible server (vLLM or SGLang) and grade them.

Each output line records the response, its token count, wall-clock latency and
token-probability statistics (used by the logits-based uncertainty score):

    python scripts/generate.py --dataset math500 \
        --model Qwen/Qwen3-4B-Instruct-2507 --base-url http://localhost:8001/v1 \
        --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
        --output results/math500/nonthinking.jsonl

Start the server without a reasoning parser so that ``content`` holds the full
output, thinking included (this is what the paper embeds and grades).
Re-running the same command resumes an interrupted run.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from pac_reasoning.benchmarks import BENCHMARKS, PROMPT_STYLES, apply_prompt_style, load_benchmark
from pac_reasoning.grading import grade
from pac_reasoning.records import read_jsonl, write_jsonl


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=BENCHMARKS)
    p.add_argument("--model", required=True, help="model name served by the endpoint")
    p.add_argument("--output", required=True)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--data-file", help="question file for arena-hard")
    p.add_argument("--prompt-style", default="default", choices=PROMPT_STYLES,
                   help="'nothinking' and 'cod' are the efficient-reasoning baselines")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--frequency-penalty", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--limit", type=int, help="only the first N queries (for smoke tests)")
    p.add_argument("--timeout", type=float, default=3600.0, help="per-request timeout in seconds")
    return p.parse_args()


def token_stats(choice):
    """Mean token probability and mean token log-probability of the sampled tokens."""
    content = getattr(choice.logprobs, "content", None) if choice.logprobs else None
    if not content:
        return None, None
    logprobs = [t.logprob for t in content]
    return (
        sum(math.exp(lp) for lp in logprobs) / len(logprobs),
        sum(logprobs) / len(logprobs),
    )


def query(client, args, item):
    messages = apply_prompt_style(item["messages"], args.prompt_style, args.dataset)
    start = time.perf_counter()
    completion = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        logprobs=True,
        extra_body={"top_k": args.top_k, "min_p": args.min_p},
    )
    latency = time.perf_counter() - start
    choice = completion.choices[0]
    prob_mean, logprob_mean = token_stats(choice)
    record = {
        "id": item["id"],
        "index": item["index"],
        "dataset": args.dataset,
        "model": args.model,
        "prompt_style": args.prompt_style,
        "question": item["question"],
        "messages": messages,
        "response": choice.message.content or "",
        "finish_reason": choice.finish_reason,
        "num_tokens": completion.usage.completion_tokens,
        "latency": latency,
        "token_prob_mean": prob_mean,
        "token_logprob_mean": logprob_mean,
    }
    reasoning = getattr(choice.message, "reasoning_content", None)
    if reasoning:
        record["reasoning_content"] = reasoning
    return record


def main():
    args = parse_args()
    items = load_benchmark(args.dataset, data_file=args.data_file)
    if args.limit:
        items = items[: args.limit]
    by_id = {it["id"]: it for it in items}

    done = {r["id"] for r in read_jsonl(args.output)} if os.path.exists(args.output) else set()
    todo = [it for it in items if it["id"] not in done]
    print(f"{args.dataset}: {len(items)} queries, {len(done)} already done, {len(todo)} to go")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout, max_retries=3)
    failures = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(query, client, args, it): it["id"] for it in todo}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                record = future.result()
            except Exception as e:  # keep going; a re-run picks up the missing ids
                failures += 1
                print(f"[{futures[future]}] request failed: {type(e).__name__}: {e}")
                continue
            # Grade in the main thread (the math grader uses SIGALRM timeouts).
            record["reference"] = by_id[record["id"]]["reference"]
            record["correct"] = grade(args.dataset, record["response"], record["reference"])
            write_jsonl(args.output, [record], mode="a")

    # Rewrite in benchmark order.
    rows = sorted(read_jsonl(args.output), key=lambda r: r["index"])
    write_jsonl(args.output, rows)
    graded = [r["correct"] for r in rows if r.get("correct") is not None]
    summary = f"wrote {len(rows)} records to {args.output}"
    if graded:
        summary += f"; accuracy {sum(graded) / len(graded):.4f}"
    if failures:
        summary += f"; {failures} requests failed - run the same command again to retry them"
    print(summary)


if __name__ == "__main__":
    main()
