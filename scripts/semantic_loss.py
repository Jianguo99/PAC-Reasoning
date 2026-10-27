"""Semantic loss between the thinking and non-thinking answers.

Embeds both responses with Qwen3-Embedding-4B and writes
``1 - cos(v_thinking, v_nonthinking)`` for every query id:

    python scripts/semantic_loss.py \
        --thinking results/arena-hard/thinking.jsonl \
        --nonthinking results/arena-hard/nonthinking.jsonl \
        --output results/arena-hard/semantic_loss.jsonl
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from pac_reasoning.losses import semantic_loss
from pac_reasoning.records import read_jsonl, write_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--thinking", required=True)
    p.add_argument("--nonthinking", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    p.add_argument("--max-model-len", type=int, default=65536)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--save-embeddings", help="optional .npz file for the raw embeddings")
    args = p.parse_args()

    # Thinking traces can be longer than the embedding model's native context;
    # the paper embeds them whole with max_model_len = 65536.
    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    from vllm import LLM

    strong = {r["id"]: r for r in read_jsonl(args.thinking)}
    weak = read_jsonl(args.nonthinking)
    ids = [r["id"] for r in weak if r["id"] in strong]
    texts_strong = [strong[i]["response"] for i in ids]
    texts_weak = [r["response"] for r in weak if r["id"] in strong]

    # Chunked prefill crashes pooling models in vLLM 0.10 (KeyError in _prepare_inputs).
    llm = LLM(model=args.model, task="embed", max_model_len=args.max_model_len,
              tensor_parallel_size=args.tensor_parallel_size, enable_chunked_prefill=False)
    emb_strong = np.array([o.outputs.embedding for o in llm.embed(texts_strong)])
    emb_weak = np.array([o.outputs.embedding for o in llm.embed(texts_weak)])
    losses = semantic_loss(emb_strong, emb_weak)

    write_jsonl(args.output, [{"id": i, "semantic_loss": float(l)} for i, l in zip(ids, losses)])
    if args.save_embeddings:
        np.savez_compressed(args.save_embeddings, ids=np.array(ids), thinking=emb_strong, nonthinking=emb_weak)
    print(f"wrote {len(ids)} losses to {args.output} (mean {losses.mean():.4f})")


if __name__ == "__main__":
    main()
