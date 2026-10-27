"""Reading generation records and assembling :class:`RoutingData`.

``scripts/generate.py`` writes one JSON object per query:

    {"id", "dataset", "model", "messages", "response", "finish_reason",
     "num_tokens", "latency", "token_prob_mean", "token_logprob_mean",
     "correct", "reference"}

Side files written by the scoring scripts are keyed by the same ``id``:

    *.verbalized.jsonl   {"id", "confidences", "verbalized_uncertainty"}
    semantic_loss.jsonl  {"id", "semantic_loss"}
    any score file       {"id", "score"}
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np

from .experiment import RoutingData
from .losses import binary_loss

SCORES = ("logits", "logprob", "verbalized", "file")


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows, mode: str = "w") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _by_id(rows: list[dict], path: str) -> dict:
    out = {}
    for row in rows:
        if row["id"] in out:
            raise ValueError(f"duplicate id {row['id']!r} in {path}")
        out[row["id"]] = row
    return out


def uncertainty_from_record(record: dict, score: str) -> float:
    """White-box uncertainty scores stored in a generation record.

    ``logits``  : 1 - mean token probability (the definition in the paper).
    ``logprob`` : 1 - mean token log-probability (a monotone variant).
    """
    if score == "logits":
        return 1.0 - record["token_prob_mean"]
    if score == "logprob":
        return 1.0 - record["token_logprob_mean"]
    raise ValueError(f"{score!r} is not stored in generation records")


def load_routing_data(
    thinking_path: str,
    nonthinking_path: str,
    loss: str = "binary",
    score: str = "logits",
    semantic_loss_path: str | None = None,
    score_path: str | None = None,
) -> RoutingData:
    """Align the two models' records by ``id`` and build the per-query arrays.

    Args:
        loss: ``"binary"`` (needs ``correct`` in both record files) or
            ``"semantic"`` (needs ``semantic_loss_path``).
        score: one of ``SCORES``.  ``"verbalized"`` and ``"file"`` read
            ``score_path``; ``"verbalized"`` defaults to
            ``<nonthinking_path without .jsonl>.verbalized.jsonl``.
    """
    if score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}")
    strong = _by_id(read_jsonl(thinking_path), thinking_path)
    weak = _by_id(read_jsonl(nonthinking_path), nonthinking_path)

    side = {}
    if loss == "semantic":
        if semantic_loss_path is None:
            raise ValueError("semantic loss needs semantic_loss_path")
        side["loss"] = (_by_id(read_jsonl(semantic_loss_path), semantic_loss_path), "semantic_loss")
    elif loss != "binary":
        raise ValueError("loss must be 'binary' or 'semantic'")
    if score in ("verbalized", "file"):
        if score_path is None and score == "verbalized":
            score_path = nonthinking_path.removesuffix(".jsonl") + ".verbalized.jsonl"
        if score_path is None:
            raise ValueError("score='file' needs score_path")
        key = "verbalized_uncertainty" if score == "verbalized" else "score"
        side["score"] = (_by_id(read_jsonl(score_path), score_path), key)

    ids = [i for i in weak if i in strong and all(i in table for table, _ in side.values())]
    # Keep benchmark order so that a given seed always gives the same split.
    ids.sort(key=lambda i: weak[i].get("index", 0))
    dropped = len(set(strong) | set(weak)) - len(ids)
    if dropped:
        warnings.warn(f"{dropped} queries are missing from at least one input file and were skipped")
    if not ids:
        raise ValueError("no query id is shared by all input files")

    s = [strong[i] for i in ids]
    w = [weak[i] for i in ids]

    def col(rows, key):
        return np.array([r[key] for r in rows], dtype=float)

    if loss == "binary":
        missing = [r["id"] for r in s + w if r.get("correct") is None]
        if missing:
            raise ValueError(f"binary loss needs correctness labels; missing for e.g. {missing[:3]}")
        losses = binary_loss(col(s, "correct"), col(w, "correct"))
    else:
        table, key = side["loss"]
        losses = np.array([table[i][key] for i in ids], dtype=float)

    if score in ("logits", "logprob"):
        scores = np.array([uncertainty_from_record(r, score) for r in w], dtype=float)
    else:
        table, key = side["score"]
        scores = np.array([table[i][key] for i in ids], dtype=float)

    has_labels = all(r.get("correct") is not None for r in s + w)
    return RoutingData(
        losses=losses,
        scores=scores,
        thinking_tokens=col(s, "num_tokens"),
        nonthinking_tokens=col(w, "num_tokens"),
        thinking_latency=col(s, "latency"),
        nonthinking_latency=col(w, "latency"),
        thinking_correct=col(s, "correct") if has_labels else None,
        nonthinking_correct=col(w, "correct") if has_labels else None,
        ids=ids,
    )
