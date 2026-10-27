"""Losses of keeping the non-thinking answer instead of the thinking answer."""

from __future__ import annotations

import numpy as np


def binary_loss(thinking_correct, nonthinking_correct) -> np.ndarray:
    """0-1 loss for verifiable tasks.

    ``l = 1`` only when the thinking model is right and the non-thinking model
    is wrong, i.e. when switching to the non-thinking model costs accuracy.
    """
    thinking_correct = np.asarray(thinking_correct, dtype=bool)
    nonthinking_correct = np.asarray(nonthinking_correct, dtype=bool)
    return (thinking_correct & ~nonthinking_correct).astype(float)


def semantic_loss(thinking_embeddings, nonthinking_embeddings) -> np.ndarray:
    """``1 - cos(v_thinking, v_nonthinking)`` for each pair of answer embeddings."""
    a = np.asarray(thinking_embeddings, dtype=np.float64)
    b = np.asarray(nonthinking_embeddings, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("embeddings must be two arrays of shape (n, d)")
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = np.einsum("ij,ij->i", a, b) / np.maximum(norms, 1e-12)
    return 1.0 - cos
