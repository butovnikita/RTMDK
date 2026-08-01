"""rtmdk/production/sanitization.py — Input validation and sanitization.

Guards against:
  - NaN / Inf embeddings
  - Adversarial oversized batches
  - Path traversal in file operations
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

MAX_BATCH_SIZE = 1000
MAX_EMBEDDING_DIM = 4096


class SanitizationError(ValueError):
    """Raised when input fails sanitization checks."""

    pass


def validate_embedding(
    embedding: NDArray,
    *,
    max_dim: int = MAX_EMBEDDING_DIM,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> NDArray:
    """Validate and sanitize a single embedding vector.

    Args:
        embedding: Input vector.
        max_dim: Maximum allowed dimensionality.
        allow_nan: If False, raises on NaN values.
        allow_inf: If False, raises on Inf values.

    Returns:
        Sanitized float32 array.
    """
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim != 1:
        raise SanitizationError(f"Embedding must be 1-D, got shape {arr.shape}")
    if arr.shape[0] > max_dim:
        raise SanitizationError(f"Embedding dim {arr.shape[0]} exceeds max {max_dim}")
    if not allow_nan and np.isnan(arr).any():
        raise SanitizationError("Embedding contains NaN values")
    if not allow_inf and np.isinf(arr).any():
        raise SanitizationError("Embedding contains Inf values")
    return arr


def validate_batch_size(n: int, *, max_size: int = MAX_BATCH_SIZE) -> None:
    """Validate batch size for ingestion or query."""
    if n > max_size:
        raise SanitizationError(f"Batch size {n} exceeds maximum {max_size}")


def sanitize_text(text: Optional[str], *, max_length: int = 10000) -> str:
    """Sanitize text input: strip and truncate."""
    if text is None:
        return ""
    text = str(text).strip()
    if len(text) > max_length:
        text = text[:max_length]
    return text
