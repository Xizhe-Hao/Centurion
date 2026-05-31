"""Synthetic classification task shared by all workers.

A fixed random teacher (seeded by model_spec.TASK_SEED) defines the function
y = argmax(W_true @ x + b_true). Every worker learns this same function, so
averaging models trained on different data shards is genuinely helpful --
which is the whole point of the DiLoCo demo.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from . import model_spec


def _teacher() -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(model_spec.TASK_SEED)
    w_true = rng.standard_normal((model_spec.NUM_CLASSES, model_spec.INPUT_DIM))
    b_true = rng.standard_normal((model_spec.NUM_CLASSES,))
    return w_true.astype(np.float32), b_true.astype(np.float32)


_W_TRUE, _B_TRUE = _teacher()


def sample_batch(batch_size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x, y): x is [batch, INPUT_DIM] fp32, y is [batch] int64 labels.

    `seed` selects the data shard so different workers see different batches of
    the SAME underlying task.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((batch_size, model_spec.INPUT_DIM)).astype(np.float32)
    logits = x @ _W_TRUE.T + _B_TRUE  # [batch, NUM_CLASSES]
    y = np.argmax(logits, axis=1).astype(np.int64)
    return x, y


def fixed_eval_batch(batch_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """A fixed held-out batch (seed 0) for comparing models consistently."""
    return sample_batch(batch_size, seed=0)
