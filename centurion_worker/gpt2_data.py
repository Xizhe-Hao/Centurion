"""Byte-level language-modeling data for the GPT-2 worker.

Tokenizer-free: tokens are raw byte values 0..255, so VOCAB_SIZE = 256 and any
worker (Windows, Mac, iOS) produces an identical-shaped embedding without
sharing a tokenizer. Next-token prediction: targets = inputs shifted by one.

A small built-in text corpus is used so the demo is self-contained; swap in a
larger UTF-8 file later by setting CENTURION_CORPUS_PATH.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

from . import gpt2_spec

# A small self-contained corpus (repeated to give the sampler room). Replace via
# the CENTURION_CORPUS_PATH env var to point at any UTF-8 text file.
_BUILTIN_TEXT = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles\n"
    "And by opposing end them. To die-to sleep,\n"
    "No more; and by a sleep to say we end\n"
    "The heart-ache and the thousand natural shocks\n"
    "That flesh is heir to: 'tis a consummation\n"
    "Devoutly to be wish'd. To die, to sleep;\n"
    "To sleep, perchance to dream-ay, there's the rub.\n"
) * 64


def _load_corpus_bytes() -> np.ndarray:
    path = os.environ.get("CENTURION_CORPUS_PATH")
    if path and os.path.isfile(path):
        with open(path, "rb") as f:
            raw = f.read()
    else:
        raw = _BUILTIN_TEXT.encode("utf-8")
    return np.frombuffer(raw, dtype=np.uint8).astype(np.int64)


_CORPUS = _load_corpus_bytes()


def sample_batch(batch_size: int, seq_len: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (inputs, targets), each [batch, seq_len] int64.

    `seed` selects the data shard so different workers see different windows of
    the SAME corpus.
    """
    rng = np.random.default_rng(seed)
    n = len(_CORPUS)
    max_start = n - seq_len - 1
    if max_start <= 0:
        raise ValueError("corpus too small for the requested seq_len")
    starts = rng.integers(0, max_start, size=batch_size)
    x = np.empty((batch_size, seq_len), dtype=np.int64)
    y = np.empty((batch_size, seq_len), dtype=np.int64)
    for i, s in enumerate(starts):
        x[i] = _CORPUS[s:s + seq_len]
        y[i] = _CORPUS[s + 1:s + 1 + seq_len]
    return x, y


def fixed_eval_batch(batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """A fixed held-out batch (seed 0) for consistent loss comparison."""
    return sample_batch(batch_size, gpt2_spec.SEQ_LEN, seed=0)
