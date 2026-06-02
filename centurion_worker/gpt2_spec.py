"""Canonical GPT-2 small specification (keyed to the MLX-Swift GPT2Model).

This is the single source of truth that lets the Windows PyTorch worker and the
iOS/Mac MLX worker represent the *same* network, so their weights can be
exchanged as fp32 safetensors and averaged by the team's checkpoint server.

The checkpoint server averages parameters BY KEY STRING and silently drops any
key that does not match across contributors. So the key set here must match the
Swift `GPT2Model` exactly (verified against CenturionMLX/.../GPT2Model.swift):

  - LayerNorm uses weight/bias (NOT scale).
  - Linear weight is [out, in] in BOTH PyTorch and MLX -> no transpose.
  - pos_embedding is a bare parameter -> key has NO ".weight" suffix.
  - Output head is TIED to the token embedding -> there is NO lm_head key.
  - Dropout layers are parameter-free -> no keys.

Phase 2 uses byte-level vocab V=256 (matches the Swift default vocabSize: 256),
so no tokenizer is needed and any worker can join with the same V.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

# --- architecture (GPT-2 small) ------------------------------------------
D_MODEL: int = 768
N_HEADS: int = 12
N_LAYERS: int = 12
FFN_HIDDEN: int = 3072          # d_model * 4
SEQ_LEN: int = 128
VOCAB_SIZE: int = 256           # byte-level (matches Swift TransformerConfig default)
HEAD_DIM: int = D_MODEL // N_HEADS

# Exchange dtype. Averaging in fp32 prevents cross-framework drift.
PARAM_DTYPE = np.float32

_PER_BLOCK = [
    "ln1.weight", "ln1.bias",
    "ln2.weight", "ln2.bias",
    "wQ.weight", "wQ.bias",
    "wK.weight", "wK.bias",
    "wV.weight", "wV.bias",
    "wO.weight", "wO.bias",
    "fc1.weight", "fc1.bias",
    "fc2.weight", "fc2.bias",
]


def _param_names(n_layers: int = N_LAYERS) -> List[str]:
    names = ["embedding.weight", "pos_embedding", "final_ln.weight", "final_ln.bias"]
    for i in range(n_layers):
        names += [f"blocks.{i}.{p}" for p in _PER_BLOCK]
    return names


PARAM_NAMES: List[str] = _param_names()


def _param_shapes() -> Dict[str, tuple]:
    d, ffn, v, seq = D_MODEL, FFN_HIDDEN, VOCAB_SIZE, SEQ_LEN
    shapes: Dict[str, tuple] = {
        "embedding.weight": (v, d),       # tied: also the output head
        "pos_embedding": (seq, d),        # bare parameter, no .weight
        "final_ln.weight": (d,),
        "final_ln.bias": (d,),
    }
    for i in range(N_LAYERS):
        p = f"blocks.{i}."
        shapes[p + "ln1.weight"] = (d,)
        shapes[p + "ln1.bias"] = (d,)
        shapes[p + "ln2.weight"] = (d,)
        shapes[p + "ln2.bias"] = (d,)
        shapes[p + "wQ.weight"] = (d, d)
        shapes[p + "wQ.bias"] = (d,)
        shapes[p + "wK.weight"] = (d, d)
        shapes[p + "wK.bias"] = (d,)
        shapes[p + "wV.weight"] = (d, d)
        shapes[p + "wV.bias"] = (d,)
        shapes[p + "wO.weight"] = (d, d)
        shapes[p + "wO.bias"] = (d,)
        shapes[p + "fc1.weight"] = (ffn, d)
        shapes[p + "fc1.bias"] = (ffn,)
        shapes[p + "fc2.weight"] = (d, ffn)
        shapes[p + "fc2.bias"] = (d,)
    return shapes


PARAM_SHAPES: Dict[str, tuple] = _param_shapes()


def init_params(seed: int = 0) -> Dict[str, np.ndarray]:
    """Framework-agnostic GPT-2 init (numpy, fp32), matching the Swift init:
      - Linear weights ~ N(0, 0.02), biases zero
      - LayerNorm weight=1, bias=0
      - embedding.weight ~ N(0, 0.02)
      - pos_embedding ~ N(0, 0.01)  (Swift: MLXRandom.normal * 0.01)
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, np.ndarray] = {}
    f32 = PARAM_DTYPE

    out["embedding.weight"] = (rng.standard_normal(PARAM_SHAPES["embedding.weight"]) * 0.02).astype(f32)
    out["pos_embedding"] = (rng.standard_normal(PARAM_SHAPES["pos_embedding"]) * 0.01).astype(f32)
    out["final_ln.weight"] = np.ones(PARAM_SHAPES["final_ln.weight"], dtype=f32)
    out["final_ln.bias"] = np.zeros(PARAM_SHAPES["final_ln.bias"], dtype=f32)

    for i in range(N_LAYERS):
        p = f"blocks.{i}."
        # LayerNorms
        out[p + "ln1.weight"] = np.ones((D_MODEL,), dtype=f32)
        out[p + "ln1.bias"] = np.zeros((D_MODEL,), dtype=f32)
        out[p + "ln2.weight"] = np.ones((D_MODEL,), dtype=f32)
        out[p + "ln2.bias"] = np.zeros((D_MODEL,), dtype=f32)
        # Linear weights/biases
        for name in ("wQ", "wK", "wV", "wO", "fc1", "fc2"):
            w_shape = PARAM_SHAPES[p + name + ".weight"]
            b_shape = PARAM_SHAPES[p + name + ".bias"]
            out[p + name + ".weight"] = (rng.standard_normal(w_shape) * 0.02).astype(f32)
            out[p + name + ".bias"] = np.zeros(b_shape, dtype=f32)
    return out


def validate(state: Dict[str, np.ndarray]) -> None:
    """Raise ValueError if a state dict does not match the canonical contract.

    Catches a dropped/renamed/misshaped key BEFORE upload, so we never silently
    contribute nothing to the server.
    """
    missing = set(PARAM_NAMES) - set(state.keys())
    extra = set(state.keys()) - set(PARAM_NAMES)
    if missing:
        raise ValueError(f"state missing canonical params: {sorted(missing)[:5]} ...")
    if extra:
        raise ValueError(f"state has unexpected params: {sorted(extra)[:5]} ...")
    for name in PARAM_NAMES:
        if tuple(state[name].shape) != PARAM_SHAPES[name]:
            raise ValueError(
                f"param {name!r} has shape {tuple(state[name].shape)}, "
                f"expected {PARAM_SHAPES[name]}"
            )
