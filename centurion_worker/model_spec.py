"""Canonical model specification shared by every backend.

This is the single source of truth that lets the PyTorch worker (Windows/CUDA)
and the MLX worker (Apple Silicon) represent the *same* network, so their
weights can be exchanged as fp32 and averaged (DiLoCo).

Step 1 uses a tiny two-layer MLP classifier: the goal is to get the
cross-device collaboration pipeline working end to end, not model scale. A later
milestone upgrades this to GPT-2 small.

Both frameworks must:
  * use exactly these layer shapes,
  * expose parameters under exactly these canonical names,
  * exchange parameters as float32.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

# --- architecture (tiny MLP) ---------------------------------------------
INPUT_DIM: int = 16
HIDDEN_DIM: int = 32
NUM_CLASSES: int = 4

# Seed for the fixed "true" function the synthetic task is built around.
# Shared by every worker so that DiLoCo averaging is meaningful.
TASK_SEED: int = 1234

# Exchange dtype. Averaging in fp32 prevents cross-framework drift.
PARAM_DTYPE = np.float32

# Canonical parameter names. nn.Linear weight is [out, in] in BOTH PyTorch and
# MLX, so no transpose/renaming is needed when moving weights between them.
PARAM_NAMES: List[str] = [
    "fc1.weight",  # [HIDDEN_DIM, INPUT_DIM]
    "fc1.bias",    # [HIDDEN_DIM]
    "fc2.weight",  # [NUM_CLASSES, HIDDEN_DIM]
    "fc2.bias",    # [NUM_CLASSES]
]

PARAM_SHAPES: Dict[str, tuple] = {
    "fc1.weight": (HIDDEN_DIM, INPUT_DIM),
    "fc1.bias": (HIDDEN_DIM,),
    "fc2.weight": (NUM_CLASSES, HIDDEN_DIM),
    "fc2.bias": (NUM_CLASSES,),
}


def init_params(seed: int = 0) -> Dict[str, np.ndarray]:
    """Framework-agnostic weight initialization (numpy, fp32).

    Both backends can load from this so every worker starts identical, which
    makes early DiLoCo rounds easy to reason about.
    """
    rng = np.random.default_rng(seed)
    params: Dict[str, np.ndarray] = {}
    # Kaiming-ish fan-in scaling, deterministic.
    params["fc1.weight"] = (rng.standard_normal(PARAM_SHAPES["fc1.weight"]) *
                            (1.0 / np.sqrt(INPUT_DIM))).astype(PARAM_DTYPE)
    params["fc1.bias"] = np.zeros(PARAM_SHAPES["fc1.bias"], dtype=PARAM_DTYPE)
    params["fc2.weight"] = (rng.standard_normal(PARAM_SHAPES["fc2.weight"]) *
                            (1.0 / np.sqrt(HIDDEN_DIM))).astype(PARAM_DTYPE)
    params["fc2.bias"] = np.zeros(PARAM_SHAPES["fc2.bias"], dtype=PARAM_DTYPE)
    return params


def validate(state: Dict[str, np.ndarray]) -> None:
    """Raise ValueError if a state dict does not match the canonical contract."""
    missing = set(PARAM_NAMES) - set(state.keys())
    extra = set(state.keys()) - set(PARAM_NAMES)
    if missing:
        raise ValueError(f"state missing canonical params: {sorted(missing)}")
    if extra:
        raise ValueError(f"state has unexpected params: {sorted(extra)}")
    for name in PARAM_NAMES:
        arr = state[name]
        if tuple(arr.shape) != PARAM_SHAPES[name]:
            raise ValueError(
                f"param {name!r} has shape {tuple(arr.shape)}, "
                f"expected {PARAM_SHAPES[name]}"
            )
