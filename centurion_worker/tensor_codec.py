"""Serialization of a model state dict to/from bytes.

We use safetensors (language- and framework-neutral) as the wire format for
DiLoCo parameter exchange. Everything is exchanged as fp32 numpy arrays.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from safetensors.numpy import load as _st_load
from safetensors.numpy import save as _st_save

from . import model_spec


def state_to_bytes(state: Dict[str, np.ndarray]) -> bytes:
    """Serialize a canonical state dict to safetensors bytes (fp32)."""
    payload = {
        name: np.ascontiguousarray(arr, dtype=model_spec.PARAM_DTYPE)
        for name, arr in state.items()
    }
    return _st_save(payload)


def state_from_bytes(blob: bytes) -> Dict[str, np.ndarray]:
    """Deserialize safetensors bytes back into a state dict (fp32 numpy)."""
    loaded = _st_load(blob)
    return {
        name: np.asarray(arr, dtype=model_spec.PARAM_DTYPE)
        for name, arr in loaded.items()
    }


def average_states(states):
    """Element-wise fp32 mean over a non-empty list of state dicts.

    This is the core DiLoCo aggregation step. All states must share the same
    canonical keys/shapes.
    """
    states = list(states)
    if not states:
        raise ValueError("cannot average an empty list of states")

    keys = set(states[0].keys())
    for s in states[1:]:
        if set(s.keys()) != keys:
            raise ValueError("states have mismatched parameter names")

    n = float(len(states))
    out: Dict[str, np.ndarray] = {}
    for name in states[0]:
        acc = np.zeros_like(states[0][name], dtype=np.float64)
        for s in states:
            acc += s[name].astype(np.float64)
        out[name] = (acc / n).astype(model_spec.PARAM_DTYPE)
    return out
