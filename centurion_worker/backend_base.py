"""Abstract training backend.

The coordinator and the DiLoCo client only ever see numpy arrays through this
interface -- they never import torch or mlx. That is what makes the system
cross-framework: a PyTorch worker (Windows) and an MLX worker (Mac) are
interchangeable behind `Backend`.
"""

from __future__ import annotations

import abc
from typing import Dict, List

import numpy as np


class Backend(abc.ABC):
    @property
    @abc.abstractmethod
    def framework(self) -> str:
        """e.g. 'pytorch' or 'mlx'."""

    @property
    @abc.abstractmethod
    def device_kind(self) -> str:
        """e.g. 'cuda:0', 'cpu', 'mlx-gpu'."""

    @abc.abstractmethod
    def train_local_steps(self, num_steps: int) -> List[float]:
        """Run `num_steps` local optimizer steps; return per-step loss."""

    @abc.abstractmethod
    def get_params(self) -> Dict[str, np.ndarray]:
        """Return current weights as fp32 numpy arrays under canonical names."""

    @abc.abstractmethod
    def set_params(self, state: Dict[str, np.ndarray]) -> None:
        """Load weights (e.g. the averaged global model) into the live model."""

    @abc.abstractmethod
    def forward_logits(self, x: np.ndarray) -> np.ndarray:
        """Deterministic forward pass; used by the cross-framework parity check."""

    def get_delta(self, reference: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """current - reference, per parameter (the DiLoCo 'outer gradient')."""
        current = self.get_params()
        return {name: current[name] - reference[name] for name in current}
