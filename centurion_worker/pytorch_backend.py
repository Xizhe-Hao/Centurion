"""PyTorch backend (Windows / CUDA, also runs on CPU and Mac MPS).

Implements the canonical tiny MLP from model_spec and the Backend interface.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from . import data, model_spec
from .backend_base import Backend


class _TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(model_spec.INPUT_DIM, model_spec.HIDDEN_DIM)
        self.fc2 = nn.Linear(model_spec.HIDDEN_DIM, model_spec.NUM_CLASSES)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PyTorchBackend(Backend):
    def __init__(self, data_seed: int = 0, lr: float = 0.05, batch_size: int = 64):
        self._device = _pick_device()
        self._model = _TinyMLP().to(self._device)
        self._opt = torch.optim.AdamW(self._model.parameters(), lr=lr)
        self._loss_fn = nn.CrossEntropyLoss()
        self._data_seed = int(data_seed)
        self._batch_size = int(batch_size)
        self._step = 0
        # Start from the canonical shared init so all workers begin identical.
        self.set_params(model_spec.init_params(seed=0))

    @property
    def framework(self) -> str:
        return "pytorch"

    @property
    def device_kind(self) -> str:
        return str(self._device)

    def train_local_steps(self, num_steps: int) -> List[float]:
        self._model.train()
        losses: List[float] = []
        for _ in range(num_steps):
            # Distinct batch each step, but within this worker's data shard.
            x_np, y_np = data.sample_batch(
                self._batch_size, seed=self._data_seed * 1_000_003 + self._step
            )
            self._step += 1
            x = torch.from_numpy(x_np).to(self._device)
            y = torch.from_numpy(y_np).to(self._device)

            self._opt.zero_grad(set_to_none=True)
            logits = self._model(x)
            loss = self._loss_fn(logits, y)
            loss.backward()
            self._opt.step()
            losses.append(float(loss.detach().cpu()))
        return losses

    def get_params(self) -> Dict[str, np.ndarray]:
        sd = self._model.state_dict()
        return {
            name: sd[name].detach().cpu().numpy().astype(model_spec.PARAM_DTYPE)
            for name in model_spec.PARAM_NAMES
        }

    def set_params(self, state: Dict[str, np.ndarray]) -> None:
        model_spec.validate(state)
        new_sd = {
            name: torch.from_numpy(
                np.ascontiguousarray(state[name], dtype=model_spec.PARAM_DTYPE)
            )
            for name in model_spec.PARAM_NAMES
        }
        self._model.load_state_dict(new_sd)
        self._model.to(self._device)

    def forward_logits(self, x: np.ndarray) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(
                np.ascontiguousarray(x, dtype=model_spec.PARAM_DTYPE)
            ).to(self._device)
            return self._model(xt).detach().cpu().numpy().astype(model_spec.PARAM_DTYPE)

    def eval_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """Convenience for verification: cross-entropy on a fixed batch."""
        self._model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self._device)
            yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)).to(self._device)
            return float(self._loss_fn(self._model(xt), yt).detach().cpu())
