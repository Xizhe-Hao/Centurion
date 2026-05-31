"""MLX backend (Apple Silicon / Mac).

Mirrors pytorch_backend.PyTorchBackend exactly: same canonical tiny MLP, same
parameter names, same fp32 exchange. This file is written to run on a Mac with
`mlx` installed; it is not importable on Windows (that's fine -- make_backend
imports it lazily).

nn.Linear weight layout is [out, in] in BOTH MLX and PyTorch, so the canonical
state dict transfers with no transpose.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim

from . import data, model_spec
from .backend_base import Backend


class _TinyMLP(mlx_nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = mlx_nn.Linear(model_spec.INPUT_DIM, model_spec.HIDDEN_DIM)
        self.fc2 = mlx_nn.Linear(model_spec.HIDDEN_DIM, model_spec.NUM_CLASSES)

    def __call__(self, x):
        x = mlx_nn.relu(self.fc1(x))
        return self.fc2(x)


def _loss_fn(model, x, y):
    logits = model(x)
    return mx.mean(mlx_nn.losses.cross_entropy(logits, y))


class MLXBackend(Backend):
    def __init__(self, data_seed: int = 0, lr: float = 0.05, batch_size: int = 64):
        self._model = _TinyMLP()
        self._opt = mlx_optim.AdamW(learning_rate=lr)
        self._data_seed = int(data_seed)
        self._batch_size = int(batch_size)
        self._step = 0
        # grad function over (model, x, y)
        self._loss_and_grad = mlx_nn.value_and_grad(self._model, _loss_fn)
        # Start from the canonical shared init so all workers begin identical.
        self.set_params(model_spec.init_params(seed=0))

    @property
    def framework(self) -> str:
        return "mlx"

    @property
    def device_kind(self) -> str:
        return "mlx-gpu"

    def train_local_steps(self, num_steps: int) -> List[float]:
        losses: List[float] = []
        for _ in range(num_steps):
            x_np, y_np = data.sample_batch(
                self._batch_size, seed=self._data_seed * 1_000_003 + self._step
            )
            self._step += 1
            x = mx.array(x_np)
            y = mx.array(y_np)

            loss, grads = self._loss_and_grad(self._model, x, y)
            self._opt.update(self._model, grads)
            mx.eval(self._model.parameters(), self._opt.state)
            losses.append(float(loss))
        return losses

    def get_params(self) -> Dict[str, np.ndarray]:
        # MLX nested params: {'fc1': {'weight':..,'bias':..}, 'fc2': {...}}
        p = self._model.parameters()
        out: Dict[str, np.ndarray] = {}
        for layer in ("fc1", "fc2"):
            for kind in ("weight", "bias"):
                out[f"{layer}.{kind}"] = np.asarray(
                    p[layer][kind], dtype=model_spec.PARAM_DTYPE
                )
        return out

    def set_params(self, state: Dict[str, np.ndarray]) -> None:
        model_spec.validate(state)
        new_params = {
            "fc1": {
                "weight": mx.array(np.ascontiguousarray(state["fc1.weight"], dtype=np.float32)),
                "bias": mx.array(np.ascontiguousarray(state["fc1.bias"], dtype=np.float32)),
            },
            "fc2": {
                "weight": mx.array(np.ascontiguousarray(state["fc2.weight"], dtype=np.float32)),
                "bias": mx.array(np.ascontiguousarray(state["fc2.bias"], dtype=np.float32)),
            },
        }
        self._model.update(new_params)
        mx.eval(self._model.parameters())

    def forward_logits(self, x: np.ndarray) -> np.ndarray:
        xt = mx.array(np.ascontiguousarray(x, dtype=np.float32))
        logits = self._model(xt)
        mx.eval(logits)
        return np.asarray(logits, dtype=model_spec.PARAM_DTYPE)

    def eval_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        loss = _loss_fn(self._model, mx.array(x), mx.array(y.astype(np.int32)))
        mx.eval(loss)
        return float(loss)
