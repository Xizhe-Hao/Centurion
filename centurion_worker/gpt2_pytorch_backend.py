"""PyTorch GPT-2 small backend, keyed to the MLX-Swift GPT2Model.

Module attribute names are chosen so that `state_dict()` yields exactly the
gpt2_spec.PARAM_NAMES key set with ZERO renaming, so the checkpoint server
merges our weights with the iOS/MLX workers byte-for-byte:

  - submodules named ln1/ln2/wQ/wK/wV/wO/fc1/fc2, embedding, blocks, final_ln
  - pos_embedding is a bare nn.Parameter (key has no ".weight")
  - output head is tied to embedding.weight.T (NO lm_head module/key)
  - causal mask registered with persistent=False (never enters state_dict)
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import gpt2_data, gpt2_spec
from .backend_base import Backend


class GPT2Block(nn.Module):
    def __init__(self, d, n_heads, ffn, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, eps=1e-5)
        self.ln2 = nn.LayerNorm(d, eps=1e-5)
        self.wQ = nn.Linear(d, d)
        self.wK = nn.Linear(d, d)
        self.wV = nn.Linear(d, d)
        self.wO = nn.Linear(d, d)
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.head_dim = d // n_heads

    def forward(self, x, mask):
        B, S, D = x.shape
        h = self.ln1(x)
        q = self.wQ(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wK(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wV(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att + mask
        att = self.attn_drop(att.softmax(dim=-1))
        o = (att @ v).transpose(1, 2).reshape(B, S, D)
        x = x + self.resid_drop(self.wO(o))
        h = self.ln2(x)
        h = self.fc2(F.gelu(self.fc1(h), approximate="tanh"))
        return x + self.resid_drop(h)


class GPT2Module(nn.Module):
    def __init__(self, vocab, d, n_heads, n_layers, ffn, seq, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d)               # -> embedding.weight
        self.pos_embedding = nn.Parameter(torch.empty(seq, d))  # -> pos_embedding (no .weight)
        self.blocks = nn.ModuleList(
            [GPT2Block(d, n_heads, ffn, dropout) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(d, eps=1e-5)
        self.emb_drop = nn.Dropout(dropout)
        # Non-persistent causal mask: never enters state_dict().
        m = torch.full((seq, seq), float("-inf")).triu(1)
        self.register_buffer("_mask", m.view(1, 1, seq, seq), persistent=False)

    def forward(self, tokens):  # tokens: [B, S] int64
        B, S = tokens.shape
        x = self.embedding(tokens) + self.pos_embedding[:S]
        x = self.emb_drop(x)
        mask = self._mask[:, :, :S, :S]
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.final_ln(x)
        return x @ self.embedding.weight.t()  # tied head -> [B, S, V]


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GPT2PyTorchBackend(Backend):
    def __init__(self, data_seed: int = 0, lr: float = 6e-4, batch_size: int = 8):
        self._device = _pick_device()
        self._model = GPT2Module(
            vocab=gpt2_spec.VOCAB_SIZE, d=gpt2_spec.D_MODEL, n_heads=gpt2_spec.N_HEADS,
            n_layers=gpt2_spec.N_LAYERS, ffn=gpt2_spec.FFN_HIDDEN, seq=gpt2_spec.SEQ_LEN,
        ).to(self._device)
        self._opt = torch.optim.AdamW(
            self._model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
        )
        self._loss_fn = nn.CrossEntropyLoss()
        self._data_seed = int(data_seed)
        self._batch_size = int(batch_size)
        self._step = 0
        # Start from the canonical shared init so all workers begin identical.
        self.set_params(gpt2_spec.init_params(seed=0))

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
            x_np, y_np = gpt2_data.sample_batch(
                self._batch_size, gpt2_spec.SEQ_LEN,
                seed=self._data_seed * 1_000_003 + self._step,
            )
            self._step += 1
            x = torch.from_numpy(x_np).to(self._device)
            y = torch.from_numpy(y_np).to(self._device)

            self._opt.zero_grad(set_to_none=True)
            logits = self._model(x)  # [B, S, V]
            B, S, V = logits.shape
            loss = self._loss_fn(logits.reshape(B * S, V), y.reshape(B * S))
            loss.backward()
            self._opt.step()
            losses.append(float(loss.detach().cpu()))
        return losses

    def get_params(self) -> Dict[str, np.ndarray]:
        sd = self._model.state_dict()
        return {
            name: sd[name].detach().cpu().numpy().astype(gpt2_spec.PARAM_DTYPE)
            for name in gpt2_spec.PARAM_NAMES
        }

    def set_params(self, state: Dict[str, np.ndarray]) -> None:
        gpt2_spec.validate(state)
        new_sd = {
            name: torch.from_numpy(
                np.ascontiguousarray(state[name], dtype=gpt2_spec.PARAM_DTYPE)
            )
            for name in gpt2_spec.PARAM_NAMES
        }
        # strict=False tolerates the non-persistent _mask buffer absence.
        self._model.load_state_dict(new_sd, strict=False)
        self._model.to(self._device)

    def forward_logits(self, x: np.ndarray) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.int64)).to(self._device)
            return self._model(xt).detach().cpu().numpy().astype(gpt2_spec.PARAM_DTYPE)

    def eval_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        self._model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.int64)).to(self._device)
            yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)).to(self._device)
            logits = self._model(xt)
            B, S, V = logits.shape
            return float(self._loss_fn(logits.reshape(B * S, V), yt.reshape(B * S)).cpu())
