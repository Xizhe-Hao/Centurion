"""A worker's slice of the pipeline-parallel GPT-2.

Built dynamically from the server-pushed PIPELINE_CONFIG (0x41), so it adapts to
whatever model dims and layer assignment the orchestrator chooses. Reuses the
GPT2Block from gpt2_pytorch_backend (identical to the MLX-Swift block).

Roles:
  - HEAD  (is_head): owns embedding + pos_embedding + blocks[first:last].
    forward: tokens -> activation[B,S,d]; backward via output.backward(upstream).
  - MIDDLE: owns blocks[first:last] only.
    forward: activation_in -> activation_out; backward yields input-activation
    gradient (to send upstream) AND parameter grads.
  - TAIL  (is_tail): owns blocks[first:last] + final_ln + tied output head.
    forward: activation_in -> logits -> cross-entropy loss against targets.

PyTorch autograd note: `output.backward(g)` performs the exact vector-Jacobian
product, populating this slice's parameter `.grad` AND `input.grad` (the
gradient w.r.t. this stage's input = what to send upstream). No cross-machine
autograd is needed; each worker backprops only its own slice.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gpt2_pytorch_backend import GPT2Block


class PipelineSlice(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.is_head = bool(cfg.is_head)
        self.is_tail = bool(cfg.is_tail)
        self.local_layers = max(1, cfg.last_layer - cfg.first_layer)
        self.seq_len = cfg.seq_len

        # Always: the assigned transformer blocks (local indices 0..local_layers).
        self.blocks = nn.ModuleList([
            GPT2Block(d, cfg.n_heads, cfg.ffn_hidden, cfg.dropout)
            for _ in range(self.local_layers)
        ])

        # Head-only modules: token + positional embeddings.
        if self.is_head:
            self.embedding = nn.Embedding(cfg.vocab_size, d)
            self.pos_embedding = nn.Parameter(
                torch.randn(cfg.seq_len, d) * 0.01
            )
            self.emb_drop = nn.Dropout(cfg.dropout)

        # Tail-only modules: final LayerNorm + a tied output projection.
        # (Swift keeps a full [V,d] embedding used only as asLinear for logits.)
        if self.is_tail:
            self.final_ln = nn.LayerNorm(d, eps=1e-5)
            self.head_embedding = nn.Embedding(cfg.vocab_size, d)

        # Non-persistent causal mask (never enters state_dict).
        m = torch.full((cfg.seq_len, cfg.seq_len), float("-inf")).triu(1)
        self.register_buffer("_mask", m.view(1, 1, cfg.seq_len, cfg.seq_len),
                             persistent=False)

    # ── HEAD ──
    def forward_head(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B,S] int64 -> output activation [B,S,d] (graph retained)."""
        B, S = tokens.shape
        x = self.embedding(tokens) + self.pos_embedding[:S]
        x = self.emb_drop(x)
        mask = self._mask[:, :, :S, :S]
        for blk in self.blocks:
            x = blk(x, mask)
        return x

    # ── MIDDLE ──
    def forward_middle(self, act_in: torch.Tensor) -> torch.Tensor:
        """act_in [B,S,d] (requires_grad) -> output activation [B,S,d]."""
        S = act_in.shape[1]
        mask = self._mask[:, :, :S, :S]
        x = act_in
        for blk in self.blocks:
            x = blk(x, mask)
        return x

    # ── TAIL ──
    def forward_tail(self, act_in: torch.Tensor) -> torch.Tensor:
        """act_in [B,S,d] -> logits [B,S,V]."""
        S = act_in.shape[1]
        mask = self._mask[:, :, :S, :S]
        x = act_in
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.final_ln(x)
        return x @ self.head_embedding.weight.t()

    @staticmethod
    def tail_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, S, V = logits.shape
        return F.cross_entropy(logits.reshape(B * S, V), targets.reshape(B * S))
