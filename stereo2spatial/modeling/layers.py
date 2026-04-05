"""Transformer layers used by SpatialDiT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _modulate_time(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    """
    Apply timestep-conditioned affine modulation.

    Shapes:
        x: `[B, T, H]`
        scale: `[B, H]`
        shift: `[B, H]`
    """
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


def _modulate_film(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    """
    Apply per-token FiLM modulation.

    Shapes:
        x: `[B, T, H]`
        scale: `[B, T, H]`
        shift: `[B, T, H]`
    """
    return x * (1.0 + scale) + shift


class RMSNorm(nn.Module):
    """RMSNorm (no bias): `x / rms(x) * weight`."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize final dimension with RMS statistics and learned scale."""
        # Keep RMS statistics in fp32 for mixed-precision stability.
        x_fp32 = x.float()
        inv_rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps))
        out = x_fp32 * inv_rms
        out = out * self.weight.float()
        return out.to(dtype=x.dtype)


class TransformerBlock(nn.Module):
    """
    Transformer block with timestep AdaLN and conditioning FiLM modulation.

    Memory tokens:
    - Live in the target token stream (x stream) as a prefix.
    - FiLM modulation is zero-padded over memory positions, so memory tokens are
      not directly FiLM-modulated.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        mlp_hidden = int(hidden_dim * mlp_ratio)

        self.norm1 = RMSNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = RMSNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        # Time AdaLN: (scale, shift) for each of the 3 norms => 6H total.
        self.time_mod = nn.Linear(hidden_dim, hidden_dim * 6)

        # Conditioning FiLM: (scale, shift) for each of 3 norms => 6H per token.
        self.cond_mod = nn.Linear(hidden_dim, hidden_dim * 6)

        # Optional KV normalization for cross-attention stability.
        self.cond_kv_norm = RMSNorm(hidden_dim)

    def forward(
        self,
        x_tokens: torch.Tensor,  # [B, Tx, H] (Tx = M + T)
        cond_tokens: torch.Tensor,  # [B, T, H]
        time_context: torch.Tensor,  # [B, H]
        pad_mask_x: torch.Tensor | None = None,  # [B, Tx], True where padding
        pad_mask_cond: torch.Tensor | None = None,  # [B, T], True where padding
        keep_mask_x: torch.Tensor | None = None,  # [B, Tx], True where keep
    ) -> torch.Tensor:
        """Apply self-attention, cross-attention, and MLP with modulation."""
        t_scale1, t_shift1, t_scale2, t_shift2, t_scale3, t_shift3 = self.time_mod(
            time_context
        ).chunk(6, dim=-1)

        cond_mod = self.cond_mod(cond_tokens)

        # If memory tokens exist, pad FiLM params with zeros at front.
        mem_len = max(0, x_tokens.shape[1] - cond_tokens.shape[1])
        if mem_len > 0:
            cond_mod = F.pad(cond_mod, (0, 0, mem_len, 0))

        c_scale1, c_shift1, c_scale2, c_shift2, c_scale3, c_shift3 = cond_mod.chunk(
            6, dim=-1
        )

        # Self-attention (Q=K=V from x stream).
        h = self.norm1(x_tokens)
        h = _modulate_time(h, t_scale1, t_shift1)
        h = _modulate_film(h, c_scale1, c_shift1)
        attn_pad_mask_x = pad_mask_x.clone() if pad_mask_x is not None else None
        h, _ = self.self_attn(
            h, h, h, need_weights=False, key_padding_mask=attn_pad_mask_x
        )
        x_tokens = x_tokens + self.dropout1(h)

        # Cross-attention (Q from x, KV from conditioning stream).
        q = self.norm2(x_tokens)
        q = _modulate_time(q, t_scale2, t_shift2)
        q = _modulate_film(q, c_scale2, c_shift2)

        kv = self.cond_kv_norm(cond_tokens)
        attn_pad_mask_cond = (
            pad_mask_cond.clone() if pad_mask_cond is not None else None
        )
        h, _ = self.cross_attn(
            q, kv, kv, need_weights=False, key_padding_mask=attn_pad_mask_cond
        )
        x_tokens = x_tokens + self.dropout2(h)

        # MLP block.
        h = self.norm3(x_tokens)
        h = _modulate_time(h, t_scale3, t_shift3)
        h = _modulate_film(h, c_scale3, c_shift3)
        h = self.mlp(h)
        x_tokens = x_tokens + h

        # Hard-zero padded positions after residual updates.
        if keep_mask_x is not None:
            x_tokens = x_tokens * keep_mask_x.to(x_tokens.dtype)[:, :, None]

        return x_tokens
