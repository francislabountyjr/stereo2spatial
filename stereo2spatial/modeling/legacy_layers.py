"""Parameter-compatible transformer layers for legacy EAR-VAE checkpoints."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm


def _modulate_time(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


def _modulate_film(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class LegacyTransformerBlock(nn.Module):
    """Original fused-MHA SpatialDiT block used by legacy latent checkpoints."""

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

        self.time_mod = nn.Linear(hidden_dim, hidden_dim * 6)
        self.cond_mod = nn.Linear(hidden_dim, hidden_dim * 6)
        self.cond_kv_norm = RMSNorm(hidden_dim)

    def forward(
        self,
        x_tokens: torch.Tensor,
        cond_tokens: torch.Tensor,
        time_context: torch.Tensor,
        pad_mask_x: torch.Tensor | None = None,
        pad_mask_cond: torch.Tensor | None = None,
        keep_mask_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the original self-attention, cross-attention, and MLP path."""
        t_scale1, t_shift1, t_scale2, t_shift2, t_scale3, t_shift3 = self.time_mod(
            time_context
        ).chunk(6, dim=-1)

        cond_mod = self.cond_mod(cond_tokens)
        mem_len = max(0, x_tokens.shape[1] - cond_tokens.shape[1])
        if mem_len > 0:
            cond_mod = F.pad(cond_mod, (0, 0, mem_len, 0))
        c_scale1, c_shift1, c_scale2, c_shift2, c_scale3, c_shift3 = cond_mod.chunk(
            6, dim=-1
        )

        h = self.norm1(x_tokens)
        h = _modulate_time(h, t_scale1, t_shift1)
        h = _modulate_film(h, c_scale1, c_shift1)
        h, _ = self.self_attn(
            h,
            h,
            h,
            need_weights=False,
            key_padding_mask=pad_mask_x.clone() if pad_mask_x is not None else None,
        )
        x_tokens = x_tokens + self.dropout1(h)

        q = self.norm2(x_tokens)
        q = _modulate_time(q, t_scale2, t_shift2)
        q = _modulate_film(q, c_scale2, c_shift2)
        kv = self.cond_kv_norm(cond_tokens)
        h, _ = self.cross_attn(
            q,
            kv,
            kv,
            need_weights=False,
            key_padding_mask=(
                pad_mask_cond.clone() if pad_mask_cond is not None else None
            ),
        )
        x_tokens = x_tokens + self.dropout2(h)

        h = self.norm3(x_tokens)
        h = _modulate_time(h, t_scale3, t_shift3)
        h = _modulate_film(h, c_scale3, c_shift3)
        x_tokens = x_tokens + self.mlp(h)

        if keep_mask_x is not None:
            x_tokens = x_tokens * keep_mask_x.to(x_tokens.dtype)[:, :, None]
        return x_tokens


__all__ = ["LegacyTransformerBlock"]
