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


class WaveformTransformerBlock(nn.Module):
    """
    PixelDiT-style waveform refinement block.

    The block keeps dense microtokens inside each waveform patch, modulates them
    from the coarse semantic token, compacts each patch to one temporal token for
    global attention, then expands the attended representation back to
    microtokens.
    """

    def __init__(
        self,
        semantic_dim: int,
        waveform_hidden_dim: int,
        num_micro_tokens: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.semantic_dim = int(semantic_dim)
        self.waveform_hidden_dim = int(waveform_hidden_dim)
        self.num_micro_tokens = int(num_micro_tokens)
        if self.num_micro_tokens <= 0:
            raise ValueError("num_micro_tokens must be > 0")

        mlp_hidden = int(self.waveform_hidden_dim * float(mlp_ratio))
        mlp_hidden = max(1, mlp_hidden)

        self.semantic_mod = nn.Linear(
            self.semantic_dim,
            self.num_micro_tokens * self.waveform_hidden_dim * 6,
        )
        nn.init.zeros_(self.semantic_mod.weight)
        nn.init.zeros_(self.semantic_mod.bias)

        self.semantic_norm = RMSNorm(self.semantic_dim)
        self.norm_attn = RMSNorm(self.waveform_hidden_dim)
        self.cond_norm_attn = RMSNorm(self.waveform_hidden_dim)
        self.compact = nn.Linear(
            self.num_micro_tokens * self.waveform_hidden_dim,
            self.semantic_dim,
        )
        self.cond_compact = nn.Linear(
            self.num_micro_tokens * self.waveform_hidden_dim,
            self.semantic_dim,
        )
        self.compact_norm = RMSNorm(self.semantic_dim)
        self.cross_q_norm = RMSNorm(self.semantic_dim)
        self.cond_kv_norm = RMSNorm(self.semantic_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.semantic_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.dropout_attn = nn.Dropout(float(dropout))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.semantic_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.dropout_cross = nn.Dropout(float(dropout))
        self.expand = nn.Linear(
            self.semantic_dim,
            self.num_micro_tokens * self.waveform_hidden_dim,
        )

        self.norm_mlp = RMSNorm(self.waveform_hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.waveform_hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(mlp_hidden, self.waveform_hidden_dim),
            nn.Dropout(float(dropout)),
        )

    def _semantic_modulation(
        self,
        semantic_tokens: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return per-microtoken scale/shift/gate tensors from semantic tokens."""
        batch_size, num_frames, semantic_dim = semantic_tokens.shape
        if semantic_dim != self.semantic_dim:
            raise ValueError(
                "semantic token dim mismatch: "
                f"expected {self.semantic_dim}, got {semantic_dim}"
            )
        mod = self.semantic_mod(self.semantic_norm(semantic_tokens))
        mod = mod.reshape(
            batch_size,
            num_frames,
            self.num_micro_tokens,
            6,
            self.waveform_hidden_dim,
        )
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = mod.unbind(
            dim=3
        )
        return shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp

    def forward(
        self,
        waveform_tokens: torch.Tensor,  # [B,T,K,Dw]
        semantic_tokens: torch.Tensor,  # [B,T,H]
        cond_waveform_tokens: torch.Tensor | None = None,  # [B,T,K,Dw]
        memory_tokens: torch.Tensor | None = None,  # [B,M,H]
        pos_embed: torch.Tensor | None = None,  # [T,H]
        pad_mask: torch.Tensor | None = None,  # [B,T], True where padding
        keep_mask: torch.Tensor | None = None,  # [B,T], True where valid
    ) -> torch.Tensor:
        """Apply compact temporal attention and microtoken MLP refinement."""
        if waveform_tokens.dim() != 4:
            raise ValueError(
                f"waveform_tokens must be [B,T,K,D], got {tuple(waveform_tokens.shape)}"
            )
        batch_size, num_frames, num_micro, width = waveform_tokens.shape
        if num_micro != self.num_micro_tokens or width != self.waveform_hidden_dim:
            raise ValueError(
                "waveform token shape mismatch: "
                f"expected K={self.num_micro_tokens}, D={self.waveform_hidden_dim}; "
                f"got {num_micro}, {width}"
            )
        if cond_waveform_tokens is not None:
            if cond_waveform_tokens.shape != waveform_tokens.shape:
                raise ValueError(
                    "cond_waveform_tokens must match waveform_tokens shape: "
                    f"expected {tuple(waveform_tokens.shape)}, "
                    f"got {tuple(cond_waveform_tokens.shape)}"
                )
        if semantic_tokens.shape != (batch_size, num_frames, self.semantic_dim):
            raise ValueError(
                "semantic_tokens must be [B,T,H]="
                f"({batch_size},{num_frames},{self.semantic_dim}), "
                f"got {tuple(semantic_tokens.shape)}"
            )
        if memory_tokens is not None:
            if memory_tokens.dim() != 3:
                raise ValueError(
                    "memory_tokens must be [B,M,H], "
                    f"got {tuple(memory_tokens.shape)}"
                )
            if (
                memory_tokens.shape[0] != batch_size
                or memory_tokens.shape[2] != self.semantic_dim
            ):
                raise ValueError(
                    "memory_tokens must match waveform batch/semantic dim: "
                    f"expected B={batch_size}, H={self.semantic_dim}; "
                    f"got {tuple(memory_tokens.shape)}"
                )

        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self._semantic_modulation(semantic_tokens)

        h = self.norm_attn(waveform_tokens)
        h = h * (1.0 + scale_attn) + shift_attn
        compact = self.compact(h.reshape(batch_size, num_frames, num_micro * width))
        compact = self.compact_norm(compact)
        cond_compact = None
        if cond_waveform_tokens is not None:
            cond_h = self.cond_norm_attn(cond_waveform_tokens)
            cond_compact = self.cond_compact(
                cond_h.reshape(batch_size, num_frames, num_micro * width)
            )
        if pos_embed is not None:
            if pos_embed.shape != (num_frames, self.semantic_dim):
                raise ValueError(
                    "pos_embed must be [T,H]="
                    f"({num_frames},{self.semantic_dim}), got {tuple(pos_embed.shape)}"
                )
            compact = (
                compact
                + pos_embed.to(device=compact.device, dtype=compact.dtype)[None, :, :]
            )
            if cond_compact is not None:
                cond_compact = (
                    cond_compact
                    + pos_embed.to(device=cond_compact.device, dtype=cond_compact.dtype)[
                        None, :, :
                    ]
                )

        attn_pad_mask = pad_mask.clone() if pad_mask is not None else None
        compact_for_attn = compact
        memory_len = 0
        if memory_tokens is not None and memory_tokens.shape[1] > 0:
            memory_len = int(memory_tokens.shape[1])
            compact_for_attn = torch.cat(
                [memory_tokens.to(dtype=compact.dtype), compact],
                dim=1,
            )
            if attn_pad_mask is not None:
                memory_pad = torch.zeros(
                    (batch_size, memory_len),
                    device=attn_pad_mask.device,
                    dtype=torch.bool,
                )
                attn_pad_mask = torch.cat([memory_pad, attn_pad_mask], dim=1)

        self_attended, _ = self.self_attn(
            compact_for_attn,
            compact_for_attn,
            compact_for_attn,
            need_weights=False,
            key_padding_mask=attn_pad_mask,
        )
        attended = compact_for_attn + self.dropout_attn(self_attended)
        if memory_len:
            attended = attended[:, memory_len:, :]
        if cond_compact is not None:
            cond_attn_pad_mask = None
            if attn_pad_mask is not None:
                cond_attn_pad_mask = (
                    attn_pad_mask[:, memory_len:] if memory_len else attn_pad_mask
                )
            q = self.cross_q_norm(attended)
            kv = self.cond_kv_norm(cond_compact)
            cross, _ = self.cross_attn(
                q,
                kv,
                kv,
                need_weights=False,
                key_padding_mask=cond_attn_pad_mask,
            )
            attended = attended + self.dropout_cross(cross)
        expanded = self.expand(attended).reshape(
            batch_size,
            num_frames,
            num_micro,
            width,
        )
        waveform_tokens = waveform_tokens + gate_attn * expanded

        h = self.norm_mlp(waveform_tokens)
        h = h * (1.0 + scale_mlp) + shift_mlp
        waveform_tokens = waveform_tokens + gate_mlp * self.mlp(h)

        if keep_mask is not None:
            waveform_tokens = (
                waveform_tokens * keep_mask.to(waveform_tokens.dtype)[:, :, None, None]
            )
        return waveform_tokens
