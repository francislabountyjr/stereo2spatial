"""Transformer layers used by SpatialDiT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import apply_rotary_embedding

RotaryEmbedding = tuple[torch.Tensor, torch.Tensor]


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


class FinalOutputBlock(nn.Module):
    """Conditioned temporal output head for clean waveform patch prediction."""

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        *,
        kernel_size: int = 7,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.kernel_size = int(kernel_size)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.out_dim <= 0:
            raise ValueError("out_dim must be > 0")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.context_norm = RMSNorm(self.hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
        )
        self.norm = RMSNorm(self.hidden_dim)
        self.conv = nn.Conv1d(
            self.hidden_dim,
            self.out_dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )

        if zero_init:
            final_linear = self.adaLN_modulation[-1]
            if isinstance(final_linear, nn.Linear):
                nn.init.zeros_(final_linear.weight)
                nn.init.zeros_(final_linear.bias)
            nn.init.zeros_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Apply conditioned normalization and temporal convolution."""
        if tokens.dim() != 3:
            raise ValueError(f"tokens must be [B,T,H], got {tuple(tokens.shape)}")
        batch_size, num_frames, hidden_dim = tokens.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"token hidden dim mismatch: expected {self.hidden_dim}, got {hidden_dim}"
            )
        if conditioning.dim() == 2:
            conditioning = conditioning[:, None, :].expand(batch_size, num_frames, -1)
        if conditioning.shape != (batch_size, num_frames, self.hidden_dim):
            raise ValueError(
                "conditioning must be [B,T,H] or [B,H]: "
                f"expected ({batch_size},{num_frames},{self.hidden_dim}), "
                f"got {tuple(conditioning.shape)}"
            )

        shift, scale = self.adaLN_modulation(self.context_norm(conditioning)).chunk(
            2, dim=-1
        )
        h = self.norm(tokens)
        h = h * (1.0 + scale) + shift
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        return h


class RotaryAttention(nn.Module):
    """Multi-head attention with optional RoPE on Q/K and SDPA execution."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.head_dim = self.hidden_dim // self.num_heads

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        return (
            x.reshape(batch_size, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_tokens, _ = x.shape
        return (
            x.transpose(1, 2)
            .reshape(batch_size, num_tokens, self.hidden_dim)
            .contiguous()
        )

    def project_key_value(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_rope: RotaryEmbedding | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project and normalize reusable attention K/V tensors."""
        k = self.k_norm(self._split_heads(self.k_proj(key)))
        v = self._split_heads(self.v_proj(value))
        k = apply_rotary_embedding(k, key_rope)
        return k.contiguous(), v.contiguous()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        query_rope: RotaryEmbedding | None = None,
        key_rope: RotaryEmbedding | None = None,
        precomputed_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Apply attention. ``key_padding_mask`` is True where key tokens pad."""
        q = self.q_norm(self._split_heads(self.q_proj(query)))
        q = apply_rotary_embedding(q, query_rope)
        if precomputed_kv is None:
            k, v = self.project_key_value(key, value, key_rope=key_rope)
        else:
            k, v = precomputed_kv

        attn_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.shape != (query.shape[0], key.shape[1]):
                raise ValueError(
                    "key_padding_mask must be [B,S]="
                    f"({query.shape[0]},{key.shape[1]}), got {tuple(key_padding_mask.shape)}"
                )
            attn_mask = ~key_padding_mask[:, None, None, :].bool()

        out = F.scaled_dot_product_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(self._merge_heads(out))


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
        self.self_attn = RotaryAttention(hidden_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(hidden_dim)
        self.cross_attn = RotaryAttention(hidden_dim, num_heads, dropout)
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

    def build_conditioning_cache(
        self,
        *,
        cond_tokens: torch.Tensor,
        mem_len: int,
        rope_cond: RotaryEmbedding | None = None,
    ) -> dict[str, torch.Tensor]:
        """Precompute conditioning tensors reused across inference solver substeps."""
        cond_mod = self.cond_mod(cond_tokens)
        if mem_len > 0:
            cond_mod = F.pad(cond_mod, (0, 0, int(mem_len), 0))
        kv = self.cond_kv_norm(cond_tokens)
        cross_k, cross_v = self.cross_attn.project_key_value(
            kv,
            kv,
            key_rope=rope_cond,
        )
        return {
            "cond_mod": cond_mod,
            "cross_k": cross_k,
            "cross_v": cross_v,
        }

    def forward(
        self,
        x_tokens: torch.Tensor,  # [B, Tx, H] (Tx = M + T)
        cond_tokens: torch.Tensor,  # [B, T, H]
        time_context: torch.Tensor,  # [B, H]
        pad_mask_x: torch.Tensor | None = None,  # [B, Tx], True where padding
        pad_mask_cond: torch.Tensor | None = None,  # [B, T], True where padding
        keep_mask_x: torch.Tensor | None = None,  # [B, Tx], True where keep
        rope_x: RotaryEmbedding | None = None,
        rope_cond: RotaryEmbedding | None = None,
        conditioning_cache: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Apply self-attention, cross-attention, and MLP with modulation."""
        t_scale1, t_shift1, t_scale2, t_shift2, t_scale3, t_shift3 = self.time_mod(
            time_context
        ).chunk(6, dim=-1)

        if conditioning_cache is None:
            # If memory tokens exist, pad FiLM params with zeros at front.
            mem_len = max(0, x_tokens.shape[1] - cond_tokens.shape[1])
            cond_mod = self.cond_mod(cond_tokens)
            if mem_len > 0:
                cond_mod = F.pad(cond_mod, (0, 0, mem_len, 0))
            kv = self.cond_kv_norm(cond_tokens)
            cross_kv = None
        else:
            cond_mod = conditioning_cache["cond_mod"]
            kv = cond_tokens
            cross_kv = (
                conditioning_cache["cross_k"],
                conditioning_cache["cross_v"],
            )

        c_scale1, c_shift1, c_scale2, c_shift2, c_scale3, c_shift3 = cond_mod.chunk(
            6, dim=-1
        )

        # Self-attention (Q=K=V from x stream).
        h = self.norm1(x_tokens)
        h = _modulate_time(h, t_scale1, t_shift1)
        h = _modulate_film(h, c_scale1, c_shift1)
        attn_pad_mask_x = pad_mask_x.clone() if pad_mask_x is not None else None
        h = self.self_attn(
            h,
            h,
            h,
            key_padding_mask=attn_pad_mask_x,
            query_rope=rope_x,
            key_rope=rope_x,
        )
        x_tokens = x_tokens + self.dropout1(h)

        # Cross-attention (Q from x, KV from conditioning stream).
        q = self.norm2(x_tokens)
        q = _modulate_time(q, t_scale2, t_shift2)
        q = _modulate_film(q, c_scale2, c_shift2)

        attn_pad_mask_cond = (
            pad_mask_cond.clone() if pad_mask_cond is not None else None
        )
        h = self.cross_attn(
            q,
            kv,
            kv,
            key_padding_mask=attn_pad_mask_cond,
            query_rope=rope_x,
            key_rope=rope_cond,
            precomputed_kv=cross_kv,
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
        self.cond_mod = nn.Linear(
            self.waveform_hidden_dim, self.waveform_hidden_dim * 4
        )
        nn.init.zeros_(self.cond_mod.weight)
        nn.init.zeros_(self.cond_mod.bias)

        self.semantic_norm = RMSNorm(self.semantic_dim)
        self.norm_attn = RMSNorm(self.waveform_hidden_dim)
        self.cond_norm_attn = RMSNorm(self.waveform_hidden_dim)
        self.cond_norm_mod = RMSNorm(self.waveform_hidden_dim)
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
        self.self_attn = RotaryAttention(self.semantic_dim, int(num_heads), dropout)
        self.dropout_attn = nn.Dropout(float(dropout))
        self.cross_attn = RotaryAttention(self.semantic_dim, int(num_heads), dropout)
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

    def _conditioning_modulation(
        self,
        cond_waveform_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-microtoken source FiLM tensors."""
        mod = self.cond_mod(self.cond_norm_mod(cond_waveform_tokens))
        shift_attn, scale_attn, shift_mlp, scale_mlp = mod.chunk(4, dim=-1)
        return shift_attn, scale_attn, shift_mlp, scale_mlp

    def build_conditioning_cache(
        self,
        *,
        cond_waveform_tokens: torch.Tensor,
        pos_embed: torch.Tensor | None = None,
        rope_frames: RotaryEmbedding | None = None,
    ) -> dict[str, torch.Tensor]:
        """Precompute source-conditioning tensors reused across solver substeps."""
        (
            cond_shift_attn,
            cond_scale_attn,
            cond_shift_mlp,
            cond_scale_mlp,
        ) = self._conditioning_modulation(cond_waveform_tokens)
        batch_size, num_frames, num_micro, width = cond_waveform_tokens.shape
        cond_h = self.cond_norm_attn(cond_waveform_tokens)
        cond_compact = self.cond_compact(
            cond_h.reshape(batch_size, num_frames, num_micro * width)
        )
        if pos_embed is not None:
            cond_compact = (
                cond_compact
                + pos_embed.to(device=cond_compact.device, dtype=cond_compact.dtype)[
                    None,
                    :,
                    :,
                ]
            )
        kv = self.cond_kv_norm(cond_compact)
        cross_k, cross_v = self.cross_attn.project_key_value(
            kv,
            kv,
            key_rope=rope_frames,
        )
        return {
            "cond_shift_attn": cond_shift_attn,
            "cond_scale_attn": cond_scale_attn,
            "cond_shift_mlp": cond_shift_mlp,
            "cond_scale_mlp": cond_scale_mlp,
            "cross_k": cross_k,
            "cross_v": cross_v,
        }

    def forward(
        self,
        waveform_tokens: torch.Tensor,  # [B,T,K,Dw]
        semantic_tokens: torch.Tensor,  # [B,T,H]
        cond_waveform_tokens: torch.Tensor | None = None,  # [B,T,K,Dw]
        memory_tokens: torch.Tensor | None = None,  # [B,M,H]
        pos_embed: torch.Tensor | None = None,  # [T,H]
        pad_mask: torch.Tensor | None = None,  # [B,T], True where padding
        keep_mask: torch.Tensor | None = None,  # [B,T], True where valid
        rope_self: RotaryEmbedding | None = None,
        rope_frames: RotaryEmbedding | None = None,
        conditioning_cache: dict[str, torch.Tensor] | None = None,
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
        cond_shift_attn = cond_scale_attn = None
        cond_shift_mlp = cond_scale_mlp = None
        cond_cross_kv = None
        if conditioning_cache is not None:
            cond_shift_attn = conditioning_cache["cond_shift_attn"]
            cond_scale_attn = conditioning_cache["cond_scale_attn"]
            cond_shift_mlp = conditioning_cache["cond_shift_mlp"]
            cond_scale_mlp = conditioning_cache["cond_scale_mlp"]
            cond_cross_kv = (
                conditioning_cache["cross_k"],
                conditioning_cache["cross_v"],
            )
        elif cond_waveform_tokens is not None:
            (
                cond_shift_attn,
                cond_scale_attn,
                cond_shift_mlp,
                cond_scale_mlp,
            ) = self._conditioning_modulation(cond_waveform_tokens)

        h = self.norm_attn(waveform_tokens)
        h = h * (1.0 + scale_attn) + shift_attn
        if cond_shift_attn is not None and cond_scale_attn is not None:
            h = h * (1.0 + cond_scale_attn) + cond_shift_attn
        compact = self.compact(h.reshape(batch_size, num_frames, num_micro * width))
        compact = self.compact_norm(compact)
        cond_compact = None
        if cond_waveform_tokens is not None and conditioning_cache is None:
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
                    + pos_embed.to(
                        device=cond_compact.device, dtype=cond_compact.dtype
                    )[None, :, :]
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

        self_attended = self.self_attn(
            compact_for_attn,
            compact_for_attn,
            compact_for_attn,
            key_padding_mask=attn_pad_mask,
            query_rope=rope_self,
            key_rope=rope_self,
        )
        attended = compact_for_attn + self.dropout_attn(self_attended)
        if memory_len:
            attended = attended[:, memory_len:, :]
        if cond_compact is not None or cond_cross_kv is not None:
            cond_attn_pad_mask = None
            if attn_pad_mask is not None:
                cond_attn_pad_mask = (
                    attn_pad_mask[:, memory_len:] if memory_len else attn_pad_mask
                )
            q = self.cross_q_norm(attended)
            if cond_cross_kv is None:
                assert cond_compact is not None
                kv = self.cond_kv_norm(cond_compact)
            else:
                kv = cond_compact if cond_compact is not None else attended
            cross = self.cross_attn(
                q,
                kv,
                kv,
                key_padding_mask=cond_attn_pad_mask,
                query_rope=rope_frames,
                key_rope=rope_frames,
                precomputed_kv=cond_cross_kv,
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
        if cond_shift_mlp is not None and cond_scale_mlp is not None:
            h = h * (1.0 + cond_scale_mlp) + cond_shift_mlp
        waveform_tokens = waveform_tokens + gate_mlp * self.mlp(h)

        if keep_mask is not None:
            waveform_tokens = (
                waveform_tokens * keep_mask.to(waveform_tokens.dtype)[:, :, None, None]
            )
        return waveform_tokens
