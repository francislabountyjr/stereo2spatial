"""Embedding utilities for diffusion timesteps and sequence positions."""

from __future__ import annotations

import math

import torch


def timestep_embedding(
    timesteps: torch.Tensor, dim: int, max_period: float = 10000.0
) -> torch.Tensor:
    """
    Build sinusoidal timestep embeddings.

    Args:
        timesteps: Float tensor with shape `[B]`.
        dim: Output embedding width.
        max_period: Maximum sinusoid period.

    Returns:
        Tensor with shape `[B, dim]`.
    """
    if timesteps.dim() != 1:
        raise ValueError(f"timesteps must have shape [B], got {tuple(timesteps.shape)}")

    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / max(half, 1)
    freqs = torch.exp(exponent)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def positional_embedding_1d(
    length: int, dim: int, device: torch.device, max_period: float = 10000.0
) -> torch.Tensor:
    """
    Build sinusoidal positional embeddings for indices `0..length-1`.

    Returns:
        Tensor with shape `[length, dim]`.
    """
    pos = torch.arange(length, device=device, dtype=torch.float32)
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half, dtype=torch.float32, device=device
    )
    exponent = exponent / max(half, 1)
    freqs = torch.exp(exponent)
    args = pos[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def rotary_embedding_1d(
    *,
    positions: torch.Tensor,
    dim: int,
    max_period: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build rotary embedding cos/sin tensors for attention head positions.

    Args:
        positions: Float tensor with shape `[N]`.
        dim: Attention head width. Must be even.
        max_period: Rotary frequency base.

    Returns:
        `(cos, sin)` each shaped `[1, 1, N, dim / 2]`.
    """
    if dim <= 0 or dim % 2 != 0:
        raise ValueError("rotary dim must be a positive even integer")
    if positions.dim() != 1:
        raise ValueError(f"positions must be [N], got {tuple(positions.shape)}")

    half = dim // 2
    exponent = torch.arange(
        start=0,
        end=dim,
        step=2,
        dtype=torch.float32,
        device=positions.device,
    )
    exponent = exponent / float(dim)
    freqs = torch.exp(-math.log(max_period) * exponent)
    angles = positions.float()[:, None] * freqs[None, :]
    cos = torch.cos(angles)[None, None, :, :]
    sin = torch.sin(angles)[None, None, :, :]
    if cos.shape[-1] != half:
        raise RuntimeError("unexpected rotary embedding shape")
    return cos, sin


def apply_rotary_embedding(
    x: torch.Tensor,
    rope: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    """Apply rotary embeddings to a `[B, heads, tokens, head_dim]` tensor."""
    if rope is None:
        return x
    cos, sin = rope
    if x.shape[-1] % 2 != 0:
        raise ValueError("rotary embedding requires an even attention head dimension")
    if cos.shape[-2] != x.shape[-2] or sin.shape[-2] != x.shape[-2]:
        raise ValueError(
            "rotary length mismatch: "
            f"x tokens={x.shape[-2]}, cos={cos.shape[-2]}, sin={sin.shape[-2]}"
        )
    if cos.shape[-1] != x.shape[-1] // 2 or sin.shape[-1] != x.shape[-1] // 2:
        raise ValueError(
            "rotary head-dim mismatch: "
            f"x head_dim={x.shape[-1]}, cos={cos.shape[-1]}, sin={sin.shape[-1]}"
        )

    x_float = x.float()
    even = x_float[..., 0::2]
    odd = x_float[..., 1::2]
    cos = cos.to(device=x.device)
    sin = sin.to(device=x.device)
    out = torch.empty_like(x_float)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out.to(dtype=x.dtype)
