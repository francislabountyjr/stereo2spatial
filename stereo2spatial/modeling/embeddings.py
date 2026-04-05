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

