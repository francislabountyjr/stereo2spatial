"""SpatialDiT backbone and supporting transformer layers for latent modeling."""

from __future__ import annotations

from typing import Literal, cast, overload

import torch
import torch.nn as nn

from .embeddings import positional_embedding_1d, timestep_embedding
from .layers import RMSNorm, TransformerBlock
from .runtime import is_compiling_runtime


class SpatialDiT(nn.Module):
    """
    Conditional latent vector-field backbone with optional memory tokens.

    Input:
    - zt: [B, C_target, D_latent, T]
    - t: [B] or broadcastable to [B]
    - z_cond: [B, C_cond, D_latent, T]
    - valid_mask (optional): [B, T] True where valid (non-padding)
    - mem (optional): [B, M, H]

    Output:
    - velocity prediction: [B, C_target, D_latent, T]
    - optionally (velocity, mem_out) when return_mem=True
    """

    def __init__(
        self,
        target_channels: int,
        cond_channels: int,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        timestep_embed_dim: int,
        timestep_scale: float,
        max_period: float,
        num_memory_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.target_channels = int(target_channels)
        self.cond_channels = int(cond_channels)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.timestep_embed_dim = int(timestep_embed_dim)
        self.timestep_scale = float(timestep_scale)
        self.max_period = float(max_period)
        self.num_memory_tokens = int(num_memory_tokens)

        target_token_dim = self.target_channels * self.latent_dim
        cond_token_dim = self.cond_channels * self.latent_dim

        # Tokenize per frame: flatten (C * D)
        self.target_in = nn.Linear(target_token_dim, self.hidden_dim)
        self.cond_in = nn.Linear(cond_token_dim, self.hidden_dim)

        # Time embedding -> context vector
        self.time_mlp = nn.Sequential(
            nn.Linear(self.timestep_embed_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(self.hidden_dim)
        self.final_proj = nn.Linear(self.hidden_dim, target_token_dim)

        self.mem_init: nn.Parameter | None
        if self.num_memory_tokens > 0:
            self.mem_init = nn.Parameter(
                torch.randn(self.num_memory_tokens, self.hidden_dim) * 0.02
            )
        else:
            self.mem_init = None

        # Non-persistent runtime caches for deterministic helper tensors.
        self._cached_pos_embed: torch.Tensor
        self._cached_mem_pad_prefix: torch.Tensor
        self._cached_mem_keep_prefix: torch.Tensor
        self.register_buffer("_cached_pos_embed", torch.empty(0), persistent=False)
        self.register_buffer(
            "_cached_mem_pad_prefix", torch.empty(0, dtype=torch.bool), persistent=False
        )
        self.register_buffer(
            "_cached_mem_keep_prefix",
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )

    def init_memory(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor | None:
        """Return learned memory tokens expanded for the requested batch, if enabled."""
        if self.num_memory_tokens <= 0:
            return None
        if self.mem_init is None:
            raise RuntimeError("num_memory_tokens > 0 but mem_init is None")
        mem = self.mem_init[None, :, :].expand(batch_size, -1, -1)
        return mem.to(device=device, dtype=dtype).contiguous()

    def _as_batch_timesteps(self, t: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Normalize scalar/vector timestep input to ``[B]`` aligned with model batch."""
        if t.dim() == 0:
            t = t.view(1).repeat(batch_size)
        elif t.dim() == 1:
            if t.shape[0] != batch_size:
                raise ValueError(
                    f"Timestep batch mismatch: t={tuple(t.shape)} batch={batch_size}"
                )
        else:
            if t.shape[0] != batch_size:
                raise ValueError(
                    f"Timestep batch mismatch: t={tuple(t.shape)} batch={batch_size}"
                )
            t = t.reshape(batch_size, -1)[:, 0]
        return t

    def _get_positional_embedding(
        self, length: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return frame positional embeddings, reusing runtime cache when safe."""
        # Do not mutate module-level caches while compiling/cudagraphing.
        if is_compiling_runtime():
            return positional_embedding_1d(
                length=length,
                dim=self.hidden_dim,
                device=device,
                max_period=self.max_period,
            ).to(dtype=dtype)

        needs_rebuild = (
            self._cached_pos_embed.numel() == 0
            or self._cached_pos_embed.device != device
            or self._cached_pos_embed.dtype != dtype
            or self._cached_pos_embed.shape[0] < length
        )
        if needs_rebuild:
            self._cached_pos_embed = positional_embedding_1d(
                length=length,
                dim=self.hidden_dim,
                device=device,
                max_period=self.max_period,
            ).to(dtype=dtype)
        return self._cached_pos_embed[:length]

    def _get_mem_mask_prefix(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached memory-token mask prefixes for pad/keep mask composition."""
        if self.num_memory_tokens <= 0:
            raise RuntimeError("num_memory_tokens must be > 0 for mask prefix cache")

        # Do not persist tensors created inside compiled graphs.
        if is_compiling_runtime():
            shape = (batch_size, self.num_memory_tokens)
            return (
                torch.zeros(shape, device=device, dtype=torch.bool),
                torch.ones(shape, device=device, dtype=torch.bool),
            )

        needs_rebuild = (
            self._cached_mem_pad_prefix.numel() == 0
            or self._cached_mem_pad_prefix.device != device
            or self._cached_mem_pad_prefix.shape[0] < batch_size
            or self._cached_mem_pad_prefix.shape[1] != self.num_memory_tokens
        )
        if needs_rebuild:
            shape = (batch_size, self.num_memory_tokens)
            self._cached_mem_pad_prefix = torch.zeros(
                shape, device=device, dtype=torch.bool
            )
            self._cached_mem_keep_prefix = torch.ones(
                shape, device=device, dtype=torch.bool
            )

        mem_pad = self._cached_mem_pad_prefix[:batch_size]
        mem_keep = self._cached_mem_keep_prefix[:batch_size]
        return mem_pad, mem_keep

    @overload
    def forward(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        mem: torch.Tensor | None = None,
        return_mem: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        mem: torch.Tensor | None = None,
        return_mem: Literal[True] = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

    def forward(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        mem: torch.Tensor | None = None,
        return_mem: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Predict latent velocity and optionally updated memory tokens."""
        if zt.dim() != 4:
            raise ValueError(f"zt must be [B,C,D,T], got {tuple(zt.shape)}")
        if z_cond.dim() != 4:
            raise ValueError(f"z_cond must be [B,C,D,T], got {tuple(z_cond.shape)}")

        batch_size, target_channels, latent_dim, num_frames = zt.shape

        if target_channels != self.target_channels:
            raise ValueError(
                f"Expected target channels={self.target_channels}, got {target_channels}"
            )
        if latent_dim != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {latent_dim}")

        if z_cond.shape[0] != batch_size or z_cond.shape[3] != num_frames:
            raise ValueError(
                "z_cond batch/time mismatch: "
                f"zt={tuple(zt.shape)} z_cond={tuple(z_cond.shape)}"
            )
        if z_cond.shape[1] != self.cond_channels:
            raise ValueError(
                f"Expected cond channels={self.cond_channels}, got {z_cond.shape[1]}"
            )
        if z_cond.shape[2] != self.latent_dim:
            raise ValueError(
                f"Expected cond latent_dim={self.latent_dim}, got {z_cond.shape[2]}"
            )

        frame_pad_mask: torch.Tensor | None = None
        frame_keep_mask: torch.Tensor | None = None
        if valid_mask is not None:
            if valid_mask.shape != (batch_size, num_frames):
                raise ValueError(
                    f"valid_mask must be [B,T]=({batch_size},{num_frames}), got {tuple(valid_mask.shape)}"
                )
            # Keep a private copy so any backend-side in-place behavior on masks
            # cannot mutate caller-owned tensors across windowed forwards.
            frame_keep_mask = valid_mask.bool().clone()
            frame_pad_mask = ~frame_keep_mask  # True where padding

        # [B, T, C*D]
        x_tokens = zt.permute(0, 3, 1, 2).reshape(batch_size, num_frames, -1)
        cond_tokens = z_cond.permute(0, 3, 1, 2).reshape(batch_size, num_frames, -1)

        # project to hidden
        x_tokens = self.target_in(x_tokens)
        cond_tokens = self.cond_in(cond_tokens)

        # positional embedding over frames (ONLY for frame tokens)
        pos = self._get_positional_embedding(
            length=num_frames,
            device=zt.device,
            dtype=x_tokens.dtype,
        )
        x_tokens = x_tokens + pos[None, :, :]
        cond_tokens = cond_tokens + pos[None, :, :]

        # time context
        t_batch = self._as_batch_timesteps(t, batch_size)
        t_embed = timestep_embedding(
            timesteps=t_batch * self.timestep_scale,
            dim=self.timestep_embed_dim,
            max_period=self.max_period,
        ).to(dtype=x_tokens.dtype)
        time_context = self.time_mlp(t_embed)  # [B, H]

        # prepend memory tokens (x stream only)
        M = self.num_memory_tokens
        mem_out: torch.Tensor | None = None
        if M > 0:
            if mem is None:
                mem = self.init_memory(
                    batch_size, device=zt.device, dtype=x_tokens.dtype
                )
            if mem is None:
                raise RuntimeError("num_memory_tokens>0 but init_memory returned None")
            if mem.shape != (batch_size, M, self.hidden_dim):
                raise ValueError(
                    f"mem must be [B,M,H]=({batch_size},{M},{self.hidden_dim}), got {tuple(mem.shape)}"
                )

            x_all = torch.cat([mem, x_tokens], dim=1)  # [B, M+T, H]

            if frame_pad_mask is not None:
                mem_pad, mem_keep = self._get_mem_mask_prefix(
                    batch_size=batch_size, device=zt.device
                )
                assert frame_keep_mask is not None
                pad_mask_x = torch.cat([mem_pad, frame_pad_mask], dim=1)  # [B, M+T]
                keep_mask_x = torch.cat(
                    [mem_keep, frame_keep_mask],
                    dim=1,
                )
            else:
                pad_mask_x = None
                keep_mask_x = None
        else:
            x_all = x_tokens
            pad_mask_x = frame_pad_mask
            keep_mask_x = frame_keep_mask

        # blocks
        for block in self.blocks:
            x_all = cast(TransformerBlock, block)(
                x_tokens=x_all,
                cond_tokens=cond_tokens,
                time_context=time_context,
                pad_mask_x=pad_mask_x,
                pad_mask_cond=frame_pad_mask,
                keep_mask_x=keep_mask_x,
            )

        # split memory + frames
        if M > 0:
            mem_out = x_all[:, :M, :].contiguous()
            x_tokens = x_all[:, M:, :].contiguous()
        else:
            x_tokens = x_all

        # project back to [B, C, D, T]
        velocity_tokens = self.final_proj(self.final_norm(x_tokens))

        # Re-mask padded frames
        if frame_keep_mask is not None:
            velocity_tokens = (
                velocity_tokens * frame_keep_mask.to(velocity_tokens.dtype)[:, :, None]
            )

        velocity = cast(
            torch.Tensor,
            velocity_tokens.reshape(
                batch_size, num_frames, self.target_channels, self.latent_dim
            )
            .permute(0, 2, 3, 1)
            .contiguous(),
        )

        if return_mem:
            return velocity, mem_out
        return velocity
