"""Original latent SpatialDiT architecture for EAR-VAE checkpoint compatibility."""

from __future__ import annotations

from typing import Literal, cast, overload

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .embeddings import positional_embedding_1d, timestep_embedding
from .layers import RMSNorm
from .legacy_layers import LegacyTransformerBlock
from .runtime import is_compiling_runtime


class LegacySpatialDiT(nn.Module):
    """Parameter-compatible legacy latent DiT with the current clean-output contract.

    The stored legacy head predicts rectified-flow velocity. ``forward`` converts
    that velocity to the clean endpoint expected by the current shared training and
    inference stack. This conversion has no parameters, so old checkpoints still
    load strictly and their original vector field is preserved during inference.
    """

    architecture = "legacy_vae"
    representation = "latent"
    raw_prediction_type = "velocity"
    prediction_type = "sample"

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
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.target_channels = int(target_channels)
        self.cond_channels = int(cond_channels)
        self.latent_dim = int(latent_dim)
        # The shared samplers call the model feature axis ``patch_size``.
        self.patch_size = self.latent_dim
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.timestep_embed_dim = int(timestep_embed_dim)
        self.timestep_scale = float(timestep_scale)
        self.max_period = float(max_period)
        self.num_memory_tokens = int(num_memory_tokens)
        self.activation_checkpointing = bool(activation_checkpointing)

        target_token_dim = self.target_channels * self.latent_dim
        cond_token_dim = self.cond_channels * self.latent_dim
        self.target_in = nn.Linear(target_token_dim, self.hidden_dim)
        self.cond_in = nn.Linear(cond_token_dim, self.hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.timestep_embed_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                LegacyTransformerBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
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
        if self.num_memory_tokens <= 0:
            return None
        if self.mem_init is None:
            raise RuntimeError("num_memory_tokens > 0 but mem_init is None")
        return (
            self.mem_init[None, :, :]
            .expand(batch_size, -1, -1)
            .to(device=device, dtype=dtype)
            .contiguous()
        )

    def _as_batch_timesteps(self, t: torch.Tensor, batch_size: int) -> torch.Tensor:
        if t.dim() == 0:
            return t.view(1).repeat(batch_size)
        if t.shape[0] != batch_size:
            raise ValueError(
                f"Timestep batch mismatch: t={tuple(t.shape)} batch={batch_size}"
            )
        return t if t.dim() == 1 else t.reshape(batch_size, -1)[:, 0]

    def _get_positional_embedding(
        self, length: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if is_compiling_runtime():
            return positional_embedding_1d(
                length=length,
                dim=self.hidden_dim,
                device=device,
                max_period=self.max_period,
            ).to(dtype=dtype)
        if (
            self._cached_pos_embed.numel() == 0
            or self._cached_pos_embed.device != device
            or self._cached_pos_embed.dtype != dtype
            or self._cached_pos_embed.shape[0] < length
        ):
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
        if self.num_memory_tokens <= 0:
            raise RuntimeError("num_memory_tokens must be > 0 for mask prefix cache")
        if is_compiling_runtime():
            shape = (batch_size, self.num_memory_tokens)
            return (
                torch.zeros(shape, device=device, dtype=torch.bool),
                torch.ones(shape, device=device, dtype=torch.bool),
            )
        if (
            self._cached_mem_pad_prefix.numel() == 0
            or self._cached_mem_pad_prefix.device != device
            or self._cached_mem_pad_prefix.shape[0] < batch_size
            or self._cached_mem_pad_prefix.shape[1] != self.num_memory_tokens
        ):
            shape = (batch_size, self.num_memory_tokens)
            self._cached_mem_pad_prefix = torch.zeros(
                shape, device=device, dtype=torch.bool
            )
            self._cached_mem_keep_prefix = torch.ones(
                shape, device=device, dtype=torch.bool
            )
        return (
            self._cached_mem_pad_prefix[:batch_size],
            self._cached_mem_keep_prefix[:batch_size],
        )

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
        if zt.dim() != 4 or z_cond.dim() != 4:
            raise ValueError("zt and z_cond must both be [B,C,D,T]")
        batch_size, target_channels, latent_dim, num_frames = zt.shape
        if target_channels != self.target_channels:
            raise ValueError(
                f"Expected target channels={self.target_channels}, got {target_channels}"
            )
        if latent_dim != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {latent_dim}")
        if z_cond.shape != (
            batch_size,
            self.cond_channels,
            self.latent_dim,
            num_frames,
        ):
            raise ValueError(
                "z_cond shape mismatch: "
                f"expected {(batch_size, self.cond_channels, self.latent_dim, num_frames)}, "
                f"got {tuple(z_cond.shape)}"
            )

        frame_pad_mask: torch.Tensor | None = None
        frame_keep_mask: torch.Tensor | None = None
        if valid_mask is not None:
            if valid_mask.shape != (batch_size, num_frames):
                raise ValueError(
                    f"valid_mask must be {(batch_size, num_frames)}, "
                    f"got {tuple(valid_mask.shape)}"
                )
            frame_keep_mask = valid_mask.bool().clone()
            frame_pad_mask = ~frame_keep_mask

        x_tokens = zt.permute(0, 3, 1, 2).reshape(batch_size, num_frames, -1)
        cond_tokens = z_cond.permute(0, 3, 1, 2).reshape(batch_size, num_frames, -1)
        x_tokens = self.target_in(x_tokens)
        cond_tokens = self.cond_in(cond_tokens)
        pos = self._get_positional_embedding(
            length=num_frames, device=zt.device, dtype=x_tokens.dtype
        )
        x_tokens = x_tokens + pos[None, :, :]
        cond_tokens = cond_tokens + pos[None, :, :]

        t_batch = self._as_batch_timesteps(t, batch_size)
        t_embed = timestep_embedding(
            timesteps=t_batch * self.timestep_scale,
            dim=self.timestep_embed_dim,
            max_period=self.max_period,
        ).to(dtype=x_tokens.dtype)
        time_context = self.time_mlp(t_embed)

        memory_tokens = self.num_memory_tokens
        mem_out: torch.Tensor | None = None
        if memory_tokens > 0:
            if mem is None:
                mem = self.init_memory(batch_size, zt.device, x_tokens.dtype)
            if mem is None or mem.shape != (
                batch_size,
                memory_tokens,
                self.hidden_dim,
            ):
                raise ValueError(
                    "mem must match "
                    f"{(batch_size, memory_tokens, self.hidden_dim)}, "
                    f"got {None if mem is None else tuple(mem.shape)}"
                )
            x_all = torch.cat([mem, x_tokens], dim=1)
            if frame_pad_mask is not None:
                mem_pad, mem_keep = self._get_mem_mask_prefix(batch_size, zt.device)
                assert frame_keep_mask is not None
                pad_mask_x = torch.cat([mem_pad, frame_pad_mask], dim=1)
                keep_mask_x = torch.cat([mem_keep, frame_keep_mask], dim=1)
            else:
                pad_mask_x = None
                keep_mask_x = None
        else:
            x_all = x_tokens
            pad_mask_x = frame_pad_mask
            keep_mask_x = frame_keep_mask

        for block in self.blocks:
            legacy_block = cast(LegacyTransformerBlock, block)
            if (
                self.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                x_all = checkpoint(
                    legacy_block,
                    x_all,
                    cond_tokens,
                    time_context,
                    pad_mask_x,
                    frame_pad_mask,
                    keep_mask_x,
                    use_reentrant=False,
                )
            else:
                x_all = legacy_block(
                    x_tokens=x_all,
                    cond_tokens=cond_tokens,
                    time_context=time_context,
                    pad_mask_x=pad_mask_x,
                    pad_mask_cond=frame_pad_mask,
                    keep_mask_x=keep_mask_x,
                )

        if memory_tokens > 0:
            mem_out = x_all[:, :memory_tokens, :].contiguous()
            x_tokens = x_all[:, memory_tokens:, :].contiguous()
        else:
            x_tokens = x_all

        velocity_tokens = self.final_proj(self.final_norm(x_tokens))
        if frame_keep_mask is not None:
            velocity_tokens = (
                velocity_tokens * frame_keep_mask.to(velocity_tokens.dtype)[:, :, None]
            )
        velocity = (
            velocity_tokens.reshape(
                batch_size, num_frames, self.target_channels, self.latent_dim
            )
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        # Normalize the legacy velocity head to the current clean-endpoint API.
        remaining = (1.0 - t_batch.to(device=zt.device, dtype=zt.dtype)).reshape(
            batch_size, 1, 1, 1
        )
        clean_prediction = zt + remaining * velocity.to(dtype=zt.dtype)
        if frame_keep_mask is not None:
            clean_prediction = (
                clean_prediction
                * frame_keep_mask.to(clean_prediction.dtype)[:, None, None, :]
            )
        if return_mem:
            return clean_prediction, mem_out
        return clean_prediction


__all__ = ["LegacySpatialDiT"]
