"""SpatialDiT backbone and supporting transformer layers for signal modeling."""

from __future__ import annotations

from typing import Literal, cast, overload

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .embeddings import positional_embedding_1d, rotary_embedding_1d, timestep_embedding
from .layers import (
    FinalOutputBlock,
    RMSNorm,
    TransformerBlock,
    WaveformTransformerBlock,
)
from .runtime import is_compiling_runtime


class SpatialDiT(nn.Module):
    """
    Conditional signal vector-field backbone with optional memory tokens.

    Input:
    - zt: [B, C_target, P_patch, T]
    - t: [B] or broadcastable to [B]
    - z_cond: [B, C_cond, P_patch, T]
    - valid_mask (optional): [B, T] True where valid (non-padding)
    - mem (optional): [B, M, H]

    Output:
    - clean target prediction: [B, C_target, P_patch, T]
    - optionally (velocity, mem_out) when return_mem=True
    """

    architecture = "waveform"
    representation = "waveform"
    raw_prediction_type = "sample"
    prediction_type = "sample"

    def __init__(
        self,
        target_channels: int,
        cond_channels: int,
        patch_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        timestep_embed_dim: int,
        timestep_scale: float,
        max_period: float,
        num_memory_tokens: int = 0,
        mix_style_dim: int = 0,
        amplitude_gain_conditioning: bool = False,
        waveform_level_depth: int = 0,
        waveform_micro_patch_size: int = 16,
        waveform_hidden_dim: int = 16,
        waveform_num_heads: int | None = None,
        waveform_mlp_ratio: float = 2.0,
        final_output_kernel_size: int = 7,
        final_output_zero_init: bool = False,
        rope_enabled: bool = True,
        rope_theta: float = 10000.0,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.target_channels = int(target_channels)
        self.cond_channels = int(cond_channels)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.timestep_embed_dim = int(timestep_embed_dim)
        self.timestep_scale = float(timestep_scale)
        self.max_period = float(max_period)
        self.num_memory_tokens = int(num_memory_tokens)
        self.mix_style_dim = int(mix_style_dim)
        self.amplitude_gain_conditioning = bool(amplitude_gain_conditioning)
        self.waveform_level_depth = int(waveform_level_depth)
        self.waveform_micro_patch_size = int(waveform_micro_patch_size)
        self.waveform_hidden_dim = int(waveform_hidden_dim)
        self.waveform_num_heads = (
            int(num_heads) if waveform_num_heads is None else int(waveform_num_heads)
        )
        self.waveform_mlp_ratio = float(waveform_mlp_ratio)
        self.final_output_kernel_size = int(final_output_kernel_size)
        self.final_output_zero_init = bool(final_output_zero_init)
        self.rope_enabled = bool(rope_enabled)
        self.rope_theta = float(rope_theta)
        self.activation_checkpointing = bool(activation_checkpointing)

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads "
                f"({self.hidden_dim} % {self.num_heads} != 0)"
            )
        if self.waveform_level_depth < 0:
            raise ValueError("waveform_level_depth must be >= 0")
        if self.final_output_kernel_size <= 0 or self.final_output_kernel_size % 2 == 0:
            raise ValueError("final_output_kernel_size must be a positive odd integer")
        if self.rope_theta <= 1.0:
            raise ValueError("rope_theta must be > 1")
        if self.rope_enabled and (self.hidden_dim // self.num_heads) % 2 != 0:
            raise ValueError("RoPE requires an even main attention head dimension")
        if self.waveform_level_depth > 0:
            if self.waveform_micro_patch_size <= 0:
                raise ValueError("waveform_micro_patch_size must be > 0")
            if self.patch_size % self.waveform_micro_patch_size != 0:
                raise ValueError(
                    "patch_size must be divisible by waveform_micro_patch_size "
                    f"({self.patch_size} % {self.waveform_micro_patch_size} != 0)"
                )
            if self.waveform_hidden_dim <= 0:
                raise ValueError("waveform_hidden_dim must be > 0")
            if self.waveform_num_heads <= 0:
                raise ValueError("waveform_num_heads must be > 0")
            if self.hidden_dim % self.waveform_num_heads != 0:
                raise ValueError(
                    "hidden_dim must be divisible by waveform_num_heads "
                    f"({self.hidden_dim} % {self.waveform_num_heads} != 0)"
                )
            if (
                self.rope_enabled
                and (self.hidden_dim // self.waveform_num_heads) % 2 != 0
            ):
                raise ValueError(
                    "RoPE requires an even waveform attention head dimension"
                )
            if self.waveform_mlp_ratio <= 0.0:
                raise ValueError("waveform_mlp_ratio must be > 0")

        target_token_dim = self.target_channels * self.patch_size
        cond_token_dim = self.cond_channels * self.patch_size

        # Tokenize per frame: flatten (C * D)
        self.target_in = nn.Linear(target_token_dim, self.hidden_dim)
        self.cond_in = nn.Linear(cond_token_dim, self.hidden_dim)

        # Time embedding -> context vector
        self.time_mlp = nn.Sequential(
            nn.Linear(self.timestep_embed_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        self.mix_style_mlp: nn.Sequential | None
        if self.mix_style_dim > 0:
            self.mix_style_mlp = nn.Sequential(
                nn.Linear(self.mix_style_dim, self.hidden_dim * 4),
                nn.SiLU(),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            )
            mix_style_output = self.mix_style_mlp[-1]
            assert isinstance(mix_style_output, nn.Linear)
            nn.init.zeros_(mix_style_output.weight)
            nn.init.zeros_(mix_style_output.bias)
        else:
            self.mix_style_mlp = None

        self.amplitude_gain_mlp: nn.Sequential | None
        if self.amplitude_gain_conditioning:
            self.amplitude_gain_mlp = nn.Sequential(
                nn.Linear(1, self.hidden_dim * 4),
                nn.SiLU(),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            )
            amplitude_gain_output = self.amplitude_gain_mlp[-1]
            assert isinstance(amplitude_gain_output, nn.Linear)
            nn.init.zeros_(amplitude_gain_output.weight)
            nn.init.zeros_(amplitude_gain_output.bias)
        else:
            self.amplitude_gain_mlp = None

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_output = FinalOutputBlock(
            hidden_dim=self.hidden_dim,
            out_dim=target_token_dim,
            kernel_size=self.final_output_kernel_size,
            zero_init=self.final_output_zero_init,
        )

        self.num_waveform_micro_tokens = (
            self.patch_size // self.waveform_micro_patch_size
            if self.waveform_level_depth > 0
            else 0
        )
        waveform_micro_dim = self.target_channels * self.waveform_micro_patch_size
        waveform_cond_micro_dim = self.cond_channels * self.waveform_micro_patch_size
        self.waveform_in: nn.Linear | None
        self.waveform_cond_in: nn.Linear | None
        self.waveform_norm: RMSNorm | None
        self.waveform_out: nn.Linear | None
        if self.waveform_level_depth > 0:
            self.waveform_in = nn.Linear(
                waveform_micro_dim,
                self.waveform_hidden_dim,
            )
            self.waveform_cond_in = nn.Linear(
                waveform_cond_micro_dim,
                self.waveform_hidden_dim,
            )
            self.waveform_blocks = nn.ModuleList(
                [
                    WaveformTransformerBlock(
                        semantic_dim=self.hidden_dim,
                        waveform_hidden_dim=self.waveform_hidden_dim,
                        num_micro_tokens=self.num_waveform_micro_tokens,
                        num_heads=self.waveform_num_heads,
                        mlp_ratio=self.waveform_mlp_ratio,
                        dropout=dropout,
                    )
                    for _ in range(self.waveform_level_depth)
                ]
            )
            self.waveform_norm = RMSNorm(self.waveform_hidden_dim)
            self.waveform_out = nn.Linear(
                self.waveform_hidden_dim,
                waveform_micro_dim,
            )
            nn.init.normal_(self.waveform_out.weight, std=1.0e-3)
            nn.init.zeros_(self.waveform_out.bias)
        else:
            self.waveform_in = None
            self.waveform_cond_in = None
            self.waveform_blocks = nn.ModuleList()
            self.waveform_norm = None
            self.waveform_out = None

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

    def _should_checkpoint(self) -> bool:
        """Return whether block activations should be recomputed in backward."""
        return bool(
            self.activation_checkpointing and self.training and torch.is_grad_enabled()
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

    def _get_rotary_embeddings(
        self,
        *,
        num_frames: int,
        memory_tokens: int,
        head_dim: int,
        device: torch.device,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor] | None,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        """Return RoPE tensors for x-stream tokens and frame tokens."""
        if not self.rope_enabled:
            return None, None
        frame_positions = torch.arange(num_frames, device=device, dtype=torch.float32)
        frame_rope = rotary_embedding_1d(
            positions=frame_positions,
            dim=head_dim,
            max_period=self.rope_theta,
        )
        if memory_tokens <= 0:
            return frame_rope, frame_rope
        memory_positions = torch.zeros(
            memory_tokens, device=device, dtype=torch.float32
        )
        x_positions = torch.cat([memory_positions, frame_positions], dim=0)
        x_rope = rotary_embedding_1d(
            positions=x_positions,
            dim=head_dim,
            max_period=self.rope_theta,
        )
        return x_rope, frame_rope

    def _waveform_refinement(
        self,
        *,
        zt: torch.Tensor,
        z_cond: torch.Tensor,
        semantic_tokens: torch.Tensor,
        time_context: torch.Tensor,
        memory_tokens: torch.Tensor | None,
        pos_embed: torch.Tensor,
        coarse_tokens: torch.Tensor,
        frame_pad_mask: torch.Tensor | None,
        frame_keep_mask: torch.Tensor | None,
        rope_self: tuple[torch.Tensor, torch.Tensor] | None,
        rope_frames: tuple[torch.Tensor, torch.Tensor] | None,
        conditioning_cache: dict[str, object] | None = None,
    ) -> torch.Tensor:
        """Return clean patch tokens after waveform-token residual refinement."""
        if (
            self.waveform_in is None
            or self.waveform_cond_in is None
            or self.waveform_norm is None
            or self.waveform_out is None
            or self.waveform_level_depth <= 0
        ):
            return coarse_tokens

        waveform_rope_self = rope_self
        waveform_rope_frames = rope_frames
        waveform_head_dim = self.hidden_dim // self.waveform_num_heads
        if conditioning_cache is not None:
            waveform_rope_self = cast(
                tuple[torch.Tensor, torch.Tensor] | None,
                conditioning_cache.get("waveform_rope_self", waveform_rope_self),
            )
            waveform_rope_frames = cast(
                tuple[torch.Tensor, torch.Tensor] | None,
                conditioning_cache.get("waveform_rope_frames", waveform_rope_frames),
            )
        elif self.rope_enabled and (
            waveform_head_dim != self.hidden_dim // self.num_heads
        ):
            waveform_rope_self, waveform_rope_frames = self._get_rotary_embeddings(
                num_frames=coarse_tokens.shape[1],
                memory_tokens=0
                if memory_tokens is None
                else int(memory_tokens.shape[1]),
                head_dim=waveform_head_dim,
                device=coarse_tokens.device,
            )

        batch_size, _, patch_size, num_frames = zt.shape
        if z_cond.shape[0] != batch_size or z_cond.shape[2:] != (
            patch_size,
            num_frames,
        ):
            raise ValueError(
                "z_cond must align with zt waveform patches for waveform refinement: "
                f"zt={tuple(zt.shape)} z_cond={tuple(z_cond.shape)}"
            )
        micro_size = self.waveform_micro_patch_size
        num_micro = self.num_waveform_micro_tokens
        micro = (
            coarse_tokens.reshape(
                batch_size,
                num_frames,
                self.target_channels,
                patch_size,
            )
            .permute(0, 1, 3, 2)
            .reshape(
                batch_size,
                num_frames,
                num_micro,
                micro_size * self.target_channels,
            )
            .contiguous()
        )
        waveform_tokens = cast(torch.Tensor, self.waveform_in(micro))
        if conditioning_cache is None:
            cond_micro = (
                z_cond.permute(0, 3, 2, 1)
                .reshape(
                    batch_size,
                    num_frames,
                    num_micro,
                    micro_size * self.cond_channels,
                )
                .contiguous()
            )
            cond_waveform_tokens = cast(torch.Tensor, self.waveform_cond_in(cond_micro))
            waveform_block_caches = None
        else:
            cond_waveform_tokens = cast(
                torch.Tensor,
                conditioning_cache["waveform_cond_tokens"],
            )
            waveform_block_caches = cast(
                list[dict[str, torch.Tensor]],
                conditioning_cache["waveform_block_caches"],
            )
        waveform_semantic = semantic_tokens + time_context[:, None, :]

        for block_index, block in enumerate(self.waveform_blocks):
            waveform_block = cast(WaveformTransformerBlock, block)
            waveform_block_cache = (
                None
                if waveform_block_caches is None
                else waveform_block_caches[block_index]
            )
            if self._should_checkpoint():
                waveform_tokens = checkpoint(
                    lambda tokens, cond_tokens: waveform_block(
                        waveform_tokens=tokens,
                        semantic_tokens=waveform_semantic,
                        cond_waveform_tokens=cond_tokens,
                        memory_tokens=memory_tokens,
                        pos_embed=pos_embed,
                        pad_mask=frame_pad_mask,
                        keep_mask=frame_keep_mask,
                        rope_self=waveform_rope_self,
                        rope_frames=waveform_rope_frames,
                        conditioning_cache=waveform_block_cache,
                    ),
                    waveform_tokens,
                    cond_waveform_tokens,
                    use_reentrant=False,
                )
            else:
                waveform_tokens = waveform_block(
                    waveform_tokens=waveform_tokens,
                    semantic_tokens=waveform_semantic,
                    cond_waveform_tokens=cond_waveform_tokens,
                    memory_tokens=memory_tokens,
                    pos_embed=pos_embed,
                    pad_mask=frame_pad_mask,
                    keep_mask=frame_keep_mask,
                    rope_self=waveform_rope_self,
                    rope_frames=waveform_rope_frames,
                    conditioning_cache=waveform_block_cache,
                )

        residual_micro = cast(
            torch.Tensor,
            self.waveform_out(self.waveform_norm(waveform_tokens)),
        )
        residual_tokens = (
            residual_micro.reshape(
                batch_size,
                num_frames,
                patch_size,
                self.target_channels,
            )
            .permute(0, 1, 3, 2)
            .reshape(batch_size, num_frames, self.target_channels * patch_size)
            .contiguous()
        )
        return coarse_tokens + residual_tokens

    @torch.inference_mode()
    def build_conditioning_cache(
        self,
        *,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """Precompute inference-only conditioning tensors for a fixed window."""
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("conditioning caches are inference-only")
        if z_cond.dim() != 4:
            raise ValueError(f"z_cond must be [B,C,D,T], got {tuple(z_cond.shape)}")
        batch_size, cond_channels, patch_size, num_frames = z_cond.shape
        if cond_channels != self.cond_channels:
            raise ValueError(
                f"Expected cond channels={self.cond_channels}, got {cond_channels}"
            )
        if patch_size != self.patch_size:
            raise ValueError(f"Expected patch_size={self.patch_size}, got {patch_size}")

        frame_pad_mask: torch.Tensor | None = None
        frame_keep_mask: torch.Tensor | None = None
        if valid_mask is not None:
            if valid_mask.shape != (batch_size, num_frames):
                raise ValueError(
                    f"valid_mask must be [B,T]=({batch_size},{num_frames}), got {tuple(valid_mask.shape)}"
                )
            frame_keep_mask = valid_mask.bool().clone()
            frame_pad_mask = ~frame_keep_mask

        cond_tokens = z_cond.permute(0, 3, 1, 2).reshape(batch_size, num_frames, -1)
        cond_tokens = self.cond_in(cond_tokens)

        pos = self._get_positional_embedding(
            length=num_frames,
            device=z_cond.device,
            dtype=cond_tokens.dtype,
        )
        if not self.rope_enabled:
            cond_tokens = cond_tokens + pos[None, :, :]

        main_head_dim = self.hidden_dim // self.num_heads
        rope_x, rope_frames = self._get_rotary_embeddings(
            num_frames=num_frames,
            memory_tokens=self.num_memory_tokens,
            head_dim=main_head_dim,
            device=z_cond.device,
        )
        transformer_caches = [
            cast(TransformerBlock, block).build_conditioning_cache(
                cond_tokens=cond_tokens,
                mem_len=self.num_memory_tokens,
                rope_cond=rope_frames,
            )
            for block in self.blocks
        ]

        cache: dict[str, object] = {
            "cond_tokens": cond_tokens,
            "pos": pos,
            "frame_pad_mask": frame_pad_mask,
            "frame_keep_mask": frame_keep_mask,
            "rope_x": rope_x,
            "rope_frames": rope_frames,
            "transformer_caches": transformer_caches,
        }

        if (
            self.waveform_level_depth > 0
            and self.waveform_cond_in is not None
            and self.waveform_blocks
        ):
            micro_size = self.waveform_micro_patch_size
            num_micro = self.num_waveform_micro_tokens
            cond_micro = (
                z_cond.permute(0, 3, 2, 1)
                .reshape(
                    batch_size,
                    num_frames,
                    num_micro,
                    micro_size * self.cond_channels,
                )
                .contiguous()
            )
            cond_waveform_tokens = cast(
                torch.Tensor,
                self.waveform_cond_in(cond_micro),
            )
            waveform_rope_self = rope_x
            waveform_rope_frames = rope_frames
            waveform_head_dim = self.hidden_dim // self.waveform_num_heads
            if self.rope_enabled and (
                waveform_head_dim != self.hidden_dim // self.num_heads
            ):
                waveform_rope_self, waveform_rope_frames = self._get_rotary_embeddings(
                    num_frames=num_frames,
                    memory_tokens=self.num_memory_tokens,
                    head_dim=waveform_head_dim,
                    device=z_cond.device,
                )
            cache["waveform_rope_self"] = waveform_rope_self
            cache["waveform_rope_frames"] = waveform_rope_frames
            cache["waveform_cond_tokens"] = cond_waveform_tokens
            cache["waveform_block_caches"] = [
                cast(WaveformTransformerBlock, block).build_conditioning_cache(
                    cond_waveform_tokens=cond_waveform_tokens,
                    pos_embed=pos,
                    rope_frames=waveform_rope_frames,
                )
                for block in self.waveform_blocks
            ]

        return cache

    @overload
    def forward(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        mix_style: torch.Tensor | None = None,
        mix_style_mask: torch.Tensor | None = None,
        amplitude_gain: torch.Tensor | None = None,
        mem: torch.Tensor | None = None,
        return_mem: Literal[False] = False,
        conditioning_cache: dict[str, object] | None = None,
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        mix_style: torch.Tensor | None = None,
        mix_style_mask: torch.Tensor | None = None,
        amplitude_gain: torch.Tensor | None = None,
        mem: torch.Tensor | None = None,
        return_mem: Literal[True] = True,
        conditioning_cache: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

    def forward(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        mix_style: torch.Tensor | None = None,
        mix_style_mask: torch.Tensor | None = None,
        amplitude_gain: torch.Tensor | None = None,
        mem: torch.Tensor | None = None,
        return_mem: bool = False,
        conditioning_cache: dict[str, object] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Predict clean target signals and optionally updated memory tokens."""
        if zt.dim() != 4:
            raise ValueError(f"zt must be [B,C,D,T], got {tuple(zt.shape)}")
        if z_cond.dim() != 4:
            raise ValueError(f"z_cond must be [B,C,D,T], got {tuple(z_cond.shape)}")

        batch_size, target_channels, patch_size, num_frames = zt.shape

        if target_channels != self.target_channels:
            raise ValueError(
                f"Expected target channels={self.target_channels}, got {target_channels}"
            )
        if patch_size != self.patch_size:
            raise ValueError(f"Expected patch_size={self.patch_size}, got {patch_size}")

        if z_cond.shape[0] != batch_size or z_cond.shape[3] != num_frames:
            raise ValueError(
                "z_cond batch/time mismatch: "
                f"zt={tuple(zt.shape)} z_cond={tuple(z_cond.shape)}"
            )
        if z_cond.shape[1] != self.cond_channels:
            raise ValueError(
                f"Expected cond channels={self.cond_channels}, got {z_cond.shape[1]}"
            )
        if z_cond.shape[2] != self.patch_size:
            raise ValueError(
                f"Expected cond patch_size={self.patch_size}, got {z_cond.shape[2]}"
            )

        if conditioning_cache is None:
            frame_pad_mask: torch.Tensor | None = None
            frame_keep_mask: torch.Tensor | None = None
        else:
            frame_pad_mask = cast(
                torch.Tensor | None,
                conditioning_cache["frame_pad_mask"],
            )
            frame_keep_mask = cast(
                torch.Tensor | None,
                conditioning_cache["frame_keep_mask"],
            )
        if valid_mask is not None and conditioning_cache is None:
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

        # project to hidden
        x_tokens = self.target_in(x_tokens)
        if conditioning_cache is None:
            cond_tokens = z_cond.permute(0, 3, 1, 2).reshape(batch_size, num_frames, -1)
            cond_tokens = self.cond_in(cond_tokens)
            transformer_caches = None
        else:
            cond_tokens = cast(torch.Tensor, conditioning_cache["cond_tokens"])
            transformer_caches = cast(
                list[dict[str, torch.Tensor]],
                conditioning_cache["transformer_caches"],
            )

        # positional embedding over frames (ONLY for frame tokens)
        pos = (
            self._get_positional_embedding(
                length=num_frames,
                device=zt.device,
                dtype=x_tokens.dtype,
            )
            if conditioning_cache is None
            else cast(torch.Tensor, conditioning_cache["pos"])
        )
        if not self.rope_enabled:
            x_tokens = x_tokens + pos[None, :, :]
            if conditioning_cache is None:
                cond_tokens = cond_tokens + pos[None, :, :]

        # time context
        t_batch = self._as_batch_timesteps(t, batch_size)
        t_embed = timestep_embedding(
            timesteps=t_batch * self.timestep_scale,
            dim=self.timestep_embed_dim,
            max_period=self.max_period,
        ).to(dtype=x_tokens.dtype)
        time_context = self.time_mlp(t_embed)  # [B, H]
        if self.mix_style_mlp is not None and mix_style is not None:
            if mix_style.shape != (batch_size, self.mix_style_dim):
                raise ValueError(
                    "mix_style must be [B,K]="
                    f"({batch_size},{self.mix_style_dim}), got {tuple(mix_style.shape)}"
                )
            mix_style_centered = (
                mix_style.to(device=zt.device, dtype=x_tokens.dtype) * 2.0 - 1.0
            )
            mix_style_context = self.mix_style_mlp(mix_style_centered)
            if mix_style_mask is not None:
                if mix_style_mask.shape != (batch_size,):
                    raise ValueError(
                        f"mix_style_mask must be [B]=({batch_size},), got {tuple(mix_style_mask.shape)}"
                    )
                mix_style_context = (
                    mix_style_context
                    * mix_style_mask.to(
                        device=zt.device,
                        dtype=x_tokens.dtype,
                    )[:, None]
                )
            time_context = time_context + mix_style_context

        if self.amplitude_gain_mlp is not None and amplitude_gain is not None:
            gain_input = amplitude_gain.to(
                device=zt.device, dtype=x_tokens.dtype
            ).reshape(batch_size, 1)
            time_context = time_context + self.amplitude_gain_mlp(gain_input)

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

        main_head_dim = self.hidden_dim // self.num_heads
        if conditioning_cache is None:
            rope_x, rope_frames = self._get_rotary_embeddings(
                num_frames=num_frames,
                memory_tokens=M,
                head_dim=main_head_dim,
                device=zt.device,
            )
        else:
            rope_x = cast(
                tuple[torch.Tensor, torch.Tensor] | None,
                conditioning_cache["rope_x"],
            )
            rope_frames = cast(
                tuple[torch.Tensor, torch.Tensor] | None,
                conditioning_cache["rope_frames"],
            )

        # blocks
        for block_index, block in enumerate(self.blocks):
            transformer_block = cast(TransformerBlock, block)
            transformer_cache = (
                None if transformer_caches is None else transformer_caches[block_index]
            )
            if self._should_checkpoint():
                x_all = checkpoint(
                    lambda tokens: transformer_block(
                        x_tokens=tokens,
                        cond_tokens=cond_tokens,
                        time_context=time_context,
                        pad_mask_x=pad_mask_x,
                        pad_mask_cond=frame_pad_mask,
                        keep_mask_x=keep_mask_x,
                        rope_x=rope_x,
                        rope_cond=rope_frames,
                        conditioning_cache=transformer_cache,
                    ),
                    x_all,
                    use_reentrant=False,
                )
            else:
                x_all = transformer_block(
                    x_tokens=x_all,
                    cond_tokens=cond_tokens,
                    time_context=time_context,
                    pad_mask_x=pad_mask_x,
                    pad_mask_cond=frame_pad_mask,
                    keep_mask_x=keep_mask_x,
                    rope_x=rope_x,
                    rope_cond=rope_frames,
                    conditioning_cache=transformer_cache,
                )

        # split memory + frames
        if M > 0:
            mem_out = x_all[:, :M, :].contiguous()
            x_tokens = x_all[:, M:, :].contiguous()
        else:
            x_tokens = x_all

        # project back to [B, C, D, T]
        final_context = cond_tokens + time_context[:, None, :]
        clean_tokens = self.final_output(x_tokens, final_context)
        clean_tokens = self._waveform_refinement(
            zt=zt,
            z_cond=z_cond,
            semantic_tokens=x_tokens,
            time_context=time_context,
            memory_tokens=mem_out,
            pos_embed=pos,
            coarse_tokens=clean_tokens,
            frame_pad_mask=frame_pad_mask,
            frame_keep_mask=frame_keep_mask,
            rope_self=rope_x,
            rope_frames=rope_frames,
            conditioning_cache=conditioning_cache,
        )

        # Re-mask padded frames
        if frame_keep_mask is not None:
            clean_tokens = (
                clean_tokens * frame_keep_mask.to(clean_tokens.dtype)[:, :, None]
            )

        clean_prediction = (
            clean_tokens.reshape(
                batch_size, num_frames, self.target_channels, self.patch_size
            )
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        if return_mem:
            return clean_prediction, mem_out
        return clean_prediction
