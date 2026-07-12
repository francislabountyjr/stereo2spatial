"""Architecture selection and construction shared by train, infer, and export."""

from __future__ import annotations

from typing import Any, TypeAlias

from .legacy_spatial_dit import LegacySpatialDiT
from .spatial_dit import SpatialDiT

WAVEFORM_ARCHITECTURE = "waveform"
LEGACY_VAE_ARCHITECTURE = "legacy_vae"
SUPPORTED_ARCHITECTURES = (WAVEFORM_ARCHITECTURE, LEGACY_VAE_ARCHITECTURE)
SpatialModel: TypeAlias = SpatialDiT | LegacySpatialDiT

_ARCHITECTURE_ALIASES = {
    "waveform": WAVEFORM_ARCHITECTURE,
    "waveform_dit": WAVEFORM_ARCHITECTURE,
    "waveform_patch": WAVEFORM_ARCHITECTURE,
    "no_vae": WAVEFORM_ARCHITECTURE,
    "legacy_vae": LEGACY_VAE_ARCHITECTURE,
    "ear_vae_latent_v1": LEGACY_VAE_ARCHITECTURE,
    "vae_latent": LEGACY_VAE_ARCHITECTURE,
    "latent": LEGACY_VAE_ARCHITECTURE,
    "vae": LEGACY_VAE_ARCHITECTURE,
}


def normalize_model_architecture(value: object) -> str:
    """Normalize an architecture name or raise a stable config error."""
    normalized = str(value).strip().lower()
    architecture = _ARCHITECTURE_ALIASES.get(normalized)
    if architecture is None:
        raise ValueError(
            "model.architecture must be one of: " + ", ".join(SUPPORTED_ARCHITECTURES)
        )
    return architecture


def resolve_model_architecture(model_config: object) -> str:
    """Return the normalized architecture stored on a model config object."""
    return normalize_model_architecture(
        getattr(model_config, "architecture", WAVEFORM_ARCHITECTURE)
    )


def is_legacy_vae_model(model_or_config: object) -> bool:
    """Return whether a model instance/config selects the legacy VAE path."""
    direct = getattr(model_or_config, "architecture", None)
    if direct is not None:
        try:
            return normalize_model_architecture(direct) == LEGACY_VAE_ARCHITECTURE
        except ValueError:
            pass
    nested = getattr(model_or_config, "model", None)
    if nested is not None:
        return resolve_model_architecture(nested) == LEGACY_VAE_ARCHITECTURE
    return False


def build_spatial_model(model_config: Any) -> SpatialModel:
    """Build the configured waveform or parameter-compatible legacy model."""
    architecture = resolve_model_architecture(model_config)
    common: dict[str, Any] = {
        "target_channels": int(model_config.target_channels),
        "cond_channels": int(model_config.cond_channels),
        "hidden_dim": int(model_config.hidden_dim),
        "num_layers": int(model_config.num_layers),
        "num_heads": int(model_config.num_heads),
        "mlp_ratio": float(model_config.mlp_ratio),
        "dropout": float(model_config.dropout),
        "timestep_embed_dim": int(model_config.timestep_embed_dim),
        "timestep_scale": float(model_config.timestep_scale),
        "max_period": float(model_config.max_period),
        "num_memory_tokens": int(getattr(model_config, "num_memory_tokens", 0)),
    }
    if architecture == LEGACY_VAE_ARCHITECTURE:
        latent_dim = getattr(model_config, "latent_dim", None)
        if latent_dim is None:
            latent_dim = getattr(model_config, "patch_size", None)
        if latent_dim is None:
            raise ValueError("legacy_vae architecture requires model.latent_dim")
        return LegacySpatialDiT(
            **common,
            latent_dim=int(latent_dim),
            activation_checkpointing=bool(
                getattr(model_config, "activation_checkpointing", False)
            ),
        )

    return SpatialDiT(
        **common,
        patch_size=int(model_config.patch_size),
        mix_style_dim=int(getattr(model_config, "mix_style_dim", 0)),
        amplitude_gain_conditioning=bool(
            getattr(model_config, "amplitude_gain_conditioning", False)
        ),
        waveform_level_depth=int(getattr(model_config, "waveform_level_depth", 0)),
        waveform_micro_patch_size=int(
            getattr(model_config, "waveform_micro_patch_size", 16)
        ),
        waveform_hidden_dim=int(getattr(model_config, "waveform_hidden_dim", 16)),
        waveform_num_heads=getattr(model_config, "waveform_num_heads", None),
        waveform_mlp_ratio=float(getattr(model_config, "waveform_mlp_ratio", 2.0)),
        final_output_kernel_size=int(
            getattr(model_config, "final_output_kernel_size", 7)
        ),
        final_output_zero_init=bool(
            getattr(model_config, "final_output_zero_init", False)
        ),
        rope_enabled=bool(getattr(model_config, "rope_enabled", True)),
        rope_theta=float(getattr(model_config, "rope_theta", 10000.0)),
        activation_checkpointing=bool(
            getattr(model_config, "activation_checkpointing", False)
        ),
    )


__all__ = [
    "LEGACY_VAE_ARCHITECTURE",
    "SUPPORTED_ARCHITECTURES",
    "SpatialModel",
    "WAVEFORM_ARCHITECTURE",
    "build_spatial_model",
    "is_legacy_vae_model",
    "normalize_model_architecture",
    "resolve_model_architecture",
]
