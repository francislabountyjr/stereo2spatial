"""Model architectures shared by training and inference."""

from .factory import (
    LEGACY_VAE_ARCHITECTURE,
    WAVEFORM_ARCHITECTURE,
    SpatialModel,
    build_spatial_model,
    is_legacy_vae_model,
    resolve_model_architecture,
)
from .legacy_spatial_dit import LegacySpatialDiT
from .spatial_dit import SpatialDiT

__all__ = [
    "LEGACY_VAE_ARCHITECTURE",
    "WAVEFORM_ARCHITECTURE",
    "LegacySpatialDiT",
    "SpatialDiT",
    "SpatialModel",
    "build_spatial_model",
    "is_legacy_vae_model",
    "resolve_model_architecture",
]
