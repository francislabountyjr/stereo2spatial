"""Training package public exports."""

from stereo2spatial.modeling import LegacySpatialDiT, SpatialDiT

from .config import TrainConfig, load_config
from .dataset import ConditioningSource, WaveformSongDataset
from .latent_dataset import LatentSongDataset

__all__ = [
    "ConditioningSource",
    "TrainConfig",
    "LegacySpatialDiT",
    "SpatialDiT",
    "WaveformSongDataset",
    "LatentSongDataset",
    "load_config",
]
