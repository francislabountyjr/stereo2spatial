"""Training package public exports."""

from stereo2spatial.modeling import SpatialDiT

from .config import TrainConfig, load_config
from .dataset import ConditioningSource, LatentSongDataset

__all__ = [
    "ConditioningSource",
    "TrainConfig",
    "SpatialDiT",
    "LatentSongDataset",
    "load_config",
]
