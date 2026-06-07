"""Training package public exports."""

from stereo2spatial.modeling import SpatialDiT

from .config import TrainConfig, load_config
from .dataset import ConditioningSource, WaveformSongDataset

__all__ = [
    "ConditioningSource",
    "TrainConfig",
    "SpatialDiT",
    "WaveformSongDataset",
    "load_config",
]
