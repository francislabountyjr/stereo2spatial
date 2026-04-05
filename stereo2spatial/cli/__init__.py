"""Command-line entrypoints for stereo2spatial."""

from .infer import build_parser as build_infer_parser
from .infer import main as infer_main
from .train import build_parser as build_train_parser
from .train import main as train_main

__all__ = [
    "build_infer_parser",
    "build_train_parser",
    "infer_main",
    "train_main",
]
