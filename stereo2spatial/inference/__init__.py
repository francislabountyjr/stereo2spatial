"""Inference package public exports."""

from .audio import read_audio_channels_first
from .checkpoint import resolve_checkpoint_path
from .dynamic_batcher import (
    SchedulerStats,
    group_compatible_queries,
    run_dynamic_controller_scheduler,
)
from .export_bundle import export_model_bundle, resolve_inference_config_path
from .offline_batch import (
    DynamicFolderInferenceResult,
    DynamicInferenceJob,
    OfflineBatchResult,
    OfflineBatchStats,
    WindowedInferenceRequest,
    WindowedInferenceResult,
    run_dynamic_folder_inference,
    run_windowed_inference_requests,
)
from .runner import run_inference
from .sampling import generate_spatial_signal, resolve_chunk_frames
from .solvers import (
    FixedStepSolverController,
    ModelQuery,
    SolverController,
    stack_model_queries,
)
from .timestep_sampling import generate_spatial_signal_timestep_major
from .windowing import (
    FixedWindowSpec,
    extract_fixed_window,
    fixed_window_specs,
    resolve_fixed_window_frames,
    stitch_fixed_windows,
)

__all__ = [
    "FixedWindowSpec",
    "FixedStepSolverController",
    "ModelQuery",
    "DynamicFolderInferenceResult",
    "DynamicInferenceJob",
    "OfflineBatchResult",
    "OfflineBatchStats",
    "SchedulerStats",
    "SolverController",
    "WindowedInferenceRequest",
    "WindowedInferenceResult",
    "extract_fixed_window",
    "export_model_bundle",
    "fixed_window_specs",
    "generate_spatial_signal",
    "generate_spatial_signal_timestep_major",
    "group_compatible_queries",
    "read_audio_channels_first",
    "resolve_fixed_window_frames",
    "resolve_inference_config_path",
    "resolve_checkpoint_path",
    "resolve_chunk_frames",
    "run_inference",
    "run_dynamic_controller_scheduler",
    "run_dynamic_folder_inference",
    "run_windowed_inference_requests",
    "stack_model_queries",
    "stitch_fixed_windows",
]
