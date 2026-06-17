"""Offline dynamic batching over fixed-window inference requests."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from stereo2spatial.common.amplitude_lift import (
    amplitude_lift_log_gain,
    apply_amplitude_lift,
    resolve_amplitude_lift_gain,
    undo_amplitude_lift,
    undo_wavflow_output_lift,
    wavflow_source_transform,
)
from stereo2spatial.inference.audio import (
    read_audio_channels_first,
    write_audio_channels_first,
)
from stereo2spatial.inference.cuda_graphs import CudaGraphModelRunner
from stereo2spatial.inference.dynamic_batcher import (
    SchedulerStats,
    group_compatible_queries,
)
from stereo2spatial.inference.runner import (
    InferenceReport,
    InferenceSession,
    RequestedSolverName,
    _patch_audio,
    _prepare_conditioning_audio,
    _resolve_inference_mix_style,
    _resolve_inference_solver,
    _unpatch_audio,
)
from stereo2spatial.inference.solvers import (
    FixedStepSolverController,
    ModelQuery,
    SolverController,
    stack_model_queries,
)
from stereo2spatial.inference.windowing import (
    FixedWindowSpec,
    extract_fixed_window,
    fixed_window_specs,
    resolve_fixed_window_frames,
    stitch_fixed_windows,
)

_DYNAMIC_SOLVERS = {"euler", "heun", "midpoint_rk2", "res6s"}


@dataclass
class WindowedInferenceResult:
    """Completed dynamic-batching result for one request."""

    request_id: str
    pred_signal: torch.Tensor
    window_count: int
    final_memory: torch.Tensor | None


@dataclass
class WindowedInferenceRequest:
    """State for one song moving through sequential fixed windows."""

    request_id: str
    cond_signal: torch.Tensor
    target_channels: int
    window_frames: int
    overlap_frames: int
    solver: str
    solver_steps: int
    seed: int
    mem: torch.Tensor | None = None
    mix_style: torch.Tensor | None = None
    amplitude_gain: torch.Tensor | None = None
    _window_index: int = 0
    _active_controller: FixedStepSolverController | None = None
    _pred_windows: list[torch.Tensor] = field(default_factory=list)
    _specs: list[FixedWindowSpec] = field(init=False)
    _generator: torch.Generator = field(init=False)

    def __post_init__(self) -> None:
        if self.cond_signal.dim() != 3:
            raise ValueError(
                f"cond_signal must be [C,P,T], got {tuple(self.cond_signal.shape)}"
            )
        if self.target_channels <= 0:
            raise ValueError("target_channels must be > 0")
        if self.window_frames <= 0:
            raise ValueError("window_frames must be > 0")
        if self.overlap_frames < 0 or self.overlap_frames >= self.window_frames:
            raise ValueError("overlap_frames must be in [0, window_frames)")
        self._specs = fixed_window_specs(
            total_frames=int(self.cond_signal.shape[-1]),
            window_frames=int(self.window_frames),
            overlap_frames=int(self.overlap_frames),
        )
        self._generator = torch.Generator(device=self.cond_signal.device)
        self._generator.manual_seed(int(self.seed))

    @property
    def is_done(self) -> bool:
        """Whether all windows have been solved and stitched."""
        return self._window_index >= len(self._specs) and self._active_controller is None

    @property
    def active_controller(self) -> FixedStepSolverController | None:
        """Current active window controller, if any."""
        return self._active_controller

    def ensure_active_controller(
        self,
        model: torch.nn.Module | None = None,
    ) -> FixedStepSolverController | None:
        """Create the next window controller if the request can advance."""
        if self._active_controller is not None or self.is_done:
            return self._active_controller

        spec = self._specs[self._window_index]
        cond_window = extract_fixed_window(self.cond_signal, spec).unsqueeze(0)
        valid_mask = None
        if spec.valid_frames < int(self.window_frames):
            valid_mask = torch.zeros(
                (1, int(self.window_frames)),
                dtype=torch.bool,
                device=self.cond_signal.device,
            )
            valid_mask[:, : spec.valid_frames] = True
        z0_window = torch.randn(
            (
                1,
                int(self.target_channels),
                int(self.cond_signal.shape[1]),
                int(self.window_frames),
            ),
            generator=self._generator,
            dtype=self.cond_signal.dtype,
            device=self.cond_signal.device,
        )
        conditioning_cache = None
        if model is not None:
            cache_builder = getattr(model, "build_conditioning_cache", None)
            if cache_builder is None and hasattr(model, "_orig_mod"):
                cache_builder = getattr(model._orig_mod, "build_conditioning_cache", None)
            if cache_builder is not None:
                conditioning_cache = cache_builder(
                    z_cond=cond_window,
                    valid_mask=valid_mask,
                )
        self._active_controller = FixedStepSolverController(
            request_id=self.request_id,
            window_index=spec.index,
            solver=self.solver,
            solver_steps=self.solver_steps,
            z0_chunk=z0_window,
            z_cond=cond_window,
            valid_mask=valid_mask,
            mem=self.mem,
            mix_style=self.mix_style,
            amplitude_gain=self.amplitude_gain,
            conditioning_cache=conditioning_cache,
        )
        return self._active_controller

    def accept_completed_controller(self) -> None:
        """Commit the active completed window and unlock the next one."""
        if self._active_controller is None:
            raise RuntimeError("no active controller to accept")
        if not self._active_controller.is_done:
            raise RuntimeError("active controller is not done")
        self._pred_windows.append(self._active_controller.result()[0].detach().cpu())
        self.mem = (
            None
            if self._active_controller.mem_out is None
            else self._active_controller.mem_out.detach()
        )
        self._active_controller = None
        self._window_index += 1

    def result(self) -> WindowedInferenceResult:
        """Return the stitched request result."""
        if not self.is_done:
            raise RuntimeError("request is not done")
        pred_signal = stitch_fixed_windows(
            self._pred_windows,
            self._specs,
            overlap_frames=self.overlap_frames,
        )
        return WindowedInferenceResult(
            request_id=self.request_id,
            pred_signal=pred_signal,
            window_count=len(self._specs),
            final_memory=self.mem,
        )


@dataclass(frozen=True)
class DynamicInferenceJob:
    """One filesystem inference job for dynamic folder inference."""

    input_audio_path: Path
    output_audio_path: Path
    report_json_path: Path | None = None


@dataclass
class _PreparedFileRequest:
    request: WindowedInferenceRequest
    input_audio_path: Path
    output_audio_path: Path
    report_json_path: Path | None
    actual_sample_rate: int
    input_channels: int
    input_samples: int
    conditioning_signal_shape: list[int]
    input_sample_count: int
    patch_fps: float
    window_seconds: float
    window_frames: int
    overlap_seconds: float
    overlap_frames: int
    solver: str
    solver_steps: int
    solver_rtol: float
    solver_atol: float
    seed: int
    mix_style: list[float] | None
    mix_style_preset: str | None
    amplitude_lift_enabled: bool
    amplitude_lift_mode: str
    amplitude_lift_reference: str
    amplitude_lift_target_rms: float
    amplitude_lift_scale: float
    amplitude_lift_clip_value: float | None
    amplitude_lift_output_lufs: float
    amplitude_lift_gain: torch.Tensor | None
    lift_power: float
    lift_min: float | None


@dataclass(frozen=True)
class DynamicFolderInferenceResult:
    """Reports and per-file errors from dynamic folder inference."""

    reports: list[InferenceReport]
    errors: list[tuple[Path, Exception]]
    stats: OfflineBatchStats


@dataclass(frozen=True)
class OfflineBatchStats:
    """Summary of one offline dynamic-batching run."""

    scheduler: SchedulerStats
    completed_requests: int


@dataclass(frozen=True)
class OfflineBatchResult:
    """Results and scheduler statistics for a dynamic offline run."""

    results: list[WindowedInferenceResult]
    stats: OfflineBatchStats


def _init_model_memory(
    model: torch.nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Return model memory tokens, handling torch.compile wrappers."""
    init_memory = getattr(model, "init_memory", None)
    if init_memory is None and hasattr(model, "_orig_mod"):
        init_memory = getattr(model._orig_mod, "init_memory", None)
    if init_memory is None:
        return None
    return init_memory(batch_size=1, device=device, dtype=dtype)


def _prepare_dynamic_file_request(
    *,
    session: InferenceSession,
    job: DynamicInferenceJob,
    request_id: str,
    sample_rate: int,
    window_seconds: float,
    overlap_seconds: float,
    solver: str,
    solver_steps: int,
    solver_rtol: float,
    solver_atol: float,
    seed: int,
    mix_style: list[float] | dict[str, float] | None,
    mix_style_preset: str | None,
) -> _PreparedFileRequest:
    config = session.config
    run_dtype = session.run_dtype
    input_path = Path(job.input_audio_path)
    audio, actual_sample_rate = read_audio_channels_first(
        audio_path=input_path,
        target_sample_rate=sample_rate,
    )
    if audio.shape[0] not in {1, 2}:
        raise ValueError(
            f"Input must be mono or stereo. Got channels={audio.shape[0]} for {input_path}"
        )

    conditioning_audio = _prepare_conditioning_audio(
        audio.float(),
        cond_channels=int(config.model.cond_channels),
    )
    amplitude_lift_gain = None
    amplitude_lift_log_gain_tensor: torch.Tensor | None = None
    amplitude_lift_mode = (
        str(getattr(config.data, "amplitude_lift_mode", "rms")).strip().lower()
    )
    lift_scale = float(getattr(config.data, "amplitude_lift_scale", 3.0))
    lift_clip = getattr(config.data, "amplitude_lift_clip_value", 4.0)
    lift_power = float(getattr(config.data, "amplitude_lift_gain_power", 1.0))
    lift_min = getattr(config.data, "amplitude_lift_gain_min_value", None)
    lift_output_lufs = float(getattr(config.data, "amplitude_lift_output_lufs", -23.0))
    if bool(getattr(config.data, "amplitude_lift_enabled", False)):
        lift_reference = (
            str(getattr(config.data, "amplitude_lift_reference", "source"))
            .strip()
            .lower()
        )
        if amplitude_lift_mode != "wavflow" and lift_reference != "source":
            raise ValueError(
                "Dynamic inference amplitude lifting requires "
                "data.amplitude_lift_reference='source' because target audio is "
                "unavailable at inference time."
            )
        if amplitude_lift_mode == "wavflow":
            conditioning_audio, amplitude_lift_gain = wavflow_source_transform(
                conditioning_audio,
                target_rms=float(
                    getattr(config.data, "amplitude_lift_target_rms", 0.33)
                ),
                scale=lift_scale,
                peak_limit=float(
                    getattr(config.data, "amplitude_lift_peak_limit", 1.0)
                ),
                eps=float(getattr(config.data, "amplitude_lift_eps", 1.0e-8)),
            )
            amplitude_lift_log_gain_tensor = torch.log(
                amplitude_lift_gain.clamp_min(
                    float(getattr(config.data, "amplitude_lift_eps", 1.0e-8))
                )
            ).to(session.run_device, dtype=run_dtype)
        else:
            amplitude_lift_gain = resolve_amplitude_lift_gain(
                conditioning_audio,
                mode=amplitude_lift_mode,
                target_rms=float(
                    getattr(config.data, "amplitude_lift_target_rms", 0.33)
                ),
                eps=float(getattr(config.data, "amplitude_lift_eps", 1.0e-8)),
            )
            lift_clip = None if amplitude_lift_mode == "scale" else lift_clip
            conditioning_audio = apply_amplitude_lift(
                conditioning_audio,
                gain=amplitude_lift_gain,
                scale=lift_scale,
                clip_value=lift_clip,
                gain_power=lift_power,
                gain_min_value=lift_min,
            )
            amplitude_lift_log_gain_tensor = amplitude_lift_log_gain(
                amplitude_lift_gain,
                scale=lift_scale,
                gain_clip_value=lift_clip,
                gain_power=lift_power,
                gain_min_value=lift_min,
                eps=float(getattr(config.data, "amplitude_lift_eps", 1.0e-8)),
            ).to(session.run_device, dtype=run_dtype)

    cond_signal, input_sample_count = _patch_audio(
        conditioning_audio,
        patch_size=int(config.model.patch_size),
    )
    cond_signal = cond_signal.to(session.run_device, dtype=run_dtype)
    window_frames, overlap_frames = resolve_fixed_window_frames(
        sample_rate=actual_sample_rate,
        patch_size=int(config.model.patch_size),
        window_seconds=window_seconds,
        overlap_seconds=overlap_seconds,
    )
    mix_style_tensor = _resolve_inference_mix_style(
        raw_mix_style=mix_style,
        mix_style_dim=int(getattr(config.model, "mix_style_dim", 0)),
        target_channels=int(getattr(config.model, "target_channels", 0)),
        preset_name=mix_style_preset,
    )
    if mix_style_tensor is not None:
        mix_style_tensor = mix_style_tensor.to(session.run_device, dtype=run_dtype)
    mem = _init_model_memory(
        session.model,
        device=session.run_device,
        dtype=run_dtype,
    )
    request = WindowedInferenceRequest(
        request_id=request_id,
        cond_signal=cond_signal,
        target_channels=int(config.model.target_channels),
        window_frames=window_frames,
        overlap_frames=overlap_frames,
        solver=solver,
        solver_steps=solver_steps,
        seed=seed,
        mem=mem,
        mix_style=mix_style_tensor,
        amplitude_gain=amplitude_lift_log_gain_tensor,
    )
    return _PreparedFileRequest(
        request=request,
        input_audio_path=input_path,
        output_audio_path=Path(job.output_audio_path),
        report_json_path=job.report_json_path,
        actual_sample_rate=actual_sample_rate,
        input_channels=int(audio.shape[0]),
        input_samples=int(audio.shape[-1]),
        conditioning_signal_shape=[int(x) for x in cond_signal.shape],
        input_sample_count=input_sample_count,
        patch_fps=float(actual_sample_rate) / float(config.model.patch_size),
        window_seconds=window_seconds,
        window_frames=window_frames,
        overlap_seconds=overlap_seconds,
        overlap_frames=overlap_frames,
        solver=solver,
        solver_steps=solver_steps,
        solver_rtol=solver_rtol,
        solver_atol=solver_atol,
        seed=seed,
        mix_style=(
            [float(x) for x in mix_style_tensor.detach().cpu().flatten().tolist()]
            if mix_style_tensor is not None
            else None
        ),
        mix_style_preset=mix_style_preset,
        amplitude_lift_enabled=bool(
            getattr(config.data, "amplitude_lift_enabled", False)
        ),
        amplitude_lift_mode=str(getattr(config.data, "amplitude_lift_mode", "rms")),
        amplitude_lift_reference=str(
            getattr(config.data, "amplitude_lift_reference", "source")
        ),
        amplitude_lift_target_rms=float(
            getattr(config.data, "amplitude_lift_target_rms", 0.33)
        ),
        amplitude_lift_scale=lift_scale,
        amplitude_lift_clip_value=lift_clip,
        amplitude_lift_output_lufs=lift_output_lufs,
        amplitude_lift_gain=amplitude_lift_gain,
        lift_power=lift_power,
        lift_min=lift_min,
    )


def _finalize_dynamic_file_result(
    *,
    session: InferenceSession,
    prepared: _PreparedFileRequest,
    result: WindowedInferenceResult,
    normalize_peak: bool,
) -> InferenceReport:
    config = session.config
    output_path = prepared.output_audio_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    decoded = _unpatch_audio(
        result.pred_signal.cpu().float(),
        sample_count=prepared.input_sample_count,
    )
    if prepared.amplitude_lift_gain is not None:
        if prepared.amplitude_lift_mode == "wavflow":
            decoded = undo_wavflow_output_lift(
                decoded,
                scale=prepared.amplitude_lift_scale,
                sample_rate=prepared.actual_sample_rate,
                target_lufs=prepared.amplitude_lift_output_lufs,
                eps=float(getattr(config.data, "amplitude_lift_eps", 1.0e-8)),
            )
        else:
            decoded = undo_amplitude_lift(
                decoded,
                gain=prepared.amplitude_lift_gain.cpu(),
                scale=prepared.amplitude_lift_scale,
                eps=float(getattr(config.data, "amplitude_lift_eps", 1.0e-8)),
                clip_value=prepared.amplitude_lift_clip_value,
                gain_power=prepared.lift_power,
                gain_min_value=prepared.lift_min,
            )

    if normalize_peak:
        peak = decoded.abs().amax().item()
        if peak > 1.0e-8:
            decoded = decoded / peak * 0.99

    tmp_output_path = output_path.with_name(
        f".{output_path.stem}.tmp-{uuid4().hex}{output_path.suffix}"
    )
    try:
        write_audio_channels_first(
            audio_path=tmp_output_path,
            audio=decoded,
            sample_rate=prepared.actual_sample_rate,
            channel_order=config.training.downmix_channel_order,
        )
        tmp_output_path.replace(output_path)
    except Exception:
        tmp_output_path.unlink(missing_ok=True)
        raise

    return {
        "input_audio_path": str(prepared.input_audio_path),
        "output_audio_path": str(output_path),
        "checkpoint_path": str(session.checkpoint_path),
        "sample_rate": int(prepared.actual_sample_rate),
        "input_channels": int(prepared.input_channels),
        "input_samples": int(prepared.input_samples),
        "conditioning_signal_shape": prepared.conditioning_signal_shape,
        "pred_signal_shape": [int(x) for x in result.pred_signal.shape],
        "decoded_shape": [int(x) for x in decoded.shape],
        "weights_source": session.used_weights_source,
        "patch_fps": float(prepared.patch_fps),
        "chunk_seconds": float(prepared.window_seconds),
        "chunk_frames": int(prepared.window_frames),
        "overlap_seconds": float(prepared.overlap_seconds),
        "overlap_frames": int(prepared.overlap_frames),
        "solver": prepared.solver,  # type: ignore[typeddict-item]
        "solver_steps": int(prepared.solver_steps),
        "solver_rtol": float(prepared.solver_rtol),
        "solver_atol": float(prepared.solver_atol),
        "seed": int(prepared.seed),
        "mix_style": prepared.mix_style,
        "mix_style_preset": prepared.mix_style_preset,
        "amplitude_lift_enabled": bool(prepared.amplitude_lift_enabled),
        "amplitude_lift_mode": prepared.amplitude_lift_mode,
        "amplitude_lift_reference": prepared.amplitude_lift_reference,
        "amplitude_lift_target_rms": float(prepared.amplitude_lift_target_rms),
        "amplitude_lift_scale": float(prepared.amplitude_lift_scale),
        "amplitude_lift_clip_value": prepared.amplitude_lift_clip_value,
        "amplitude_lift_output_lufs": float(prepared.amplitude_lift_output_lufs),
        "amplitude_lift_gain": (
            float(prepared.amplitude_lift_gain.flatten()[0].item())
            if prepared.amplitude_lift_gain is not None
            else None
        ),
        "device": str(session.run_device),
        "compiled": bool(session.compiled),
        "compile_mode": session.compile_mode,
        "inference_dtype": str(session.run_dtype).replace("torch.", ""),
    }


@torch.inference_mode()
def run_windowed_inference_requests(
    *,
    requests: Iterable[WindowedInferenceRequest],
    model: torch.nn.Module,
    max_batch_size: int,
    max_active_requests: int | None = None,
    cuda_graph_runner: CudaGraphModelRunner | None = None,
    on_result: Callable[[WindowedInferenceResult], None] | None = None,
) -> OfflineBatchResult:
    """Run windowed requests to completion with dynamic model-query batching."""
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be > 0")
    if max_active_requests is not None and max_active_requests <= 0:
        raise ValueError("max_active_requests must be > 0 when set")

    pending = iter(requests)
    pending_exhausted = False
    active: list[WindowedInferenceRequest] = []
    results: list[WindowedInferenceResult] = []
    model_batches = 0
    model_queries = 0
    max_observed_batch_size = 0
    max_observed_active = 0
    batch_size_counts: dict[int, int] = {}

    def fill_active() -> None:
        nonlocal max_observed_active, pending_exhausted
        limit = max_active_requests if max_active_requests is not None else None
        while not pending_exhausted and (limit is None or len(active) < limit):
            try:
                request = next(pending)
            except StopIteration:
                pending_exhausted = True
                break
            request.ensure_active_controller(model=model)
            active.append(request)
        max_observed_active = max(max_observed_active, len(active))

    def execute_batches(
        query_items: list[tuple[WindowedInferenceRequest, SolverController, ModelQuery]],
    ) -> tuple[int, int, int, dict[int, int]]:
        local_batches = 0
        local_queries = 0
        local_max_batch = 0
        local_counts: dict[int, int] = {}
        query_to_controller = {id(query): controller for _, controller, query in query_items}
        for batch_queries in group_compatible_queries(
            [query for _, _, query in query_items],
            max_batch_size=max_batch_size,
        ):
            batch_size = len(batch_queries)
            batch = stack_model_queries(batch_queries)
            kwargs: dict[str, Any] = {
                "zt": batch["zt"],
                "t": batch["t"],
                "z_cond": batch["z_cond"],
                "valid_mask": batch.get("valid_mask"),
            }
            if "mem" in batch:
                kwargs["mem"] = batch["mem"]
            if "mix_style" in batch:
                kwargs["mix_style"] = batch["mix_style"]
            if "amplitude_gain" in batch:
                kwargs["amplitude_gain"] = batch["amplitude_gain"]
            if "conditioning_cache" in batch:
                kwargs["conditioning_cache"] = batch["conditioning_cache"]
            if batch_queries[0].return_mem:
                kwargs["return_mem"] = True

            output = (
                cuda_graph_runner.run(kwargs, actual_batch_size=batch_size)
                if cuda_graph_runner is not None
                else model(**kwargs)
            )
            mem_batch = None
            if batch_queries[0].return_mem:
                if not isinstance(output, tuple) or len(output) != 2:
                    raise TypeError("return_mem model query must return (prediction, mem)")
                prediction_batch, mem_batch = output
            else:
                prediction_batch = output
            if not isinstance(prediction_batch, torch.Tensor):
                raise TypeError("model must return a tensor prediction")

            local_batches += 1
            local_queries += batch_size
            local_max_batch = max(local_max_batch, batch_size)
            local_counts[batch_size] = local_counts.get(batch_size, 0) + 1
            for item_idx, query in enumerate(batch_queries):
                controller = query_to_controller[id(query)]
                mem_out = (
                    None
                    if mem_batch is None
                    else mem_batch[item_idx : item_idx + 1].contiguous()
                )
                controller.accept_output(
                    prediction_batch[item_idx : item_idx + 1].contiguous(),
                    mem_out=mem_out,
                )
        return local_batches, local_queries, local_max_batch, local_counts

    fill_active()
    while active:
        query_items: list[tuple[WindowedInferenceRequest, SolverController, ModelQuery]] = []
        for request in active:
            controller = request.ensure_active_controller(model=model)
            if controller is None:
                continue
            query = controller.next_query()
            if query is None:
                raise RuntimeError("active controller returned no query before completion")
            query_items.append((request, controller, query))
        if not query_items:
            raise RuntimeError("offline batch scheduler made no progress")

        batches, queries, batch_peak, counts = execute_batches(query_items)
        model_batches += batches
        model_queries += queries
        max_observed_batch_size = max(max_observed_batch_size, batch_peak)
        for batch_size, count in counts.items():
            batch_size_counts[batch_size] = batch_size_counts.get(batch_size, 0) + count

        still_active: list[WindowedInferenceRequest] = []
        for request in active:
            controller = request.active_controller
            if controller is not None and controller.is_done:
                request.accept_completed_controller()
                request.ensure_active_controller(model=model)
            if request.is_done:
                result = request.result()
                results.append(result)
                if on_result is not None:
                    on_result(result)
            else:
                still_active.append(request)
        active = still_active
        fill_active()

    scheduler_stats = SchedulerStats(
        completed_controllers=sum(result.window_count for result in results),
        model_batches=model_batches,
        model_queries=model_queries,
        max_observed_batch_size=max_observed_batch_size,
        max_observed_active_controllers=max_observed_active,
        batch_size_counts=batch_size_counts,
    )
    return OfflineBatchResult(
        results=results,
        stats=OfflineBatchStats(
            scheduler=scheduler_stats,
            completed_requests=len(results),
        ),
    )


@torch.inference_mode()
def run_dynamic_folder_inference(
    *,
    session: InferenceSession,
    jobs: Iterable[DynamicInferenceJob],
    sample_rate: int,
    chunk_seconds: float | None,
    overlap_seconds: float,
    solver: RequestedSolverName,
    solver_steps: int | None,
    solver_rtol: float,
    solver_atol: float,
    seed: int,
    normalize_peak: bool,
    mix_style: list[float] | dict[str, float] | None = None,
    mix_style_preset: str | None = None,
    max_batch_size: int = 4,
    max_active_requests: int | None = None,
    preprocess_workers: int = 1,
    postprocess_workers: int = 1,
    cuda_graph_runner: CudaGraphModelRunner | None = None,
) -> DynamicFolderInferenceResult:
    """Run dynamic-batched inference for a collection of filesystem jobs."""
    if bool(getattr(session.config.training, "flow_one_step", False)):
        raise ValueError("dynamic batching does not support flow_one_step inference yet")

    resolved_solver = _resolve_inference_solver(requested_solver=solver)
    if resolved_solver not in _DYNAMIC_SOLVERS:
        raise ValueError(
            "dynamic batching currently supports euler, heun, midpoint_rk2, and res6s "
            f"(got {resolved_solver})"
        )
    resolved_solver_steps = 64 if solver_steps is None else max(1, int(solver_steps))
    window_seconds = (
        float(chunk_seconds)
        if chunk_seconds is not None
        else float(session.config.data.segment_seconds)
    )
    if preprocess_workers <= 0:
        raise ValueError("preprocess_workers must be > 0")
    if postprocess_workers <= 0:
        raise ValueError("postprocess_workers must be > 0")

    prepared_by_id: dict[str, _PreparedFileRequest] = {}
    errors: list[tuple[Path, Exception]] = []

    def iter_requests() -> Iterable[WindowedInferenceRequest]:
        def prepare(index: int, job: DynamicInferenceJob) -> _PreparedFileRequest:
            request_id = str(index)
            return _prepare_dynamic_file_request(
                session=session,
                job=job,
                request_id=request_id,
                sample_rate=sample_rate,
                window_seconds=window_seconds,
                overlap_seconds=overlap_seconds,
                solver=resolved_solver,
                solver_steps=resolved_solver_steps,
                solver_rtol=solver_rtol,
                solver_atol=solver_atol,
                seed=seed,
                mix_style=mix_style,
                mix_style_preset=mix_style_preset,
            )

        if preprocess_workers == 1:
            for index, job in enumerate(jobs):
                try:
                    prepared = prepare(index, job)
                except Exception as exc:
                    errors.append((Path(job.input_audio_path), exc))
                    continue
                prepared_by_id[prepared.request.request_id] = prepared
                yield prepared.request
            return

        indexed_jobs = iter(enumerate(jobs))
        max_in_flight = max(
            preprocess_workers,
            2 * int(max_active_requests or max_batch_size),
        )
        with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
            futures: dict[Future[_PreparedFileRequest], DynamicInferenceJob] = {}

            def submit_next() -> bool:
                try:
                    index, job = next(indexed_jobs)
                except StopIteration:
                    return False
                futures[executor.submit(prepare, index, job)] = job
                return True

            for _ in range(max_in_flight):
                if not submit_next():
                    break

            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    job = futures.pop(future)
                    submit_next()
                    try:
                        prepared = future.result()
                    except Exception as exc:
                        errors.append((Path(job.input_audio_path), exc))
                        continue
                    prepared_by_id[prepared.request.request_id] = prepared
                    yield prepared.request

    reports: list[InferenceReport] = []
    postprocess_futures: dict[Future[InferenceReport], _PreparedFileRequest] = {}
    postprocess_executor = (
        ThreadPoolExecutor(max_workers=postprocess_workers)
        if postprocess_workers > 1
        else None
    )

    def handle_completed_result(result: WindowedInferenceResult) -> None:
        prepared = prepared_by_id[result.request_id]
        if postprocess_executor is None:
            try:
                reports.append(
                    _finalize_dynamic_file_result(
                        session=session,
                        prepared=prepared,
                        result=result,
                        normalize_peak=normalize_peak,
                    )
                )
            except Exception as exc:
                errors.append((prepared.input_audio_path, exc))
            return
        future = postprocess_executor.submit(
            _finalize_dynamic_file_result,
            session=session,
            prepared=prepared,
            result=result,
            normalize_peak=normalize_peak,
        )
        postprocess_futures[future] = prepared

    try:
        batch_result = run_windowed_inference_requests(
            requests=iter_requests(),
            model=session.model,
            max_batch_size=max_batch_size,
            max_active_requests=max_active_requests,
            cuda_graph_runner=cuda_graph_runner,
            on_result=handle_completed_result,
        )
    finally:
        if postprocess_executor is not None:
            postprocess_executor.shutdown(wait=True)

    for future, prepared in postprocess_futures.items():
        try:
            reports.append(future.result())
        except Exception as exc:
            errors.append((prepared.input_audio_path, exc))

    return DynamicFolderInferenceResult(
        reports=reports,
        errors=errors,
        stats=batch_result.stats,
    )
