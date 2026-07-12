"""CUDA-only inference microprofiler for SpatialDiT fixed-window sampling.

This profiles the model/solver hot path, not audio decode/write. It intentionally
uses random CUDA tensors so kernel timings are not hidden by disk or CPU work.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stereo2spatial.inference.cuda_graphs import CudaGraphModelRunner  # noqa: E402
from stereo2spatial.inference.runner import (  # noqa: E402
    _build_model_from_config,
    _resolve_inference_dtype,
)
from stereo2spatial.inference.sdpa import sdpa_backend_context  # noqa: E402
from stereo2spatial.inference.solvers import (  # noqa: E402
    _CLEAN_PREDICTION_EPS,
    _INTEGRATION_T_END,
    clean_prediction_to_velocity,
)
from stereo2spatial.training.config import load_config  # noqa: E402


@dataclass(frozen=True)
class ProfileResult:
    config: str
    device: str
    dtype: str
    batch_size: int
    frames: int
    patch_size: int
    target_channels: int
    cond_channels: int
    hidden_dim: int
    num_layers: int
    waveform_level_depth: int
    solver: str
    solver_steps: int
    model_forwards: int
    cache_enabled: bool
    compile_enabled: bool
    cuda_graphs_enabled: bool
    sdpa_backend: str
    cache_build_ms: float
    loop_ms_mean: float
    loop_ms_min: float
    loop_ms_max: float
    ms_per_model_forward: float
    peak_allocated_mb: float
    peak_reserved_mb: float


def _parse_csv_ints(raw: str) -> list[int]:
    values = []
    for item in str(raw).replace(";", ",").split(","):
        clean = item.strip()
        if clean:
            values.append(int(clean))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _cuda_event_time_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _init_memory(model: torch.nn.Module, batch_size: int, dtype: torch.dtype) -> torch.Tensor | None:
    init_memory = getattr(model, "init_memory", None)
    if init_memory is None and hasattr(model, "_orig_mod"):
        init_memory = getattr(model._orig_mod, "init_memory", None)
    if init_memory is None:
        return None
    return init_memory(
        batch_size=batch_size,
        device=torch.device("cuda"),
        dtype=dtype,
    )


def _build_conditioning_cache(
    model: torch.nn.Module,
    *,
    z_cond: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> dict[str, object] | None:
    cache_builder = getattr(model, "build_conditioning_cache", None)
    if cache_builder is None and hasattr(model, "_orig_mod"):
        cache_builder = getattr(model._orig_mod, "build_conditioning_cache", None)
    if cache_builder is None:
        return None
    return cache_builder(z_cond=z_cond, valid_mask=valid_mask)


@torch.inference_mode()
def _run_midpoint_rk2_window(
    *,
    model: torch.nn.Module,
    z_cond: torch.Tensor,
    z0: torch.Tensor,
    valid_mask: torch.Tensor | None,
    mem: torch.Tensor | None,
    solver_steps: int,
    mix_style: torch.Tensor | None,
    amplitude_gain: torch.Tensor | None,
    conditioning_cache: dict[str, object] | None,
    cuda_graph_runner: CudaGraphModelRunner | None,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    batch_size = int(z_cond.shape[0])
    dtype = z_cond.dtype
    device = z_cond.device
    dt = _INTEGRATION_T_END / float(solver_steps)
    z_state = z0
    model_forwards = 0
    t_buffer = torch.empty((batch_size,), device=device, dtype=dtype)

    def predict_clean(t_value: float, state: torch.Tensor) -> torch.Tensor:
        nonlocal model_forwards
        t = t_buffer.fill_(float(t_value))
        kwargs: dict[str, Any] = {
            "zt": state,
            "t": t,
            "z_cond": z_cond,
            "valid_mask": valid_mask,
        }
        if mem is not None:
            kwargs["mem"] = mem
        if mix_style is not None:
            kwargs["mix_style"] = mix_style
        if amplitude_gain is not None:
            kwargs["amplitude_gain"] = amplitude_gain
        if conditioning_cache is not None:
            kwargs["conditioning_cache"] = conditioning_cache
        model_forwards += 1
        if cuda_graph_runner is not None:
            return cuda_graph_runner.run(kwargs, actual_batch_size=batch_size)
        return model(**kwargs)

    for step_idx in range(int(solver_steps)):
        t0 = float(step_idx) * dt
        t_mid = min(t0 + 0.5 * dt, _INTEGRATION_T_END)
        clean0 = predict_clean(t0, z_state)
        v0 = clean_prediction_to_velocity(clean0, z_state, t0)
        z_mid = z_state + 0.5 * dt * v0
        clean_mid = predict_clean(t_mid, z_mid)
        v_mid = clean_prediction_to_velocity(clean_mid, z_mid, t_mid)
        z_state = z_state + dt * v_mid

    final_clean = predict_clean(_INTEGRATION_T_END, z_state)
    if mem is None:
        return final_clean, None, model_forwards

    final_t = torch.full(
        (batch_size,),
        1.0 - _CLEAN_PREDICTION_EPS,
        device=device,
        dtype=dtype,
    )
    kwargs = {
        "zt": final_clean,
        "t": final_t,
        "z_cond": z_cond,
        "valid_mask": valid_mask,
        "mem": mem,
        "return_mem": True,
    }
    if mix_style is not None:
        kwargs["mix_style"] = mix_style
    if amplitude_gain is not None:
        kwargs["amplitude_gain"] = amplitude_gain
    if conditioning_cache is not None:
        kwargs["conditioning_cache"] = conditioning_cache
    if cuda_graph_runner is not None:
        _, mem_out = cuda_graph_runner.run(kwargs, actual_batch_size=batch_size)
    else:
        _, mem_out = model(**kwargs)
    model_forwards += 1
    return final_clean, mem_out, model_forwards


def _profile_case(
    *,
    config_path: Path,
    batch_size: int,
    frames: int,
    dtype_name: str,
    solver_steps: int,
    warmup: int,
    iters: int,
    use_cache: bool,
    compile_model: bool,
    compile_mode: str,
    cuda_graphs: bool,
    sdpa_backend: str,
    valid_mask_mode: str,
    profiler_table_output: Path | None = None,
    profiler_top_rows: int = 30,
) -> ProfileResult:
    config = load_config(config_path)
    device = torch.device("cuda")
    dtype = _resolve_inference_dtype(requested_dtype=dtype_name, device=device)
    model = _build_model_from_config(config).to(device=device, dtype=dtype)
    model.eval()
    if compile_model:
        model = torch.compile(model, mode=compile_mode)  # type: ignore[assignment]
    cuda_graph_runner = (
        CudaGraphModelRunner(model, bucket_sizes=(batch_size,), clone_outputs=False)
        if cuda_graphs
        else None
    )

    target_channels = int(config.model.target_channels)
    cond_channels = int(config.model.cond_channels)
    patch_size = int(config.model.patch_size)
    generator = torch.Generator(device=device)
    generator.manual_seed(1234)
    z_cond = torch.randn(
        (batch_size, cond_channels, patch_size, frames),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    z0 = torch.randn(
        (batch_size, target_channels, patch_size, frames),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    valid_mask = None
    if valid_mask_mode == "on":
        valid_mask = torch.ones((batch_size, frames), device=device, dtype=torch.bool)
    elif valid_mask_mode != "off":
        raise ValueError("valid_mask_mode must be on or off")
    mem = _init_memory(model, batch_size=batch_size, dtype=dtype)
    mix_style = None
    if int(getattr(config.model, "mix_style_dim", 0)) > 0:
        mix_style = torch.full(
            (batch_size, int(config.model.mix_style_dim)),
            0.5,
            device=device,
            dtype=dtype,
        )
    amplitude_gain = None
    if bool(getattr(config.model, "amplitude_gain_conditioning", False)):
        amplitude_gain = torch.zeros((batch_size, 1), device=device, dtype=dtype)

    cache: dict[str, object] | None = None
    cache_build_ms = 0.0
    if use_cache:
        cache_build_ms = _cuda_event_time_ms(
            lambda: _build_conditioning_cache(
                model,
                z_cond=z_cond,
                valid_mask=valid_mask,
            )
        )
        cache = _build_conditioning_cache(model, z_cond=z_cond, valid_mask=valid_mask)

    def run_once() -> tuple[torch.Tensor, torch.Tensor | None, int]:
        with sdpa_backend_context(sdpa_backend):
            return _run_midpoint_rk2_window(
                model=model,
                z_cond=z_cond,
                z0=z0,
                valid_mask=valid_mask,
                mem=mem,
                solver_steps=solver_steps,
                mix_style=mix_style,
                amplitude_gain=amplitude_gain,
                conditioning_cache=cache,
                cuda_graph_runner=cuda_graph_runner,
            )

    model_forwards = 0
    for _ in range(warmup):
        _, _, model_forwards = run_once()
    torch.cuda.synchronize()

    if profiler_table_output is not None:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
        ) as profiler:
            run_once()
        torch.cuda.synchronize()
        table = profiler.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=int(profiler_top_rows),
        )
        profiler_table_output.parent.mkdir(parents=True, exist_ok=True)
        profiler_table_output.write_text(table, encoding="utf-8")

    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(iters):
        times.append(_cuda_event_time_ms(run_once))
    peak_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)

    loop_mean = sum(times) / len(times)
    return ProfileResult(
        config=str(config_path),
        device=torch.cuda.get_device_name(0),
        dtype=str(dtype).replace("torch.", ""),
        batch_size=batch_size,
        frames=frames,
        patch_size=patch_size,
        target_channels=target_channels,
        cond_channels=cond_channels,
        hidden_dim=int(config.model.hidden_dim),
        num_layers=int(config.model.num_layers),
        waveform_level_depth=int(getattr(config.model, "waveform_level_depth", 0)),
        solver="midpoint_rk2",
        solver_steps=int(solver_steps),
        model_forwards=int(model_forwards),
        cache_enabled=bool(use_cache),
        compile_enabled=bool(compile_model),
        cuda_graphs_enabled=bool(cuda_graphs),
        sdpa_backend=str(sdpa_backend),
        cache_build_ms=cache_build_ms,
        loop_ms_mean=loop_mean,
        loop_ms_min=min(times),
        loop_ms_max=max(times),
        ms_per_model_forward=loop_mean / float(model_forwards),
        peak_allocated_mb=peak_allocated,
        peak_reserved_mb=peak_reserved,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA-only SpatialDiT inference profiler.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16", "auto"))
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--solver-steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--cache-modes", default="off,on", help="Comma list: off,on")
    parser.add_argument("--valid-mask", default="off", choices=("off", "on"))
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument(
        "--sdpa-backend",
        default="auto",
        choices=("auto", "flash", "efficient", "math"),
    )
    parser.add_argument(
        "--profiler-table-output",
        default=None,
        help=(
            "Optional path for a torch.profiler top-ops table. Only use with a "
            "single batch/cache case; profiler overhead is high."
        ),
    )
    parser.add_argument("--profiler-top-rows", type=int, default=30)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. Refusing to run CPU inference profiling.")

    config = load_config(args.config)
    frames = (
        int(args.frames)
        if args.frames is not None
        else int(round(float(args.window_seconds) * int(args.sample_rate) / int(config.model.patch_size)))
    )
    batch_sizes = _parse_csv_ints(args.batch_sizes)
    cache_modes = [mode.strip().lower() for mode in str(args.cache_modes).split(",") if mode.strip()]
    profiler_table_output = Path(args.profiler_table_output) if args.profiler_table_output else None
    if profiler_table_output is not None and (
        len(batch_sizes) != 1 or len(cache_modes) != 1
    ):
        raise ValueError(
            "--profiler-table-output requires exactly one batch size and one cache mode"
        )
    results = []
    for batch_size in batch_sizes:
        for cache_mode in cache_modes:
            if cache_mode not in {"off", "on"}:
                raise ValueError("--cache-modes entries must be off or on")
            gc.collect()
            torch.cuda.empty_cache()
            result = _profile_case(
                config_path=Path(args.config),
                batch_size=batch_size,
                frames=frames,
                dtype_name=args.dtype,
                solver_steps=int(args.solver_steps),
                warmup=int(args.warmup),
                iters=int(args.iters),
                use_cache=cache_mode == "on",
                compile_model=bool(args.compile_model),
                compile_mode=str(args.compile_mode),
                cuda_graphs=bool(args.cuda_graphs),
                sdpa_backend=str(args.sdpa_backend),
                valid_mask_mode=str(args.valid_mask),
                profiler_table_output=profiler_table_output,
                profiler_top_rows=int(args.profiler_top_rows),
            )
            results.append(result)
            print(
                "profile "
                f"batch={result.batch_size} cache={result.cache_enabled} "
                f"dtype={result.dtype} frames={result.frames} "
                f"steps={result.solver_steps} forwards={result.model_forwards} "
                f"cache_build_ms={result.cache_build_ms:.3f} "
                f"cuda_graphs={result.cuda_graphs_enabled} "
                f"sdpa={result.sdpa_backend} "
                f"loop_ms={result.loop_ms_mean:.3f} "
                f"ms_per_forward={result.ms_per_model_forward:.3f} "
                f"peak_allocated_mb={result.peak_allocated_mb:.1f}"
            )

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps([asdict(result) for result in results], indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
