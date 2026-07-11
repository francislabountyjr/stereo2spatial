from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import stereo2spatial.inference.offline_batch as offline_batch
from stereo2spatial.inference.offline_batch import (
    DynamicInferenceJob,
    WindowedInferenceRequest,
    run_dynamic_folder_inference,
    run_windowed_inference_requests,
)
from stereo2spatial.inference.runner import InferenceSession
from stereo2spatial.inference.sampling import generate_spatial_signal
from stereo2spatial.inference.timestep_sampling import (
    generate_spatial_signal_timestep_major,
)


class _RecordingCleanPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []
        self.t_batches: list[torch.Tensor] = []
        self.return_mem_batches = 0

    def forward(
        self,
        *,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor,
        mem: torch.Tensor | None = None,
        return_mem: bool = False,
        mix_style: torch.Tensor | None = None,
        amplitude_gain: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del valid_mask
        clean = 0.15 * zt + 0.45 * z_cond + t.view(-1, 1, 1, 1).to(zt) * 0.1
        if mix_style is not None:
            clean = clean + mix_style.mean(dim=-1).view(-1, 1, 1, 1) * 0.01
        if amplitude_gain is not None:
            clean = clean + amplitude_gain.view(-1, 1, 1, 1).to(clean) * 0.001
        self.batch_sizes.append(int(zt.shape[0]))
        self.t_batches.append(t.detach().cpu())
        if return_mem:
            self.return_mem_batches += 1
            if mem is None:
                raise AssertionError("return_mem requires mem")
            return clean, mem + 1.0
        return clean


class _ParityCleanPredictor(torch.nn.Module):
    cond_channels = 2
    target_channels = 2
    patch_size = 3

    def init_memory(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        del batch_size, device, dtype
        return None

    def forward(
        self,
        *,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        del valid_mask
        return 0.2 * zt + 0.3 * z_cond + 0.05 * t.view(-1, 1, 1, 1).to(zt)


class _MemorySweepPredictor(torch.nn.Module):
    cond_channels = 2
    target_channels = 2
    patch_size = 3

    def __init__(self) -> None:
        super().__init__()
        self.window_starts: list[float] = []
        self.memory_inputs: list[float] = []

    def init_memory(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros((batch_size, 1, 1), device=device, dtype=dtype)

    def forward(
        self,
        *,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor | None,
        mem: torch.Tensor,
        return_mem: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del valid_mask
        if not return_mem:
            raise AssertionError("memory sweeps must request updated memory")
        self.window_starts.extend(
            float(value) for value in z_cond[:, 0, 0, 0].detach().cpu().tolist()
        )
        self.memory_inputs.extend(
            float(value) for value in mem[:, 0, 0].detach().cpu().tolist()
        )
        clean = (
            0.2 * zt
            + 0.3 * z_cond
            + 0.01 * mem[:, :, :, None]
            + 0.05 * t.view(-1, 1, 1, 1).to(zt)
        )
        return clean, mem + 1.0


def _request(
    request_id: str,
    *,
    total_frames: int,
    solver_steps: int = 2,
    mem: torch.Tensor | None = None,
) -> WindowedInferenceRequest:
    generator = torch.Generator().manual_seed(999 + len(request_id))
    cond = torch.randn(2, 3, total_frames, generator=generator)
    return WindowedInferenceRequest(
        request_id=request_id,
        cond_signal=cond,
        target_channels=2,
        window_frames=10,
        overlap_frames=2,
        solver="euler",
        solver_steps=solver_steps,
        seed=123,
        mem=mem,
        mix_style=torch.zeros(1, 2),
        amplitude_gain=torch.zeros(1, 1),
    )


def _session(tmp_path: Path, model: torch.nn.Module) -> InferenceSession:
    config = SimpleNamespace(
        output_dir=str(tmp_path / "run"),
        model=SimpleNamespace(
            target_channels=2,
            cond_channels=2,
            patch_size=2,
            mix_style_dim=0,
        ),
        data=SimpleNamespace(
            segment_seconds=2.0,
            amplitude_lift_enabled=False,
            amplitude_lift_mode="rms",
            amplitude_lift_reference="source",
            amplitude_lift_target_rms=0.33,
            amplitude_lift_scale=3.0,
            amplitude_lift_clip_value=4.0,
            amplitude_lift_output_lufs=-23.0,
        ),
        training=SimpleNamespace(
            downmix_channel_order=None,
            flow_one_step=False,
        ),
    )
    return InferenceSession(
        config=config,  # type: ignore[arg-type]
        model=model,
        checkpoint_path=tmp_path / "checkpoint",
        run_device=torch.device("cpu"),
        used_weights_source="student",
        compiled=False,
        compile_mode=None,
    )


def _legacy_session(tmp_path: Path, model: torch.nn.Module) -> InferenceSession:
    config = SimpleNamespace(
        output_dir=str(tmp_path / "run"),
        model=SimpleNamespace(
            architecture="legacy_vae",
            target_channels=2,
            cond_channels=1,
            patch_size=3,
            latent_dim=3,
            mix_style_dim=0,
        ),
        data=SimpleNamespace(
            segment_seconds=0.04,
            latent_fps=50.0,
            amplitude_lift_enabled=False,
            amplitude_lift_mode="rms",
            amplitude_lift_reference="source",
            amplitude_lift_target_rms=0.33,
            amplitude_lift_scale=3.0,
            amplitude_lift_clip_value=4.0,
            amplitude_lift_output_lufs=-23.0,
        ),
        training=SimpleNamespace(
            downmix_channel_order=None,
            flow_one_step=False,
        ),
    )
    return InferenceSession(
        config=config,  # type: ignore[arg-type]
        model=model,
        checkpoint_path=tmp_path / "checkpoint",
        run_device=torch.device("cpu"),
        used_weights_source="student",
        compiled=False,
        compile_mode=None,
        vae=torch.nn.Identity(),
    )


def test_windowed_requests_batch_multiple_songs_and_stitch_to_original_length() -> None:
    model = _RecordingCleanPredictor()
    requests = [
        _request("a", total_frames=19, solver_steps=1),
        _request("b", total_frames=12, solver_steps=3),
    ]

    result = run_windowed_inference_requests(
        requests=requests,
        model=model,
        max_batch_size=2,
        sampling_order="window_major",
    )

    assert result.stats.completed_requests == 2
    assert result.stats.scheduler.max_observed_batch_size == 2
    by_id = {item.request_id: item for item in result.results}
    assert by_id["a"].pred_signal.shape == (2, 3, 19)
    assert by_id["a"].window_count == 3
    assert by_id["b"].pred_signal.shape == (2, 3, 12)
    assert by_id["b"].window_count == 2
    assert any(
        batch.numel() == 2 and batch.unique().numel() == 2 for batch in model.t_batches
    )


def test_windowed_requests_call_result_callback() -> None:
    model = _RecordingCleanPredictor()
    seen: list[str] = []

    result = run_windowed_inference_requests(
        requests=[
            _request("a", total_frames=10, solver_steps=1),
            _request("b", total_frames=10, solver_steps=1),
        ],
        model=model,
        max_batch_size=2,
        on_result=lambda item: seen.append(item.request_id),
        sampling_order="window_major",
    )

    assert result.stats.completed_requests == 2
    assert set(seen) == {"a", "b"}


def test_windowed_requests_carry_memory_between_sequential_windows() -> None:
    model = _RecordingCleanPredictor()
    mem = torch.zeros(1, 4, 8)
    request = _request("memory", total_frames=19, solver_steps=1, mem=mem)

    result = run_windowed_inference_requests(
        requests=[request],
        model=model,
        max_batch_size=2,
        sampling_order="window_major",
    )

    completed = result.results[0]
    assert completed.window_count == 3
    assert completed.final_memory is not None
    assert torch.allclose(completed.final_memory, mem + 3.0)
    assert model.return_mem_batches == 3


def test_windowed_requests_respect_max_active_requests() -> None:
    model = _RecordingCleanPredictor()
    requests = [
        _request("a", total_frames=10),
        _request("b", total_frames=10),
        _request("c", total_frames=10),
    ]

    result = run_windowed_inference_requests(
        requests=requests,
        model=model,
        max_batch_size=8,
        max_active_requests=2,
        sampling_order="window_major",
    )

    assert result.stats.completed_requests == 3
    assert result.stats.scheduler.max_observed_active_controllers == 2
    assert result.stats.scheduler.max_observed_batch_size == 2


@pytest.mark.parametrize("solver", ["euler", "heun", "midpoint_rk2", "res6s"])
def test_dynamic_windowing_matches_sequential_full_song_noise(
    solver: str,
) -> None:
    model = _ParityCleanPredictor()
    cond = torch.randn(2, 3, 23, generator=torch.Generator().manual_seed(77))
    seed = 1234

    sequential = generate_spatial_signal(
        model=model,  # type: ignore[arg-type]
        cond_signal=cond,
        chunk_frames=10,
        overlap_frames=3,
        solver=solver,
        solver_steps=2,
        solver_rtol=1.0e-5,
        solver_atol=1.0e-5,
        seed=seed,
    )
    dynamic = (
        run_windowed_inference_requests(
            requests=[
                WindowedInferenceRequest(
                    request_id="song",
                    cond_signal=cond,
                    target_channels=2,
                    window_frames=10,
                    overlap_frames=3,
                    solver=solver,  # type: ignore[arg-type]
                    solver_steps=2,
                    seed=seed,
                )
            ],
            model=model,
            max_batch_size=1,
            sampling_order="window_major",
        )
        .results[0]
        .pred_signal
    )

    torch.testing.assert_close(dynamic, sequential)


@pytest.mark.parametrize("solver", ["euler", "heun", "midpoint_rk2", "res6s"])
def test_dynamic_timestep_major_matches_reference_sampler(solver: str) -> None:
    cond = torch.randn(2, 3, 23, generator=torch.Generator().manual_seed(77))
    seed = 1234
    reference_model = _MemorySweepPredictor()
    reference = generate_spatial_signal_timestep_major(
        model=reference_model,  # type: ignore[arg-type]
        cond_signal=cond,
        chunk_frames=10,
        overlap_frames=3,
        solver=solver,
        solver_steps=2,
        solver_rtol=1.0e-5,
        solver_atol=1.0e-5,
        seed=seed,
    )
    dynamic_model = _MemorySweepPredictor()
    initial_mem = dynamic_model.init_memory(
        batch_size=1,
        device=cond.device,
        dtype=cond.dtype,
    )
    dynamic = (
        run_windowed_inference_requests(
            requests=[
                WindowedInferenceRequest(
                    request_id="song",
                    cond_signal=cond,
                    target_channels=2,
                    window_frames=10,
                    overlap_frames=3,
                    solver=solver,  # type: ignore[arg-type]
                    solver_steps=2,
                    seed=seed,
                    mem=initial_mem,
                )
            ],
            model=dynamic_model,
            max_batch_size=1,
        )
        .results[0]
        .pred_signal
    )

    torch.testing.assert_close(dynamic, reference)


def test_dynamic_timestep_major_resets_memory_and_preserves_window_order() -> None:
    model = _MemorySweepPredictor()
    frame_values = torch.arange(8, dtype=torch.float32)
    cond = frame_values.view(1, 1, -1).expand(2, 3, -1).contiguous()
    initial_mem = model.init_memory(
        batch_size=1,
        device=cond.device,
        dtype=cond.dtype,
    )

    result = run_windowed_inference_requests(
        requests=[
            WindowedInferenceRequest(
                request_id="memory-song",
                cond_signal=cond,
                target_channels=2,
                window_frames=4,
                overlap_frames=1,
                solver="euler",
                solver_steps=1,
                seed=123,
                mem=initial_mem,
            )
        ],
        model=model,
        max_batch_size=1,
    )

    # Euler/1 evaluates the global clean field at t=0 and once more at the
    # endpoint. Each evaluation must restart at memory zero, then sweep 0,3,6.
    assert model.window_starts == [0.0, 3.0, 6.0, 0.0, 3.0, 6.0]
    assert model.memory_inputs == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    assert result.results[0].final_memory is None


def test_dynamic_timestep_major_batches_ready_windows_across_songs() -> None:
    model = _RecordingCleanPredictor()
    requests = [
        _request("a", total_frames=19, solver_steps=1),
        _request("b", total_frames=19, solver_steps=1),
    ]

    result = run_windowed_inference_requests(
        requests=requests,
        model=model,
        max_batch_size=2,
        max_active_requests=2,
    )

    assert result.stats.completed_requests == 2
    assert result.stats.scheduler.completed_controllers == 6
    assert result.stats.scheduler.max_observed_batch_size == 2
    assert result.stats.scheduler.max_observed_active_controllers == 2
    assert all(batch_size == 2 for batch_size in model.batch_sizes)


def test_dynamic_window_noise_shares_overlaps_and_zero_pads_training_tail() -> None:
    request = WindowedInferenceRequest(
        request_id="noise",
        cond_signal=torch.zeros(2, 3, 8),
        target_channels=2,
        window_frames=4,
        overlap_frames=1,
        solver="euler",
        solver_steps=1,
        seed=123,
    )
    captured_z0: list[torch.Tensor] = []

    while not request.is_done:
        controller = request.ensure_active_controller()
        assert controller is not None
        while not controller.is_done:
            query = controller.next_query()
            assert query is not None
            if query.query_index == 0:
                captured_z0.append(query.zt.detach().clone())
            controller.accept_output(query.zt, query.mem if query.return_mem else None)
        request.accept_completed_controller()

    assert len(captured_z0) == 3
    torch.testing.assert_close(captured_z0[0][..., -1:], captured_z0[1][..., :1])
    torch.testing.assert_close(captured_z0[1][..., -1:], captured_z0[2][..., :1])
    assert torch.count_nonzero(captured_z0[2][..., 2:]).item() == 0


def test_dynamic_window_stitching_accumulates_in_float32() -> None:
    model = _ParityCleanPredictor().to(dtype=torch.bfloat16)
    cond = torch.zeros((2, 3, 19), dtype=torch.bfloat16)
    result = run_windowed_inference_requests(
        requests=[
            WindowedInferenceRequest(
                request_id="low-precision",
                cond_signal=cond,
                target_channels=2,
                window_frames=10,
                overlap_frames=2,
                solver="euler",
                solver_steps=1,
                seed=123,
            )
        ],
        model=model,
        max_batch_size=1,
        sampling_order="window_major",
    ).results[0]

    assert result.pred_signal.dtype == torch.float32


def test_dynamic_folder_inference_runs_jobs_with_batched_model_queries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = _RecordingCleanPredictor()
    session = _session(tmp_path, model)
    writes: list[tuple[Path, torch.Tensor, int]] = []

    def fake_read(
        audio_path: Path, target_sample_rate: int
    ) -> tuple[torch.Tensor, int]:
        del audio_path
        return torch.randn(2, 37), target_sample_rate

    def fake_write(
        audio_path: Path,
        audio: torch.Tensor,
        sample_rate: int,
        channel_order: list[str] | None = None,
    ) -> None:
        del channel_order
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"fake audio")
        writes.append((audio_path, audio, sample_rate))

    monkeypatch.setattr(offline_batch, "read_audio_channels_first", fake_read)
    monkeypatch.setattr(offline_batch, "write_audio_channels_first", fake_write)
    jobs = [
        DynamicInferenceJob(
            input_audio_path=tmp_path / "a.wav",
            output_audio_path=tmp_path / "out" / "a.flac",
        ),
        DynamicInferenceJob(
            input_audio_path=tmp_path / "b.wav",
            output_audio_path=tmp_path / "out" / "b.flac",
        ),
    ]

    result = run_dynamic_folder_inference(
        session=session,
        jobs=jobs,
        sample_rate=8,
        chunk_seconds=2.0,
        overlap_seconds=0.5,
        solver="midpoint_rk2",
        solver_steps=2,
        solver_rtol=1.0e-5,
        solver_atol=1.0e-5,
        seed=123,
        normalize_peak=False,
        max_batch_size=2,
        max_active_requests=2,
        preprocess_workers=2,
        postprocess_workers=2,
    )

    assert len(result.errors) == 0
    assert len(result.reports) == 2
    assert len(writes) == 2
    assert {Path(report["output_audio_path"]) for report in result.reports} == {
        tmp_path / "out" / "a.flac",
        tmp_path / "out" / "b.flac",
    }
    assert all(
        report["sampling_order"] == "timestep_major" for report in result.reports
    )
    assert (tmp_path / "out" / "a.flac").exists()
    assert (tmp_path / "out" / "b.flac").exists()
    assert all(
        path.name.startswith(".") and ".tmp-" in path.name for path, _, _ in writes
    )
    assert result.stats.scheduler.max_observed_batch_size == 2
    assert result.stats.scheduler.completed_controllers == 6
    assert writes[0][1].shape == (2, 37)


def test_dynamic_folder_inference_accepts_hyphenated_timestep_major(
    tmp_path: Path,
) -> None:
    result = run_dynamic_folder_inference(
        session=_session(tmp_path, _RecordingCleanPredictor()),
        jobs=[],
        sample_rate=8,
        chunk_seconds=2.0,
        overlap_seconds=0.5,
        solver="euler",
        solver_steps=1,
        solver_rtol=1.0e-5,
        solver_atol=1.0e-5,
        seed=123,
        normalize_peak=False,
        sampling_order="timestep-major",  # type: ignore[arg-type]
    )

    assert result.reports == []
    assert result.errors == []
    assert result.stats.completed_requests == 0


def test_dynamic_folder_inference_supports_legacy_vae_codec_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = _RecordingCleanPredictor()
    session = _legacy_session(tmp_path, model)
    encode_shapes: list[tuple[int, ...]] = []
    decode_shapes: list[tuple[int, ...]] = []
    encode_controls: list[tuple[object, object]] = []
    decode_controls: list[tuple[object, object, object]] = []
    writes: list[torch.Tensor] = []

    def fake_read(
        audio_path: Path, target_sample_rate: int
    ) -> tuple[torch.Tensor, int]:
        del audio_path
        return torch.randn(2, 960), target_sample_rate

    def fake_encode(**kwargs: object) -> torch.Tensor:
        audio = kwargs["audio"]
        assert isinstance(audio, torch.Tensor)
        encode_shapes.append(tuple(audio.shape))
        encode_controls.append(
            (kwargs["chunk_size_samples"], kwargs["overlap_samples"])
        )
        return torch.ones(3, 3)

    def fake_decode(**kwargs: object) -> torch.Tensor:
        channel_latents = kwargs["channel_latents"]
        assert isinstance(channel_latents, torch.Tensor)
        decode_shapes.append(tuple(channel_latents.shape))
        decode_controls.append(
            (
                kwargs["use_chunked_decode"],
                kwargs["chunk_size_frames"],
                kwargs["overlap_frames"],
            )
        )
        return torch.full((2, 960), 0.25)

    def fake_write(
        audio_path: Path,
        audio: torch.Tensor,
        sample_rate: int,
        channel_order: list[str] | None = None,
    ) -> None:
        del sample_rate, channel_order
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"fake audio")
        writes.append(audio.clone())

    monkeypatch.setattr(offline_batch, "read_audio_channels_first", fake_read)
    monkeypatch.setattr(offline_batch, "vae_encode", fake_encode)
    monkeypatch.setattr(offline_batch, "decode_channels_independent", fake_decode)
    monkeypatch.setattr(offline_batch, "write_audio_channels_first", fake_write)

    jobs = [
        DynamicInferenceJob(
            input_audio_path=tmp_path / f"{name}.wav",
            output_audio_path=tmp_path / "out" / f"{name}.wav",
        )
        for name in ("a", "b")
    ]
    result = run_dynamic_folder_inference(
        session=session,
        jobs=jobs,
        sample_rate=48_000,
        chunk_seconds=0.04,
        overlap_seconds=0.0,
        solver="euler",
        solver_steps=1,
        solver_rtol=1.0e-5,
        solver_atol=1.0e-5,
        seed=123,
        normalize_peak=False,
        max_batch_size=2,
        max_active_requests=2,
        encode_chunk_size_samples=480,
        encode_overlap_samples=48,
        decode_chunk_size_frames=17,
        decode_overlap_frames=3,
        disable_chunked_decode=True,
        sampling_order="window_major",
    )

    assert result.errors == []
    assert len(result.reports) == 2
    assert result.stats.scheduler.max_observed_batch_size == 2
    assert encode_shapes == [(2, 960), (2, 960)]
    assert decode_shapes == [(2, 3, 3), (2, 3, 3)]
    assert encode_controls == [(480, 48), (480, 48)]
    assert decode_controls == [(False, 17, 3), (False, 17, 3)]
    assert [tuple(audio.shape) for audio in writes] == [(2, 960), (2, 960)]
    assert all(report["architecture"] == "legacy_vae" for report in result.reports)
    assert all(report["representation"] == "latent" for report in result.reports)
    assert all(report["sampling_order"] == "window_major" for report in result.reports)


def test_legacy_dynamic_folder_inference_rejects_parallel_vae_workers(
    tmp_path: Path,
) -> None:
    session = _legacy_session(tmp_path, _RecordingCleanPredictor())
    with pytest.raises(ValueError, match="requires preprocess_workers=1"):
        run_dynamic_folder_inference(
            session=session,
            jobs=[],
            sample_rate=48_000,
            chunk_seconds=1.0,
            overlap_seconds=0.0,
            solver="euler",
            solver_steps=1,
            solver_rtol=1.0e-5,
            solver_atol=1.0e-5,
            seed=123,
            normalize_peak=False,
            preprocess_workers=2,
        )


def test_legacy_dynamic_folder_inference_requires_48khz(tmp_path: Path) -> None:
    session = _legacy_session(tmp_path, _RecordingCleanPredictor())
    with pytest.raises(ValueError, match="requires sample_rate=48000"):
        run_dynamic_folder_inference(
            session=session,
            jobs=[],
            sample_rate=44_100,
            chunk_seconds=1.0,
            overlap_seconds=0.0,
            solver="euler",
            solver_steps=1,
            solver_rtol=1.0e-5,
            solver_atol=1.0e-5,
            seed=123,
            normalize_peak=False,
        )


def test_dynamic_folder_inference_removes_failed_temp_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = _RecordingCleanPredictor()
    session = _session(tmp_path, model)

    def fake_read(
        audio_path: Path, target_sample_rate: int
    ) -> tuple[torch.Tensor, int]:
        del audio_path
        return torch.randn(2, 37), target_sample_rate

    def fake_write(
        audio_path: Path,
        audio: torch.Tensor,
        sample_rate: int,
        channel_order: list[str] | None = None,
    ) -> None:
        del audio, sample_rate, channel_order
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"partial")
        raise RuntimeError("simulated encoder failure")

    monkeypatch.setattr(offline_batch, "read_audio_channels_first", fake_read)
    monkeypatch.setattr(offline_batch, "write_audio_channels_first", fake_write)

    result = run_dynamic_folder_inference(
        session=session,
        jobs=[
            DynamicInferenceJob(
                input_audio_path=tmp_path / "bad.wav",
                output_audio_path=tmp_path / "out" / "bad.flac",
            )
        ],
        sample_rate=8,
        chunk_seconds=2.0,
        overlap_seconds=0.5,
        solver="midpoint_rk2",
        solver_steps=2,
        solver_rtol=1.0e-5,
        solver_atol=1.0e-5,
        seed=123,
        normalize_peak=False,
        max_batch_size=1,
        max_active_requests=1,
    )

    assert len(result.reports) == 0
    assert len(result.errors) == 1
    assert not (tmp_path / "out" / "bad.flac").exists()
    assert list((tmp_path / "out").glob(".*.tmp-*")) == []
