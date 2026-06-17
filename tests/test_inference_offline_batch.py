from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

import stereo2spatial.inference.offline_batch as offline_batch
from stereo2spatial.inference.offline_batch import (
    DynamicInferenceJob,
    WindowedInferenceRequest,
    run_dynamic_folder_inference,
    run_windowed_inference_requests,
)
from stereo2spatial.inference.runner import InferenceSession


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
    )

    assert result.stats.completed_requests == 2
    assert result.stats.scheduler.max_observed_batch_size == 2
    by_id = {item.request_id: item for item in result.results}
    assert by_id["a"].pred_signal.shape == (2, 3, 19)
    assert by_id["a"].window_count == 3
    assert by_id["b"].pred_signal.shape == (2, 3, 12)
    assert by_id["b"].window_count == 2
    assert any(batch.numel() == 2 and batch.unique().numel() == 2 for batch in model.t_batches)


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
    )

    assert result.stats.completed_requests == 3
    assert result.stats.scheduler.max_observed_active_controllers == 2
    assert result.stats.scheduler.max_observed_batch_size == 2


def test_dynamic_folder_inference_runs_jobs_with_batched_model_queries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = _RecordingCleanPredictor()
    session = _session(tmp_path, model)
    writes: list[tuple[Path, torch.Tensor, int]] = []

    def fake_read(audio_path: Path, target_sample_rate: int) -> tuple[torch.Tensor, int]:
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
    assert (tmp_path / "out" / "a.flac").exists()
    assert (tmp_path / "out" / "b.flac").exists()
    assert all(path.name.startswith(".") and ".tmp-" in path.name for path, _, _ in writes)
    assert result.stats.scheduler.max_observed_batch_size == 2
    assert result.stats.scheduler.completed_controllers == 6
    assert writes[0][1].shape == (2, 37)


def test_dynamic_folder_inference_removes_failed_temp_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = _RecordingCleanPredictor()
    session = _session(tmp_path, model)

    def fake_read(audio_path: Path, target_sample_rate: int) -> tuple[torch.Tensor, int]:
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
