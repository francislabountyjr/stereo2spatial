from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from safetensors.torch import save_file as save_safetensors_file

from stereo2spatial.inference.checkpoint import (
    load_model_weights,
    resolve_checkpoint_path,
)
from stereo2spatial.inference.sampling import (
    generate_spatial_latent,
    resolve_chunk_frames,
)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)


class _DummySamplingModel(torch.nn.Module):
    target_channels = 2
    latent_dim = 3

    def init_memory(
        self, *, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.zeros((batch_size, 1, 1), device=device, dtype=dtype)

    def forward(
        self,
        *,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor,
        mem: torch.Tensor | None = None,
        return_mem: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del t, z_cond, valid_mask
        velocity = torch.full_like(zt, 0.1)
        if return_mem:
            if mem is None:
                raise RuntimeError("mem is required when return_mem=True")
            return velocity, mem + 1.0
        return velocity


class _StateAwareSamplingModel(torch.nn.Module):
    target_channels = 2
    latent_dim = 3

    def init_memory(
        self, *, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.zeros((batch_size, 1, 1), device=device, dtype=dtype)

    def forward(
        self,
        *,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor,
        mem: torch.Tensor | None = None,
        return_mem: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del z_cond, valid_mask
        velocity = zt + t[:, None, None, None].to(dtype=zt.dtype, device=zt.device)
        if return_mem:
            if mem is None:
                raise RuntimeError("mem is required when return_mem=True")
            return velocity, mem + 1.0
        return velocity


def _make_checkpoint_dirs(output_dir: Path, steps: list[int]) -> None:
    checkpoint_root = output_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for step in steps:
        (checkpoint_root / f"step_{step:07d}").mkdir(parents=True, exist_ok=True)


def test_resolve_chunk_frames_clamps_overlap_and_handles_short_sequences() -> None:
    chunk_frames, overlap_frames = resolve_chunk_frames(
        cond_latent_frames=200,
        latent_fps=50.0,
        chunk_seconds=1.0,
        overlap_seconds=2.0,
    )
    assert (chunk_frames, overlap_frames) == (50, 49)

    short_chunk_frames, short_overlap_frames = resolve_chunk_frames(
        cond_latent_frames=30,
        latent_fps=50.0,
        chunk_seconds=1.0,
        overlap_seconds=0.5,
    )
    assert (short_chunk_frames, short_overlap_frames) == (30, 0)


def test_generate_spatial_latent_is_deterministic_for_same_seed() -> None:
    model = _DummySamplingModel()
    cond_latent = torch.randn(1, 3, 9)

    kwargs = {
        "model": cast(Any, model),
        "cond_latent": cond_latent,
        "chunk_frames": 4,
        "overlap_frames": 1,
        "solver": "heun",
        "solver_steps": 2,
        "solver_rtol": 1e-5,
        "solver_atol": 1e-5,
    }
    pred_a = generate_spatial_latent(seed=123, **kwargs)
    pred_b = generate_spatial_latent(seed=123, **kwargs)
    pred_c = generate_spatial_latent(seed=999, **kwargs)

    assert pred_a.shape == (2, 3, 9)
    assert torch.allclose(pred_a, pred_b)
    assert not torch.allclose(pred_a, pred_c)


def test_generate_spatial_latent_validates_cond_latent_rank() -> None:
    model = _DummySamplingModel()
    with pytest.raises(ValueError, match="cond_latent must be \\[C,D,T\\]"):
        generate_spatial_latent(
            model=cast(Any, model),
            cond_latent=torch.randn(3, 12),
            chunk_frames=4,
            overlap_frames=1,
            solver="heun",
            solver_steps=2,
            solver_rtol=1e-5,
            solver_atol=1e-5,
            seed=0,
        )


def test_generate_spatial_latent_supports_unipc_solver() -> None:
    model = _StateAwareSamplingModel()
    cond_latent = torch.randn(1, 3, 9)

    pred_unipc = generate_spatial_latent(
        model=cast(Any, model),
        cond_latent=cond_latent,
        chunk_frames=4,
        overlap_frames=1,
        solver="unipc",
        solver_steps=4,
        solver_rtol=1e-5,
        solver_atol=1e-5,
        seed=123,
    )
    pred_heun = generate_spatial_latent(
        model=cast(Any, model),
        cond_latent=cond_latent,
        chunk_frames=4,
        overlap_frames=1,
        solver="heun",
        solver_steps=4,
        solver_rtol=1e-5,
        solver_atol=1e-5,
        seed=123,
    )

    assert pred_unipc.shape == (2, 3, 9)
    assert not torch.allclose(pred_unipc, pred_heun)


def test_generate_spatial_latent_threads_memory_between_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _DummySamplingModel()
    cond_latent = torch.randn(1, 3, 8)
    seen_mem_values: list[float] = []

    def _fake_sample_chunk_latent(
        model: Any,
        cond_chunk: torch.Tensor,
        valid_mask: torch.Tensor,
        z0_chunk: torch.Tensor,
        solver: str,
        solver_steps: int,
        solver_rtol: float,
        solver_atol: float,
        mem: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del model, cond_chunk, valid_mask, solver, solver_steps, solver_rtol, solver_atol
        if mem is not None:
            seen_mem_values.append(float(mem.mean().item()))
            return z0_chunk, mem + 1.0
        return z0_chunk, None

    monkeypatch.setattr(
        "stereo2spatial.inference.sampling._sample_chunk_latent",
        _fake_sample_chunk_latent,
    )

    pred = generate_spatial_latent(
        model=cast(Any, model),
        cond_latent=cond_latent,
        chunk_frames=4,
        overlap_frames=0,
        solver="euler",
        solver_steps=2,
        solver_rtol=1e-5,
        solver_atol=1e-5,
        seed=123,
    )

    assert pred.shape == (2, 3, 8)
    assert len(seen_mem_values) == 2
    assert seen_mem_values[0] == pytest.approx(0.0)
    assert seen_mem_values[1] == pytest.approx(1.0)


def test_resolve_checkpoint_path_supports_latest_and_validates_inputs(
    tmp_path: Path,
) -> None:
    _make_checkpoint_dirs(tmp_path, [2, 10, 3])

    latest = resolve_checkpoint_path(checkpoint="latest", output_dir=tmp_path)
    assert latest.name == "step_0000010"

    with pytest.raises(ValueError, match="checkpoint cannot be empty"):
        resolve_checkpoint_path(checkpoint="   ", output_dir=tmp_path)

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        resolve_checkpoint_path(
            checkpoint=tmp_path / "missing.pt",
            output_dir=tmp_path,
        )


def test_load_model_weights_supports_wrapped_model_state_dict(tmp_path: Path) -> None:
    source_model = _TinyModel()
    for parameter in source_model.parameters():
        parameter.data.uniform_(-0.5, 0.5)

    checkpoint_path = tmp_path / "weights.pt"
    torch.save(
        {"model_state_dict": source_model.state_dict()},
        checkpoint_path,
    )

    target_model = _TinyModel()
    for parameter in target_model.parameters():
        parameter.data.zero_()

    source = load_model_weights(target_model, checkpoint_path)
    assert source == "student"

    for name, tensor in source_model.state_dict().items():
        assert torch.allclose(target_model.state_dict()[name], tensor)


def test_load_model_weights_supports_safetensors_file(tmp_path: Path) -> None:
    source_model = _TinyModel()
    for parameter in source_model.parameters():
        parameter.data.uniform_(-0.5, 0.5)

    checkpoint_path = tmp_path / "model.safetensors"
    save_safetensors_file(source_model.state_dict(), str(checkpoint_path))

    target_model = _TinyModel()
    for parameter in target_model.parameters():
        parameter.data.zero_()

    source = load_model_weights(target_model, checkpoint_path)
    assert source == "student"

    for name, tensor in source_model.state_dict().items():
        assert torch.allclose(target_model.state_dict()[name], tensor)


def test_load_model_weights_rejects_unsupported_payload_type(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "bad.pt"
    torch.save(["not", "a", "dict"], checkpoint_path)

    with pytest.raises(TypeError, match="Unsupported checkpoint payload type"):
        load_model_weights(_TinyModel(), checkpoint_path)


def test_load_model_weights_prefers_ema_when_available_in_checkpoint_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    student_model = _TinyModel()
    ema_model = _TinyModel()
    for parameter in student_model.parameters():
        parameter.data.zero_()
    for parameter in ema_model.parameters():
        parameter.data.fill_(1.0)

    checkpoint_dir = tmp_path / "step_0000001"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"decay": 0.999, "model": ema_model.state_dict()},
        checkpoint_dir / "custom_checkpoint_0.pkl",
    )
    monkeypatch.setattr(
        "stereo2spatial.inference.checkpoint._load_state_dict_from_checkpoint_dir",
        lambda _path: student_model.state_dict(),
    )

    target_model = _TinyModel()
    source = load_model_weights(target_model, checkpoint_dir, weights_source="auto")

    assert source == "ema"
    for name, tensor in ema_model.state_dict().items():
        assert torch.allclose(target_model.state_dict()[name], tensor)


def test_load_model_weights_can_force_student_from_checkpoint_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    student_model = _TinyModel()
    ema_model = _TinyModel()
    for parameter in student_model.parameters():
        parameter.data.fill_(0.25)
    for parameter in ema_model.parameters():
        parameter.data.fill_(0.75)

    checkpoint_dir = tmp_path / "step_0000002"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"decay": 0.999, "model": ema_model.state_dict()},
        checkpoint_dir / "custom_checkpoint_0.pkl",
    )
    monkeypatch.setattr(
        "stereo2spatial.inference.checkpoint._load_state_dict_from_checkpoint_dir",
        lambda _path: student_model.state_dict(),
    )

    target_model = _TinyModel()
    source = load_model_weights(target_model, checkpoint_dir, weights_source="student")

    assert source == "student"
    for name, tensor in student_model.state_dict().items():
        assert torch.allclose(target_model.state_dict()[name], tensor)


def test_load_model_weights_ema_requested_raises_when_missing_in_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    student_model = _TinyModel()
    checkpoint_dir = tmp_path / "step_0000003"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "stereo2spatial.inference.checkpoint._load_state_dict_from_checkpoint_dir",
        lambda _path: student_model.state_dict(),
    )

    with pytest.raises(FileNotFoundError, match="EMA"):
        load_model_weights(_TinyModel(), checkpoint_dir, weights_source="ema")
