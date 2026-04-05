from __future__ import annotations

from pathlib import Path

import pytest
import torch

from stereo2spatial.training.checkpointing import (
    _checkpoint_has_ema_state,
    _cleanup_old_checkpoints,
    _find_latest_checkpoint,
    _load_model_weights_only,
    _load_trainer_state,
    _resolve_model_init_checkpoint,
    _resolve_resume_checkpoint,
    _save_trainer_state,
)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)


class _FakeAccelerator:
    def __init__(self, *, is_main_process: bool) -> None:
        self.is_main_process = is_main_process
        self.wait_calls = 0

    def wait_for_everyone(self) -> None:
        self.wait_calls += 1


def _make_checkpoint_dirs(output_dir: Path, steps: list[int]) -> None:
    checkpoint_root = output_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for step in steps:
        (checkpoint_root / f"step_{step:07d}").mkdir(parents=True, exist_ok=True)


def test_save_and_load_trainer_state_round_trip(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "step_0000001"
    checkpoint_dir.mkdir(parents=True)

    _save_trainer_state(
        checkpoint_dir=checkpoint_dir,
        global_step=42,
        epoch=3,
        batches_seen_in_epoch=7,
    )

    loaded = _load_trainer_state(checkpoint_dir)
    assert loaded == (42, 3, 7)


def test_load_trainer_state_requires_json_file(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "step_0000001"
    checkpoint_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="trainer_state.json"):
        _load_trainer_state(checkpoint_dir)


def test_find_latest_checkpoint_returns_highest_step_directory(tmp_path: Path) -> None:
    _make_checkpoint_dirs(tmp_path, [2, 12, 3])

    latest = _find_latest_checkpoint(tmp_path)

    assert latest is not None
    assert latest.name == "step_0000012"


def test_resolve_resume_checkpoint_with_latest_uses_most_recent(tmp_path: Path) -> None:
    _make_checkpoint_dirs(tmp_path, [9, 10])

    resolved = _resolve_resume_checkpoint("latest", tmp_path)

    assert resolved is not None
    assert resolved.name == "step_0000010"


def test_resolve_resume_checkpoint_latest_errors_when_none_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="latest"):
        _resolve_resume_checkpoint("latest", tmp_path)


def test_resolve_model_init_checkpoint_handles_explicit_and_empty_values(
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "model.pt"
    torch.save({"value": 1}, explicit)

    assert _resolve_model_init_checkpoint(str(explicit), tmp_path) == explicit
    assert _resolve_model_init_checkpoint("   ", tmp_path) is None

    with pytest.raises(FileNotFoundError, match="Init checkpoint not found"):
        _resolve_model_init_checkpoint(str(tmp_path / "missing.pt"), tmp_path)


def test_cleanup_old_checkpoints_prunes_oldest_on_main_process(tmp_path: Path) -> None:
    _make_checkpoint_dirs(tmp_path, [1, 2, 3])
    accelerator = _FakeAccelerator(is_main_process=True)

    _cleanup_old_checkpoints(
        output_dir=tmp_path,
        max_to_keep=2,
        accelerator=accelerator,
    )

    checkpoint_root = tmp_path / "checkpoints"
    remaining = sorted(path.name for path in checkpoint_root.glob("step_*"))
    assert remaining == ["step_0000002", "step_0000003"]
    assert accelerator.wait_calls == 1


def test_cleanup_old_checkpoints_non_main_process_only_waits(tmp_path: Path) -> None:
    _make_checkpoint_dirs(tmp_path, [1, 2, 3])
    accelerator = _FakeAccelerator(is_main_process=False)

    _cleanup_old_checkpoints(
        output_dir=tmp_path,
        max_to_keep=1,
        accelerator=accelerator,
    )

    checkpoint_root = tmp_path / "checkpoints"
    remaining = sorted(path.name for path in checkpoint_root.glob("step_*"))
    assert remaining == ["step_0000001", "step_0000002", "step_0000003"]
    assert accelerator.wait_calls == 1


def test_load_model_weights_only_supports_wrapped_and_prefixed_state_dict(
    tmp_path: Path,
) -> None:
    source_model = _TinyModel()
    for parameter in source_model.parameters():
        parameter.data.normal_(mean=0.0, std=0.1)

    payload = {
        "model_state_dict": {
            f"module.{name}": tensor.clone()
            for name, tensor in source_model.state_dict().items()
        }
    }
    checkpoint_path = tmp_path / "weights.pt"
    torch.save(payload, checkpoint_path)

    target_model = _TinyModel()
    for parameter in target_model.parameters():
        parameter.data.zero_()

    _load_model_weights_only(target_model, checkpoint_path)

    for name, tensor in source_model.state_dict().items():
        assert torch.allclose(target_model.state_dict()[name], tensor)


def test_load_model_weights_only_rejects_non_mapping_payload(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "bad.pt"
    torch.save([1, 2, 3], checkpoint_path)

    with pytest.raises(TypeError, match="Unsupported checkpoint payload type"):
        _load_model_weights_only(_TinyModel(), checkpoint_path)


def test_checkpoint_has_ema_state_detects_valid_custom_payload(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "step_0000001"
    checkpoint_dir.mkdir(parents=True)
    torch.save(
        {"decay": 0.999, "model": _TinyModel().state_dict()},
        checkpoint_dir / "custom_checkpoint_0.pkl",
    )

    assert _checkpoint_has_ema_state(checkpoint_dir) is True


def test_checkpoint_has_ema_state_ignores_missing_or_invalid_payloads(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "step_0000001"
    checkpoint_dir.mkdir(parents=True)

    assert _checkpoint_has_ema_state(checkpoint_dir) is False

    torch.save(
        {"decay": 0.999, "foo": "bar"},
        checkpoint_dir / "custom_checkpoint_0.pkl",
    )
    assert _checkpoint_has_ema_state(checkpoint_dir) is False

    torch.save(
        {"model": _TinyModel().state_dict()},
        checkpoint_dir / "not_custom_checkpoint.pkl",
    )
    assert _checkpoint_has_ema_state(checkpoint_dir) is False
