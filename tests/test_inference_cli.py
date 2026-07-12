from __future__ import annotations

import pytest

from stereo2spatial.cli import infer


def test_infer_cli_defaults_to_timestep_major_sampling() -> None:
    parser = infer.build_parser()

    assert parser.parse_args([]).sampling_order == "timestep-major"
    assert (
        parser.parse_args(["--sampling-order", "window-major"]).sampling_order
        == "window-major"
    )


def test_infer_cli_lists_mix_style_presets(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["infer", "--list-mix-style-presets"],
    )

    infer.main()

    captured = capsys.readouterr()
    assert "Mix-style presets:" in captured.out
    assert "balanced:" in captured.out
    assert "front_focus:" in captured.out
    assert "ambient_wide:" in captured.out


def test_infer_cli_requires_runtime_args_unless_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["infer"])

    with pytest.raises(SystemExit, match="--checkpoint"):
        infer.main()
