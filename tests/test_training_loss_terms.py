from __future__ import annotations

import torch
import pytest

from stereo2spatial.training.loss_terms import (
    _binaural_ccf_loss,
    _binaural_cue_loss,
    _binaural_stft_cue_loss,
    _charbonnier_loss,
    _channel_correlation_l1_loss,
    _downmix_consistency_loss,
    _downmix_to_stereo,
    _frame_rms_ild_loss,
    _mid_side_loss,
    _multi_resolution_stft_loss,
    _stereo_log_mel_perceptual_loss,
)


def _manual_weighted_cov(
    channels_by_samples: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Return linearly weighted covariance over samples."""
    norm = weights.sum()
    mean = (channels_by_samples * weights[None, :]).sum(dim=1, keepdim=True) / norm
    centered = channels_by_samples - mean
    weighted = centered * torch.sqrt(weights)[None, :]
    return weighted @ weighted.transpose(0, 1) / norm


def test_channel_correlation_uses_linear_overlap_weighting() -> None:
    # Shape convention is [B,C,D,T]. Use D=1 to make weighted covariance explicit.
    prediction = torch.tensor(
        [[[[0.0, 2.0, 4.0]], [[1.0, 3.0, 5.0]]]],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [[[[1.0, 2.5, 4.5]], [[0.0, 2.0, 6.0]]]],
        dtype=torch.float32,
    )
    mask_dt = torch.tensor([[[1.0, 0.5, 0.25]]], dtype=torch.float32)

    loss = _channel_correlation_l1_loss(
        prediction_x1=prediction,
        target_x1=target,
        mask_dt=mask_dt,
        eps=1e-6,
        offdiag_only=False,
        use_correlation=False,
    )

    pred_cd = prediction[0, :, 0, :]
    tgt_cd = target[0, :, 0, :]
    weights = mask_dt[0, 0, :]
    cov_pred = _manual_weighted_cov(pred_cd, weights)
    cov_tgt = _manual_weighted_cov(tgt_cd, weights)
    expected = (cov_pred - cov_tgt).abs().mean()

    assert torch.allclose(loss, expected, rtol=1e-6, atol=1e-6)


def test_downmix_to_stereo_uses_channel_order_coefficients() -> None:
    signal = torch.zeros(1, 12, 1, 1, dtype=torch.float32)
    signal[:, 0] = 1.0  # FL
    signal[:, 1] = 2.0  # FR
    signal[:, 2] = 4.0  # FC

    stereo = _downmix_to_stereo(signal)

    expected_center = 4.0 / (2.0**0.5)
    assert stereo.shape == (1, 2, 1, 1)
    assert stereo[0, 0, 0, 0].item() == pytest.approx(1.0 + expected_center)
    assert stereo[0, 1, 0, 0].item() == pytest.approx(2.0 + expected_center)


def test_downmix_to_stereo_defaults_six_channels_to_5_1_rear() -> None:
    signal = torch.zeros(1, 6, 1, 1, dtype=torch.float32)
    signal[:, 4] = 1.0  # BL
    signal[:, 5] = 2.0  # BR

    stereo = _downmix_to_stereo(signal)

    surround_gain = 1.0 / (2.0**0.5)
    assert stereo[0, 0, 0, 0].item() == pytest.approx(surround_gain)
    assert stereo[0, 1, 0, 0].item() == pytest.approx(2.0 * surround_gain)


def test_downmix_consistency_loss_compares_against_target_downmix_signal() -> None:
    prediction = torch.zeros(1, 12, 1, 2, dtype=torch.float32)
    prediction[:, 0] = torch.tensor([[[1.0, 2.0]]])
    prediction[:, 1] = torch.tensor([[[3.0, 4.0]]])
    target_downmix = torch.tensor([[[[1.0, 1.0]], [[3.0, 5.0]]]], dtype=torch.float32)
    mask_dt = torch.ones(1, 1, 2, dtype=torch.float32)

    loss = _downmix_consistency_loss(
        prediction_x1=prediction,
        target_downmix_signal=target_downmix,
        mask_dt=mask_dt,
    )

    assert loss.item() == pytest.approx(0.5, abs=1e-7)


def test_charbonnier_loss_matches_smooth_l1_formula() -> None:
    prediction = torch.tensor([0.0, 2.0], dtype=torch.float32)
    target = torch.tensor([0.0, 1.0], dtype=torch.float32)

    loss = _charbonnier_loss(prediction, target, eps=1e-3, reduction="none")

    expected = torch.sqrt(torch.tensor([1e-6, 1.0 + 1e-6])) - 1e-3
    assert torch.allclose(loss, expected, atol=1e-7, rtol=1e-7)


def test_multi_resolution_stft_loss_is_zero_for_matching_waveforms() -> None:
    signal = torch.randn(1, 2, 16, 4)
    mask_dt = torch.ones(1, 16, 4)

    loss = _multi_resolution_stft_loss(
        prediction_x1=signal,
        target_x1=signal.clone(),
        mask_dt=mask_dt,
        fft_sizes=[16, 32],
        hop_lengths=[4, 8],
        win_lengths=[16, 32],
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_multi_resolution_stft_loss_penalizes_spectral_difference() -> None:
    target = torch.zeros(1, 1, 16, 4)
    prediction = target.clone()
    prediction[..., 8:12, 1] = 1.0
    mask_dt = torch.ones(1, 16, 4)

    loss = _multi_resolution_stft_loss(
        prediction_x1=prediction,
        target_x1=target,
        mask_dt=mask_dt,
        fft_sizes=[16],
        hop_lengths=[4],
        win_lengths=[16],
        spectral_convergence_weight=0.0,
        log_magnitude_weight=1.0,
    )

    assert loss.item() > 0.0


def test_binaural_stft_cue_loss_is_zero_for_matching_headphone_audio() -> None:
    signal = torch.randn(1, 2, 16, 4)
    mask_dt = torch.ones(1, 16, 4)

    loss = _binaural_stft_cue_loss(
        prediction_x1=signal,
        target_x1=signal.clone(),
        mask_dt=mask_dt,
        fft_sizes=[16],
        hop_lengths=[4],
        win_lengths=[16],
        ild_weight=0.1,
        ipd_weight=0.1,
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_binaural_stft_cue_loss_penalizes_left_right_cue_difference() -> None:
    time = torch.linspace(0.0, 1.0, 64)
    left = torch.sin(time * 10.0)[None, None, :]
    right = torch.cos(time * 10.0)[None, None, :]
    target = torch.cat([left, right], dim=1).reshape(1, 2, 16, 4)
    prediction = torch.cat([left, left], dim=1).reshape(1, 2, 16, 4)
    mask_dt = torch.ones(1, 16, 4)

    loss = _binaural_stft_cue_loss(
        prediction_x1=prediction,
        target_x1=target,
        mask_dt=mask_dt,
        fft_sizes=[16],
        hop_lengths=[4],
        win_lengths=[16],
        ild_weight=0.1,
        ipd_weight=0.1,
    )

    assert loss.item() > 0.0


def test_binaural_ccf_loss_is_zero_for_matching_headphone_audio() -> None:
    signal = torch.randn(1, 2, 16, 4)
    mask_dt = torch.ones(1, 16, 4)

    loss = _binaural_ccf_loss(
        prediction_x1=signal,
        target_x1=signal.clone(),
        mask_dt=mask_dt,
        sample_rate=1_000,
        frame_ms=16.0,
        hop_ms=8.0,
        max_delay_ms=2.0,
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_frame_rms_ild_loss_penalizes_level_ratio_difference() -> None:
    target = torch.zeros(1, 2, 16, 4)
    target[:, 0] = 1.0
    target[:, 1] = 0.5
    prediction = target.clone()
    prediction[:, 1] = 1.0
    mask_dt = torch.ones(1, 16, 4)

    loss = _frame_rms_ild_loss(
        prediction_x1=prediction,
        target_x1=target,
        mask_dt=mask_dt,
        frame_size=16,
        hop_size=8,
    )

    assert loss.item() > 0.0


def test_mid_side_loss_can_use_l1_side_only() -> None:
    target = torch.zeros(1, 2, 2, 2)
    prediction = target.clone()
    prediction[:, 0] = 1.0
    prediction[:, 1] = -1.0
    mask_dt = torch.ones(1, 2, 2)

    loss = _mid_side_loss(
        prediction_x1=prediction,
        target_x1=target,
        mask_dt=mask_dt,
        loss_type="l1",
        mid_weight=0.0,
        side_weight=1.0,
    )

    assert loss.item() == pytest.approx(1.0, abs=1e-7)


def test_binaural_cue_loss_ignores_non_headphone_targets() -> None:
    signal = torch.randn(1, 6, 16, 4)
    mask_dt = torch.ones(1, 16, 4)

    loss = _binaural_cue_loss(
        prediction_x1=signal,
        target_x1=signal.clone(),
        mask_dt=mask_dt,
        sample_rate=16_000,
        fft_sizes=[16],
        hop_lengths=[4],
        win_lengths=[16],
        ild_weight=0.1,
        ipd_weight=0.1,
        ccf_weight=0.1,
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_stereo_log_mel_perceptual_loss_is_zero_for_matching_stereo() -> None:
    signal = torch.randn(1, 2, 16, 4)
    mask_dt = torch.ones(1, 16, 4)

    loss = _stereo_log_mel_perceptual_loss(
        prediction_x1=signal,
        target_x1=signal.clone(),
        mask_dt=mask_dt,
        sample_rate=16_000,
        n_fft=16,
        hop_length=4,
        win_length=16,
        n_mels=8,
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_stereo_log_mel_perceptual_loss_penalizes_spectral_difference() -> None:
    target = torch.zeros(1, 2, 16, 4)
    prediction = target.clone()
    prediction[:, :, 8:12, 1] = 1.0
    mask_dt = torch.ones(1, 16, 4)

    loss = _stereo_log_mel_perceptual_loss(
        prediction_x1=prediction,
        target_x1=target,
        mask_dt=mask_dt,
        sample_rate=16_000,
        n_fft=16,
        hop_length=4,
        win_length=16,
        n_mels=8,
    )

    assert loss.item() > 0.0


def test_stereo_log_mel_perceptual_loss_uses_target_downmix_for_spatial() -> None:
    prediction = torch.zeros(1, 6, 16, 4)
    target = torch.zeros_like(prediction)
    prediction[:, 0] = 1.0
    target_downmix = torch.zeros(1, 2, 16, 4)
    mask_dt = torch.ones(1, 16, 4)

    loss = _stereo_log_mel_perceptual_loss(
        prediction_x1=prediction,
        target_x1=target,
        target_downmix_signal=target_downmix,
        mask_dt=mask_dt,
        sample_rate=16_000,
        n_fft=16,
        hop_length=4,
        win_length=16,
        n_mels=8,
    )

    assert loss.item() > 0.0
