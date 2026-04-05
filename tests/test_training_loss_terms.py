from __future__ import annotations

import torch

from stereo2spatial.training.loss_terms import _channel_correlation_l1_loss


def _manual_weighted_cov(channels_by_samples: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
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
