"""Shared loss-term primitives for flow-matching training."""

from __future__ import annotations

import torch


def _compute_loss_weighted(
    prediction: torch.Tensor,  # [B,C,D,T]
    target_velocity: torch.Tensor,  # [B,C,D,T]
    valid_mask: torch.Tensor,  # [B,T]
    frame_weight: torch.Tensor,  # [T]
) -> torch.Tensor:
    """Compute frame-weighted masked MSE for flow-matching velocity targets."""
    mse = (prediction - target_velocity).pow(2)  # [B,C,D,T]
    w = frame_weight[None, :].to(
        dtype=prediction.dtype, device=prediction.device
    )  # [1,T]
    m = valid_mask.to(dtype=prediction.dtype, device=prediction.device)  # [B,T]
    wm = (w * m)[:, None, None, :]  # [B,1,1,T]

    weighted = mse * wm
    denom = wm.sum() * prediction.shape[1] * prediction.shape[2]
    denom = torch.clamp(denom, min=1.0)
    return weighted.sum() / denom


def _channel_routing_kl_loss(
    prediction_x1: torch.Tensor,  # [B,C,D,T]
    target_x1: torch.Tensor,  # [B,C,D,T]
    mask_dt: torch.Tensor,  # [B,D,T]  (valid_mask * overlap_weight)
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Encourage per-(D,T) energy distribution across channels to match target.

    Uses:
    - p(c|d,t) from target energy
    - q(c|d,t) from prediction energy
    - KL(p || q), averaged with mask_dt.
    """
    pred_sq = prediction_x1.float().pow(2)
    tgt_sq = target_x1.float().pow(2)

    tau = max(float(temperature), 1e-6)
    eps_f = max(float(eps), 1e-12)

    tgt_logits = torch.log(tgt_sq + eps_f) / tau  # [B,C,D,T]
    pred_logits = torch.log(pred_sq + eps_f) / tau

    p = torch.softmax(tgt_logits, dim=1)
    log_p = torch.log_softmax(tgt_logits, dim=1)
    log_q = torch.log_softmax(pred_logits, dim=1)

    kl_dt = torch.sum(p * (log_p - log_q), dim=1)  # [B,D,T]

    w = mask_dt.to(dtype=kl_dt.dtype, device=kl_dt.device)
    denom = torch.clamp(w.sum(), min=1.0)
    return (kl_dt * w).sum() / denom


def _channel_correlation_l1_loss(
    prediction_x1: torch.Tensor,  # [B,C,D,T]
    target_x1: torch.Tensor,  # [B,C,D,T]
    mask_dt: torch.Tensor,  # [B,D,T]  (valid_mask * overlap_weight)
    eps: float = 1e-6,
    offdiag_only: bool = True,
    use_correlation: bool = True,
) -> torch.Tensor:
    """
    Match cross-channel structure between prediction and target.

    Default behavior compares off-diagonal entries of correlation matrices.
    """
    pred = prediction_x1.float()
    tgt = target_x1.float()

    mask = mask_dt[:, None, :, :].to(dtype=pred.dtype, device=pred.device)
    n = torch.clamp(
        mask.sum(dim=(2, 3), keepdim=True), min=max(float(eps), 1e-6)
    )  # [B,1,1,1]

    pred_mean = (pred * mask).sum(dim=(2, 3), keepdim=True) / n
    tgt_mean = (tgt * mask).sum(dim=(2, 3), keepdim=True) / n

    # Use sqrt(weights) so covariance is weighted linearly by mask_dt.
    sqrt_mask = torch.sqrt(mask.clamp_min(0.0))
    pred_centered = (pred - pred_mean) * sqrt_mask
    tgt_centered = (tgt - tgt_mean) * sqrt_mask

    pred_flat = pred_centered.reshape(pred.shape[0], pred.shape[1], -1)  # [B,C,N]
    tgt_flat = tgt_centered.reshape(tgt.shape[0], tgt.shape[1], -1)  # [B,C,N]
    n_scalar = n.reshape(n.shape[0], 1, 1)  # [B,1,1]

    cov_pred = torch.bmm(pred_flat, pred_flat.transpose(1, 2)) / n_scalar  # [B,C,C]
    cov_tgt = torch.bmm(tgt_flat, tgt_flat.transpose(1, 2)) / n_scalar  # [B,C,C]

    if use_correlation:
        diag_pred = torch.diagonal(cov_pred, dim1=1, dim2=2).clamp_min(eps)  # [B,C]
        diag_tgt = torch.diagonal(cov_tgt, dim1=1, dim2=2).clamp_min(eps)  # [B,C]

        std_pred = torch.sqrt(diag_pred)  # [B,C]
        std_tgt = torch.sqrt(diag_tgt)  # [B,C]

        denom_pred = std_pred[:, :, None] * std_pred[:, None, :]
        denom_tgt = std_tgt[:, :, None] * std_tgt[:, None, :]

        corr_pred = cov_pred / denom_pred.clamp_min(eps)
        corr_tgt = cov_tgt / denom_tgt.clamp_min(eps)
        mat_pred, mat_tgt = corr_pred, corr_tgt
    else:
        mat_pred, mat_tgt = cov_pred, cov_tgt

    diff = (mat_pred - mat_tgt).abs()  # [B,C,C]
    if offdiag_only:
        c = diff.shape[1]
        eye = torch.eye(c, device=diff.device, dtype=diff.dtype)[None, :, :]
        diff = diff * (1.0 - eye)
        denom = float(max(c * (c - 1), 1))
        return diff.sum() / denom / diff.shape[0]

    return diff.mean()
