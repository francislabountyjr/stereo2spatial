"""Shared helpers for windowed flow-matching loss paths."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from accelerate import Accelerator
from torch.distributions import Beta

from .loss_terms import _masked_waveform_reconstruction_loss, _masked_x_pred_v_loss
from .windowing import _chunk_weight, _segment_starts

WindowMetadata = dict[int, tuple[list[int], list[torch.Tensor]]]
_ALLOWED_FLOW_TIMESTEP_SAMPLERS = {"uniform", "logit_normal", "beta", "custom"}
_ALLOWED_FLOW_LOSS_WEIGHTING = {"none", "sigma_sqrt", "cosmap"}
_FLOW_TIMESTEP_DENOM = 1000.0


@dataclass(frozen=True)
class FlowMatchingInputs:
    """Normalized signal tensors and sampled flow-matching state."""

    z1: torch.Tensor
    z_cond: torch.Tensor
    valid_mask: torch.Tensor
    t: torch.Tensor
    sigma: torch.Tensor
    loss_weight: torch.Tensor
    z0: torch.Tensor
    zt: torch.Tensor
    batch_size: int
    t_eff: int


def _resolve_flow_schedule_shift(
    *,
    training_config: object | None,
    sequence_length: int,
) -> float | None:
    """Resolve static or auto flow schedule-shift value."""
    if training_config is None:
        return None

    static_shift = getattr(training_config, "flow_schedule_shift", None)
    if static_shift is not None:
        shift = float(static_shift)
        if shift > 0.0:
            return shift

    if not bool(getattr(training_config, "flow_schedule_auto_shift", False)):
        return None

    seq_len = max(1, int(sequence_length))
    base_seq_len = max(1, int(getattr(training_config, "flow_schedule_base_seq_len", 256)))
    max_seq_len = max(
        base_seq_len + 1,
        int(getattr(training_config, "flow_schedule_max_seq_len", 4096)),
    )
    base_shift = float(getattr(training_config, "flow_schedule_base_shift", 0.5))
    max_shift = float(getattr(training_config, "flow_schedule_max_shift", 1.15))

    slope = (max_shift - base_shift) / float(max_seq_len - base_seq_len)
    intercept = base_shift - slope * float(base_seq_len)
    mu = float(seq_len) * slope + intercept
    return float(math.exp(mu))


def _apply_flow_schedule_shift(sigmas: torch.Tensor, shift: float | None) -> torch.Tensor:
    """Apply SD3-style nonlinear shift remapping to sigma values."""
    if shift is None:
        return sigmas
    shift_t = torch.as_tensor(shift, device=sigmas.device, dtype=sigmas.dtype)
    shifted = (sigmas * shift_t) / (1.0 + (shift_t - 1.0) * sigmas)
    return shifted.clamp(0.0, 1.0)


def _parse_custom_sigmas(
    *,
    custom_timesteps: list[float] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Parse user custom sigma/timestep list into sigma space [0, 1]."""
    if custom_timesteps is None:
        return None
    if len(custom_timesteps) <= 0:
        raise ValueError("flow_custom_timesteps cannot be empty when provided.")
    values = torch.tensor(custom_timesteps, device=device, dtype=dtype).flatten()
    if values.numel() <= 0:
        raise ValueError("flow_custom_timesteps cannot be empty when provided.")
    if bool((torch.max(values) <= 1.0).item()):
        return values.clamp(0.0, 1.0)
    return (values / _FLOW_TIMESTEP_DENOM).clamp(0.0, 1.0)


def _sample_flow_sigmas(
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    sequence_length: int,
    training_config: object | None,
) -> torch.Tensor:
    """Sample sigma values for flow-matching signal interpolation."""
    sampler = "uniform"
    use_fast_schedule = False
    if training_config is not None:
        sampler = str(getattr(training_config, "flow_timestep_sampling", "uniform")).strip().lower()
        use_fast_schedule = bool(getattr(training_config, "flow_fast_schedule", False))
    if sampler not in _ALLOWED_FLOW_TIMESTEP_SAMPLERS:
        sampler = "uniform"

    if use_fast_schedule:
        available_sigmas = torch.tensor(
            [1.0] * 7 + [0.75, 0.5, 0.25],
            device=device,
            dtype=dtype,
        )
        indices = torch.randint(
            low=0,
            high=int(available_sigmas.shape[0]),
            size=(batch_size,),
            device=device,
        )
        sigmas = available_sigmas[indices]
    elif sampler == "logit_normal":
        mean = float(getattr(training_config, "flow_logit_mean", 0.0)) if training_config is not None else 0.0
        std = float(getattr(training_config, "flow_logit_std", 1.0)) if training_config is not None else 1.0
        sigmas = torch.sigmoid(
            torch.normal(
                mean=mean,
                std=std,
                size=(batch_size,),
                device=device,
                dtype=torch.float32,
            )
        ).to(dtype=dtype)
    elif sampler == "beta":
        alpha = float(getattr(training_config, "flow_beta_alpha", 1.0)) if training_config is not None else 1.0
        beta = float(getattr(training_config, "flow_beta_beta", 1.0)) if training_config is not None else 1.0
        distribution = Beta(alpha, beta)
        sigmas = distribution.sample((batch_size,)).to(device=device, dtype=dtype)
    elif sampler == "custom":
        custom = _parse_custom_sigmas(
            custom_timesteps=(
                getattr(training_config, "flow_custom_timesteps", None)
                if training_config is not None
                else None
            ),
            device=device,
            dtype=dtype,
        )
        if custom is None:
            raise ValueError(
                "flow_timestep_sampling=custom requires training.flow_custom_timesteps."
            )
        if int(custom.numel()) == 1:
            sigmas = custom.expand(batch_size)
        else:
            indices = torch.randint(
                low=0,
                high=int(custom.shape[0]),
                size=(batch_size,),
                device=device,
            )
            sigmas = custom[indices]
    else:
        sigmas = torch.rand((batch_size,), device=device, dtype=dtype)

    shift = _resolve_flow_schedule_shift(
        training_config=training_config,
        sequence_length=sequence_length,
    )
    return _apply_flow_schedule_shift(sigmas, shift)


def _compute_sd3_style_flow_loss_weight(
    *,
    sigmas: torch.Tensor,
    weighting_scheme: str,
) -> torch.Tensor:
    """Compute optional SD3-style FM loss weights from sigma."""
    scheme = str(weighting_scheme).strip().lower()
    if scheme not in _ALLOWED_FLOW_LOSS_WEIGHTING:
        scheme = "none"
    if scheme == "sigma_sqrt":
        return sigmas.clamp_min(1e-6).pow(-2.0).float()
    if scheme == "cosmap":
        denom = (1.0 - 2.0 * sigmas + 2.0 * sigmas.pow(2)).clamp_min(1e-6)
        return (2.0 / (math.pi * denom)).float()
    return torch.ones_like(sigmas, dtype=torch.float32)


def prepare_flow_matching_inputs(
    *,
    z1: torch.Tensor,
    z_cond: torch.Tensor,
    valid_mask: torch.Tensor,
    t_eff: int,
    training_config: object | None = None,
) -> FlowMatchingInputs:
    """Slice to ``t_eff``, mask invalid frames, and sample diffusion state."""
    t_eff = int(t_eff)
    z1 = z1[..., :t_eff]
    z_cond = z_cond[..., :t_eff]
    valid_mask = valid_mask[:, :t_eff]

    frame_mask = valid_mask[:, None, None, :].to(dtype=z1.dtype)
    z1 = z1 * frame_mask
    z_cond = z_cond * frame_mask

    batch_size = z1.shape[0]
    one_step = bool(getattr(training_config, "flow_one_step", False)) if training_config is not None else False
    if one_step:
        sigma = torch.zeros((batch_size,), device=z1.device, dtype=torch.float32)
        t = torch.ones((batch_size,), device=z1.device, dtype=z1.dtype)
        z0 = torch.zeros_like(z1)
        one_step_input = str(getattr(training_config, "flow_one_step_input", "zeros")).strip().lower() if training_config is not None else "zeros"
        zt = z_cond.clone() if one_step_input == "cond" else torch.zeros_like(z1)
    else:
        sigma = _sample_flow_sigmas(
            batch_size=batch_size,
            device=z1.device,
            dtype=torch.float32,
            sequence_length=t_eff,
            training_config=training_config,
        )
        t = (1.0 - sigma).clamp(0.0, 1.0)
        z0 = torch.randn_like(z1)
        zt = (1.0 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1
    loss_weight = _compute_sd3_style_flow_loss_weight(
        sigmas=sigma.to(dtype=torch.float32),
        weighting_scheme=(
            getattr(training_config, "flow_loss_weighting", "none")
            if training_config is not None
            else "none"
        ),
    ).to(device=z1.device)
    return FlowMatchingInputs(
        z1=z1,
        z_cond=z_cond,
        valid_mask=valid_mask,
        t=t,
        sigma=sigma,
        loss_weight=loss_weight,
        z0=z0,
        zt=zt,
        batch_size=batch_size,
        t_eff=t_eff,
    )


def init_memory_if_enabled(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Initialize model memory tokens when the unwrapped model supports them."""
    raw_model = accelerator.unwrap_model(model)
    if getattr(raw_model, "num_memory_tokens", 0) <= 0:
        return None
    return cast(
        torch.Tensor | None,
        raw_model.init_memory(batch_size=batch_size, device=device, dtype=dtype),
    )


def resolve_window_plan(
    *,
    t_eff: int,
    window_frames: int,
    overlap_frames: int,
    window_metadata: WindowMetadata | None = None,
) -> tuple[list[int], list[torch.Tensor] | None]:
    """Resolve segment starts and optional cached frame weights."""
    cached = window_metadata.get(t_eff) if window_metadata is not None else None
    if cached is None:
        stride_frames = window_frames - overlap_frames
        starts = _segment_starts(
            total_frames=t_eff,
            window_frames=window_frames,
            stride_frames=stride_frames,
        )
        return starts, None
    starts, cached_weights = cached
    return starts, cached_weights


def resolve_window_weight(
    *,
    idx: int,
    num_windows: int,
    window_frames: int,
    overlap_frames: int,
    cached_weights: list[torch.Tensor] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return overlap weighting for one window, using cache when available."""
    if cached_weights is None:
        return _chunk_weight(
            chunk_length=window_frames,
            overlap_frames=overlap_frames,
            is_first=(idx == 0),
            is_last=(idx == num_windows - 1),
            device=device,
            dtype=dtype,
        )
    return cached_weights[idx].to(device=device, dtype=dtype)


def slice_and_pad_window(
    *,
    zt: torch.Tensor,
    z_cond: torch.Tensor,
    valid_mask: torch.Tensor,
    start: int,
    end: int,
    window_frames: int,
    batch_size: int,
    z1: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Slice one temporal window and right-pad to fixed ``window_frames``."""
    segment_len = int(end - start)
    zt_w = zt[..., start:end]
    zc_w = z_cond[..., start:end]
    vm_w = valid_mask[:, start:end]
    z1_w = z1[..., start:end] if z1 is not None else None

    if segment_len >= window_frames:
        return zt_w, zc_w, vm_w, z1_w

    pad_t = window_frames - segment_len
    zt_w = torch.cat([zt_w, torch.zeros_like(zt_w[..., :pad_t])], dim=-1)
    zc_w = torch.cat([zc_w, torch.zeros_like(zc_w[..., :pad_t])], dim=-1)
    if z1_w is not None:
        z1_w = torch.cat([z1_w, torch.zeros_like(z1_w[..., :pad_t])], dim=-1)
    vm_w = torch.cat(
        [
            vm_w,
            torch.zeros((batch_size, pad_t), device=vm_w.device, dtype=vm_w.dtype),
        ],
        dim=1,
    )
    return zt_w, zc_w, vm_w, z1_w


def slice_and_pad_tensor4d(
    *,
    tensor: torch.Tensor,
    start: int,
    end: int,
    window_frames: int,
) -> torch.Tensor:
    """Slice one ``[B,C,D,T]`` tensor window and right-pad to ``window_frames``."""
    segment_len = int(end - start)
    tensor_w = tensor[..., start:end]
    if segment_len >= window_frames:
        return tensor_w

    pad_t = window_frames - segment_len
    return torch.cat([tensor_w, torch.zeros_like(tensor_w[..., :pad_t])], dim=-1)


def compute_flow_matching_window_loss(
    *,
    prediction: torch.Tensor,  # clean x1 prediction [B,C,D,T]
    target_clean: torch.Tensor,  # clean x1 target [B,C,D,T]
    noisy_state: torch.Tensor | None = None,  # current xt state [B,C,D,T]
    t: torch.Tensor | None = None,  # [B]
    valid_mask: torch.Tensor,  # [B,T]
    frame_weight: torch.Tensor,  # [T]
    sample_loss_weight: torch.Tensor | None = None,  # [B]
    waveform_mse_loss_weight: float = 1.0,
    waveform_l1_loss_weight: float = 0.0,
    waveform_charbonnier_loss_weight: float = 0.0,
    waveform_charbonnier_eps: float = 1e-3,
    x_pred_v_loss_weight: float = 0.0,
    loss_metrics: dict[str, torch.Tensor] | None = None,
    reflex_enabled: bool = False,
    reflex_clean_pred: torch.Tensor | None = None,  # [B,C,D,T]
    reflex_biased_pred: torch.Tensor | None = None,  # [B,C,D,T]
    reflex_prediction_vector: torch.Tensor | None = None,  # [B,C,D,T]
    reflex_target_vector: torch.Tensor | None = None,  # [B,C,D,T]
    reflex_alpha: float = 1.0,
    reflex_beta1: float = 10.0,
    reflex_beta2: float = 1.0,
) -> torch.Tensor:
    """
    Compute frame-weighted clean-prediction loss with optional ReflexFlow terms.

    ReflexFlow support:
    - FC weighting from clean-vs-biased prediction difference.
    - Directional anti-drift regularizer using the current biased-state direction.
    """
    pred_f = prediction.float()
    target_f = target_clean.float()
    element_weight: torch.Tensor | float | None = None

    if reflex_enabled:
        if reflex_clean_pred is not None and reflex_biased_pred is not None:
            exposure = (reflex_clean_pred - reflex_biased_pred).detach().to(
                dtype=pred_f.dtype,
                device=pred_f.device,
            )
            norm_dims = tuple(range(1, exposure.dim()))
            exposure_norm = exposure.abs().sum(dim=norm_dims, keepdim=True).clamp_min(1e-6)
            if float(reflex_alpha) != 0.0:
                element_weight = 1.0 + float(reflex_alpha) * exposure / exposure_norm
        if float(reflex_beta2) != 1.0:
            if element_weight is None:
                element_weight = float(reflex_beta2)
            else:
                element_weight = element_weight * float(reflex_beta2)

    if (
        float(waveform_mse_loss_weight)
        + float(waveform_l1_loss_weight)
        + float(waveform_charbonnier_loss_weight)
        + float(x_pred_v_loss_weight)
    ) <= 0.0:
        raise ValueError("At least one waveform/x-prediction loss weight must be > 0")

    if (
        float(waveform_mse_loss_weight)
        + float(waveform_l1_loss_weight)
        + float(waveform_charbonnier_loss_weight)
    ) > 0.0:
        loss = _masked_waveform_reconstruction_loss(
            prediction=prediction,
            target_clean=target_clean,
            valid_mask=valid_mask,
            frame_weight=frame_weight,
            mse_weight=float(waveform_mse_loss_weight),
            l1_weight=float(waveform_l1_loss_weight),
            charbonnier_weight=float(waveform_charbonnier_loss_weight),
            charbonnier_eps=float(waveform_charbonnier_eps),
            sample_loss_weight=sample_loss_weight,
            element_weight=element_weight,
            loss_metrics=loss_metrics,
        )
    else:
        loss = prediction.float().new_zeros(())
    if float(x_pred_v_loss_weight) > 0.0:
        if noisy_state is None or t is None:
            raise ValueError("x_pred_v_loss_weight > 0 requires noisy_state and t")
        loss = loss + _masked_x_pred_v_loss(
            prediction=prediction,
            target_clean=target_clean,
            noisy_state=noisy_state,
            t=t,
            valid_mask=valid_mask,
            frame_weight=frame_weight,
            weight=float(x_pred_v_loss_weight),
            sample_loss_weight=sample_loss_weight,
            element_weight=element_weight,
            loss_metrics=loss_metrics,
        )

    if reflex_enabled and float(reflex_beta1) != 0.0:
        mask4 = valid_mask[:, None, None, :].to(dtype=pred_f.dtype, device=pred_f.device)
        adr_target = target_f
        if reflex_target_vector is not None:
            adr_target = reflex_target_vector.float().to(device=pred_f.device)
        adr_pred = pred_f
        if reflex_prediction_vector is not None:
            adr_pred = reflex_prediction_vector.float().to(device=pred_f.device)
        flat_target = (adr_target * mask4).reshape(adr_target.shape[0], -1)
        flat_pred = (adr_pred * mask4).reshape(adr_pred.shape[0], -1)
        target_norm = torch.norm(flat_target, dim=1, keepdim=True).clamp_min(1e-6)
        pred_norm = torch.norm(flat_pred, dim=1, keepdim=True).clamp_min(1e-6)
        target_dir = flat_target / target_norm
        pred_dir = flat_pred / pred_norm
        adr = (pred_dir - target_dir).pow(2).sum(dim=1).mean()
        loss = loss + float(reflex_beta1) * adr

    return loss


def apply_mix_style_dropout(
    *,
    mix_style: torch.Tensor | None,
    training_config: object | None,
    model: torch.nn.Module,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Randomly mark per-sample style conditioning as absent during training."""
    if mix_style is None:
        return None, None
    probability = (
        float(getattr(training_config, "mix_style_dropout_probability", 0.0) or 0.0)
        if training_config is not None
        else 0.0
    )
    if probability <= 0.0 or not bool(getattr(model, "training", False)):
        return mix_style, None
    probability = min(1.0, max(0.0, probability))
    keep_mask = torch.rand(
        (mix_style.shape[0],),
        device=mix_style.device,
        dtype=torch.float32,
    ) >= probability
    return mix_style, keep_mask


def forward_window(
    *,
    model: torch.nn.Module,
    zt_w: torch.Tensor,
    zc_w: torch.Tensor,
    vm_w: torch.Tensor,
    t: torch.Tensor,
    mem: torch.Tensor | None,
    detach_memory: bool,
    mix_style: torch.Tensor | None = None,
    mix_style_mask: torch.Tensor | None = None,
    amplitude_gain: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run one model forward with optional recurrent memory handling."""
    if mem is None:
        kwargs: dict[str, object] = {
            "zt": zt_w,
            "t": t,
            "z_cond": zc_w,
            "valid_mask": vm_w,
        }
        if mix_style is not None:
            kwargs["mix_style"] = mix_style
        if mix_style_mask is not None:
            kwargs["mix_style_mask"] = mix_style_mask
        if amplitude_gain is not None:
            kwargs["amplitude_gain"] = amplitude_gain
        pred = cast(torch.Tensor, model(**kwargs))
        return pred, None

    kwargs = {
        "zt": zt_w,
        "t": t,
        "z_cond": zc_w,
        "valid_mask": vm_w,
        "mem": mem,
        "return_mem": True,
    }
    if mix_style is not None:
        kwargs["mix_style"] = mix_style
    if mix_style_mask is not None:
        kwargs["mix_style_mask"] = mix_style_mask
    if amplitude_gain is not None:
        kwargs["amplitude_gain"] = amplitude_gain
    pred, mem = cast(tuple[torch.Tensor, torch.Tensor], model(**kwargs))
    if detach_memory:
        mem = mem.detach()
    return pred, mem
