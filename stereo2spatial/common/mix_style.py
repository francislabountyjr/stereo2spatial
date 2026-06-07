"""Mix-style conditioning features for spatial waveform targets."""

from __future__ import annotations

import math
from typing import Any

import torch

MIX_STYLE_NAMES: tuple[str, ...] = (
    "center_focus",
    "center_lock",
    "front_width",
    "surround_amount",
    "rear_depth",
    "height_amount",
    "lfe_amount",
    "ambience_amount",
    "placement_sharpness",
    "spatial_contrast",
    "chorus_expansion",
    "lead_bloom",
    "motion_amount",
)

DEFAULT_MIX_STYLE_VALUE = 0.5
DEFAULT_MIX_STYLE_VECTOR: tuple[float, ...] = tuple(
    DEFAULT_MIX_STYLE_VALUE for _ in MIX_STYLE_NAMES
)
MIX_STYLE_PRESETS: dict[str, dict[str, Any]] = {
    "balanced": {
        "description": "Neutral corpus-normalized style with no directional bias.",
        "values": {
            "center_focus": 0.50,
            "center_lock": 0.50,
            "front_width": 0.50,
            "surround_amount": 0.50,
            "ambience_amount": 0.50,
            "placement_sharpness": 0.50,
            "spatial_contrast": 0.50,
            "chorus_expansion": 0.50,
            "lead_bloom": 0.50,
            "motion_amount": 0.50,
        },
    },
    "front_focus": {
        "description": "Stable lead-focused image with reduced ambience and motion.",
        "values": {
            "center_focus": 0.72,
            "center_lock": 0.75,
            "front_width": 0.40,
            "surround_amount": 0.32,
            "ambience_amount": 0.30,
            "placement_sharpness": 0.72,
            "spatial_contrast": 0.38,
            "chorus_expansion": 0.35,
            "lead_bloom": 0.35,
            "motion_amount": 0.25,
        },
    },
    "wide_studio": {
        "description": "Wider stereo-stage presentation while keeping the center controlled.",
        "values": {
            "center_focus": 0.55,
            "center_lock": 0.58,
            "front_width": 0.70,
            "surround_amount": 0.58,
            "ambience_amount": 0.55,
            "placement_sharpness": 0.55,
            "spatial_contrast": 0.52,
            "chorus_expansion": 0.55,
            "lead_bloom": 0.50,
            "motion_amount": 0.42,
        },
    },
    "ambient_wide": {
        "description": "Diffuse spacious render with more ambience and less pinpoint placement.",
        "values": {
            "center_focus": 0.42,
            "center_lock": 0.38,
            "front_width": 0.68,
            "surround_amount": 0.78,
            "ambience_amount": 0.80,
            "placement_sharpness": 0.28,
            "spatial_contrast": 0.58,
            "chorus_expansion": 0.60,
            "lead_bloom": 0.55,
            "motion_amount": 0.45,
        },
    },
    "cinematic": {
        "description": "Large contrast and bloom for a bigger section-to-section image.",
        "values": {
            "center_focus": 0.55,
            "center_lock": 0.55,
            "front_width": 0.62,
            "surround_amount": 0.65,
            "ambience_amount": 0.62,
            "placement_sharpness": 0.48,
            "spatial_contrast": 0.78,
            "chorus_expansion": 0.75,
            "lead_bloom": 0.70,
            "motion_amount": 0.55,
        },
    },
    "energetic_motion": {
        "description": "More animated spatial changes and chorus-style expansion.",
        "values": {
            "center_focus": 0.48,
            "center_lock": 0.42,
            "front_width": 0.62,
            "surround_amount": 0.60,
            "ambience_amount": 0.55,
            "placement_sharpness": 0.45,
            "spatial_contrast": 0.72,
            "chorus_expansion": 0.72,
            "lead_bloom": 0.58,
            "motion_amount": 0.78,
        },
    },
    "intimate": {
        "description": "Close, dry, centered render with minimal ambience and motion.",
        "values": {
            "center_focus": 0.78,
            "center_lock": 0.82,
            "front_width": 0.30,
            "surround_amount": 0.22,
            "ambience_amount": 0.20,
            "placement_sharpness": 0.78,
            "spatial_contrast": 0.25,
            "chorus_expansion": 0.22,
            "lead_bloom": 0.25,
            "motion_amount": 0.18,
        },
    },
}
_INACTIVE_BY_MODE: dict[str, tuple[str, ...]] = {
    "5_1_rear": ("rear_depth", "height_amount"),
    "binaural_stereo": ("rear_depth", "height_amount", "lfe_amount"),
}

_EPS = 1e-8
_BANDS: dict[str, tuple[float, float]] = {
    "M": (80.0, 16_000.0),
    "V": (150.0, 6_000.0),
    "L": (20.0, 120.0),
    "D": (500.0, 8_000.0),
}
_POSITIONS: dict[str, tuple[float, float, float]] = {
    "FL": (-1.0, 1.0, 0.0),
    "FR": (1.0, 1.0, 0.0),
    "FC": (0.0, 1.0, 0.0),
    "SL": (-1.0, 0.0, 0.0),
    "SR": (1.0, 0.0, 0.0),
    "BL": (-1.0, -1.0, 0.0),
    "BR": (1.0, -1.0, 0.0),
    "TFL": (-1.0, 1.0, 1.0),
    "TFR": (1.0, 1.0, 1.0),
    "TBL": (-1.0, -1.0, 1.0),
    "TBR": (1.0, -1.0, 1.0),
}


def mix_style_dict_to_vector(
    values: dict[str, Any] | None,
    *,
    default: float = DEFAULT_MIX_STYLE_VALUE,
    names: tuple[str, ...] | list[str] | None = None,
) -> list[float]:
    """Return a vector ordered by ``MIX_STYLE_NAMES`` with finite defaults."""
    resolved_names = list(MIX_STYLE_NAMES if names is None else names)
    if not isinstance(values, dict):
        return [float(default) for _ in resolved_names]
    vector: list[float] = []
    for name in resolved_names:
        raw = values.get(name, default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = float(default)
        if not math.isfinite(value):
            value = float(default)
        vector.append(value)
    return vector


def mix_style_active_names(
    *,
    layout_mode: str | None = None,
    inactive_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Return the active mix-style names for a target layout mode."""
    inactive = {
        str(name)
        for name in _INACTIVE_BY_MODE.get(str(layout_mode or ""), ())
    }
    if inactive_names is not None:
        inactive.update(str(name) for name in inactive_names)
    return tuple(name for name in MIX_STYLE_NAMES if name not in inactive)


def mix_style_preset_names() -> tuple[str, ...]:
    """Return available named mix-style presets in display order."""
    return tuple(MIX_STYLE_PRESETS)


def mix_style_preset_description(name: str) -> str:
    """Return a human-readable description for a mix-style preset."""
    key = str(name).strip().lower().replace("-", "_")
    if key not in MIX_STYLE_PRESETS:
        raise KeyError(f"unknown mix-style preset: {name}")
    return str(MIX_STYLE_PRESETS[key]["description"])


def mix_style_preset_values(name: str) -> dict[str, float]:
    """Return normalized knob values for a named mix-style preset."""
    key = str(name).strip().lower().replace("-", "_")
    if key not in MIX_STYLE_PRESETS:
        raise KeyError(f"unknown mix-style preset: {name}")
    values = MIX_STYLE_PRESETS[key]["values"]
    return {str(knob): float(value) for knob, value in values.items()}


def mix_style_inactive_names_for_layout(
    *,
    target_channels: int,
    is_binaural_stereo: bool,
    channel_labels: list[str],
) -> tuple[str, ...]:
    """Return controls that have no physical meaning for a target layout."""
    labels = {str(label).upper() for label in channel_labels}
    inactive: set[str] = set()
    if is_binaural_stereo:
        inactive.update(_INACTIVE_BY_MODE["binaural_stereo"])
    else:
        if not {"TFL", "TFR", "TBL", "TBR"} & labels:
            inactive.add("height_amount")
        if {"BL", "BR"} & labels and not {"SL", "SR"} & labels:
            inactive.add("rear_depth")
    return tuple(name for name in MIX_STYLE_NAMES if name in inactive)


def normalize_mix_style_raw(
    raw_values: dict[str, Any],
    stats: dict[str, dict[str, float]],
    *,
    names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, float]:
    """Normalize raw mix-style scalars with robust corpus percentile stats."""
    normalized: dict[str, float] = {}
    for name in (MIX_STYLE_NAMES if names is None else names):
        raw = float(raw_values.get(name, DEFAULT_MIX_STYLE_VALUE))
        bounds = stats.get(name, {})
        lower = float(bounds.get("p01", bounds.get("lower", 0.0)))
        upper = float(bounds.get("p99", bounds.get("upper", 1.0)))
        denom = max(upper - lower, _EPS)
        value = (raw - lower) / denom
        normalized[name] = float(min(1.0, max(0.0, value)))
    return normalized


def _label_index(channel_labels: list[str], label: str) -> int | None:
    label = label.upper()
    for index, candidate in enumerate(channel_labels):
        if str(candidate).upper() == label:
            return index
    return None


def _group_indices(channel_labels: list[str], labels: tuple[str, ...]) -> list[int]:
    return [
        index
        for label in labels
        if (index := _label_index(channel_labels, label)) is not None
    ]


def _previous_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value.bit_length() - 1)


def _compute_band_energies(
    audio: torch.Tensor,
    sample_rate: int,
) -> dict[str, torch.Tensor]:
    samples = int(audio.shape[-1])
    if samples <= 1:
        zeros = torch.zeros((audio.shape[0], 1), dtype=torch.float32)
        return {name: zeros.clone() for name in _BANDS}

    target_fft = min(8192, max(512, _previous_power_of_two(max(512, sample_rate // 4))))
    n_fft = min(target_fft, _previous_power_of_two(samples))
    n_fft = max(16, int(n_fft))
    hop = max(1, min(samples, sample_rate // 4))
    window = torch.hann_window(n_fft, dtype=torch.float32, device=audio.device)
    spec = torch.stft(
        audio.float(),
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    power = spec.abs().pow(2)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate)).to(audio.device)

    energies: dict[str, torch.Tensor] = {}
    for name, (low, high) in _BANDS.items():
        mask = (freqs >= float(low)) & (freqs <= float(high))
        if not bool(mask.any().item()):
            energies[name] = torch.zeros(
                (audio.shape[0], power.shape[-1]), dtype=torch.float32, device=audio.device
            )
        else:
            energies[name] = power[:, mask, :].mean(dim=1).float()
    return energies


def _channel_energy(
    band_energy: torch.Tensor,
    channel_labels: list[str],
    label: str,
) -> torch.Tensor:
    index = _label_index(channel_labels, label)
    if index is None or index >= band_energy.shape[0]:
        return torch.zeros(
            (band_energy.shape[-1],), dtype=band_energy.dtype, device=band_energy.device
        )
    return band_energy[index]


def _group_energy(
    band_energy: torch.Tensor,
    channel_labels: list[str],
    labels: tuple[str, ...],
) -> torch.Tensor:
    indices = [idx for idx in _group_indices(channel_labels, labels) if idx < band_energy.shape[0]]
    if not indices:
        return torch.zeros(
            (band_energy.shape[-1],), dtype=band_energy.dtype, device=band_energy.device
        )
    return band_energy[indices].sum(dim=0)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    values = values.float()
    weights = weights.float().to(device=values.device)
    return (values * weights).sum() / weights.sum().clamp_min(_EPS)


def _weighted_percentile(
    values: torch.Tensor,
    weights: torch.Tensor,
    percentile: float,
) -> torch.Tensor:
    values = values.flatten().float()
    weights = weights.flatten().float().to(device=values.device).clamp_min(0.0)
    if values.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=values.device)
    total = weights.sum()
    if float(total.item()) <= _EPS:
        return torch.quantile(values, float(percentile) / 100.0)
    order = torch.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = total * (float(percentile) / 100.0)
    index = int(torch.searchsorted(torch.cumsum(sorted_weights, dim=0), cutoff).item())
    index = min(max(index, 0), sorted_values.numel() - 1)
    return sorted_values[index]


def _weighted_corr(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    a = a.float()
    b = b.float().to(device=a.device)
    weights = weights.float().to(device=a.device)
    ma = _weighted_mean(a, weights)
    mb = _weighted_mean(b, weights)
    da = a - ma
    db = b - mb
    num = (weights * da * db).sum()
    den = torch.sqrt((weights * da.pow(2)).sum() * (weights * db.pow(2)).sum() + _EPS)
    return num / den.clamp_min(_EPS)


def _resize_like(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] == reference.shape[-1]:
        return values.to(device=reference.device, dtype=reference.dtype)
    src = values.flatten().float()[None, None, :]
    resized = torch.nn.functional.interpolate(
        src,
        size=int(reference.shape[-1]),
        mode="linear",
        align_corners=False,
    )[0, 0]
    return resized.to(device=reference.device, dtype=reference.dtype)


def _frame_correlations(
    audio: torch.Tensor,
    pairs: list[tuple[int, int]],
    sample_rate: int,
    target_frames: int,
) -> dict[tuple[int, int], torch.Tensor]:
    if not pairs:
        return {}
    samples = int(audio.shape[-1])
    frame_size = max(16, min(samples, sample_rate // 4))
    hop = frame_size
    if samples < frame_size:
        pad = torch.zeros(
            (audio.shape[0], frame_size - samples), dtype=audio.dtype, device=audio.device
        )
        audio = torch.cat([audio, pad], dim=-1)
    frames = audio.float().unfold(dimension=1, size=frame_size, step=hop)
    if frames.numel() == 0:
        frames = audio.float()[:, None, :]

    out: dict[tuple[int, int], torch.Tensor] = {}
    for left, right in pairs:
        if left >= frames.shape[0] or right >= frames.shape[0]:
            continue
        a = frames[left]
        b = frames[right]
        a = a - a.mean(dim=-1, keepdim=True)
        b = b - b.mean(dim=-1, keepdim=True)
        corr = (a * b).sum(dim=-1) / torch.sqrt(
            a.pow(2).sum(dim=-1) * b.pow(2).sum(dim=-1) + _EPS
        )
        out[(left, right)] = _resize_like(corr.clamp(-1.0, 1.0), torch.empty(target_frames))
    return out


def _value(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu().item())
    if not math.isfinite(float(value)):
        return 0.0
    return float(value)


def _mid_side_band_energy(
    left: torch.Tensor,
    right: torch.Tensor,
    rho: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate mid/side energies from L/R energies and correlation."""
    cross = rho.to(device=left.device, dtype=left.dtype) * torch.sqrt(
        (left * right).clamp_min(0.0)
    )
    mid = (0.5 * (left + right + 2.0 * cross)).clamp_min(0.0)
    side = (0.5 * (left + right - 2.0 * cross)).clamp_min(0.0)
    return mid, side


def _compute_stereo_binaural_mix_style_raw(
    audio: torch.Tensor,
    sample_rate: int,
    labels: list[str],
) -> dict[str, float]:
    """Compute useful style surrogates for 2-channel binaural/stereo targets."""
    energies = _compute_band_energies(audio, sample_rate)
    e_m = energies["M"]
    e_v = energies["V"]
    e_l = energies["L"]
    frame_count = int(e_m.shape[-1])

    fl_idx = _label_index(labels, "FL") or 0
    fr_idx = _label_index(labels, "FR")
    fr_idx = 1 if fr_idx is None and audio.shape[0] > 1 else fr_idx
    if fr_idx is None:
        fr_idx = fl_idx

    corr_by_pair = _frame_correlations(
        audio=audio,
        pairs=[(fl_idx, fr_idx)] if fl_idx != fr_idx else [],
        sample_rate=sample_rate,
        target_frames=frame_count,
    )
    rho_lr = corr_by_pair.get((fl_idx, fr_idx), torch.ones(frame_count)).to(
        dtype=e_m.dtype
    )
    rho_lr = rho_lr.clamp(-1.0, 1.0)

    fl_m = e_m[fl_idx]
    fr_m = e_m[fr_idx]
    fl_v = e_v[fl_idx]
    fr_v = e_v[fr_idx]
    fl_l = e_l[fl_idx]
    fr_l = e_l[fr_idx]

    mid_m, side_m = _mid_side_band_energy(fl_m, fr_m, rho_lr)
    mid_v, side_v = _mid_side_band_energy(fl_v, fr_v, rho_lr)
    mid_l, side_l = _mid_side_band_energy(fl_l, fr_l, rho_lr)
    total_m = (mid_m + side_m).clamp_min(_EPS)
    total_v = (mid_v + side_v).clamp_min(_EPS)
    total_l = (mid_l + side_l).clamp_min(_EPS)

    intensity = torch.log(total_m + _EPS)
    p10 = torch.quantile(intensity, 0.10)
    p95 = torch.quantile(intensity, 0.95)
    weights = ((intensity - p10) / (p95 - p10).clamp_min(_EPS)).clamp(0.0, 1.0).pow(2)
    if float(weights.sum().item()) <= _EPS:
        weights = torch.ones_like(weights)

    center_ratio = mid_v / total_v
    center_focus = _weighted_mean(center_ratio, weights)
    center_range = _weighted_percentile(center_ratio, weights, 90.0) - _weighted_percentile(
        center_ratio, weights, 10.0
    )
    center_lock = 1.0 - torch.clamp(center_range / 0.50, 0.0, 1.0)

    d_lr = ((1.0 - rho_lr) / 2.0).clamp(0.0, 1.0)
    b_lr = 1.0 - (fl_m - fr_m).abs() / (fl_m + fr_m + _EPS)
    front_width = _weighted_mean(d_lr * b_lr.clamp(0.0, 1.0), weights)

    side_ratio_m = (side_m / total_m).clamp(0.0, 1.0)
    side_ratio_v = (side_v / total_v).clamp(0.0, 1.0)
    surround_amount = _weighted_mean(side_ratio_m, weights)
    height_amount = torch.tensor(DEFAULT_MIX_STYLE_VALUE)
    rear_depth = torch.tensor(DEFAULT_MIX_STYLE_VALUE)
    lfe_amount = torch.tensor(DEFAULT_MIX_STYLE_VALUE)

    ambience_amount = _weighted_mean(side_ratio_m * d_lr, weights)
    balance = 1.0 - (fl_m - fr_m).abs() / (fl_m + fr_m + _EPS)
    placement_sharpness = _weighted_mean((1.0 - side_ratio_m) * balance, weights)
    spatial_range = _weighted_percentile(side_ratio_m, weights, 90.0) - _weighted_percentile(
        side_ratio_m, weights, 10.0
    )
    spatial_contrast = torch.clamp(spatial_range / 0.50, 0.0, 1.0)
    chorus_expansion = _weighted_corr(intensity, side_ratio_m, weights).clamp(0.0, 1.0)
    chorus_expansion = chorus_expansion * torch.clamp(spatial_range / 0.35, 0.0, 1.0)

    lead_range = _weighted_percentile(side_ratio_v, weights, 90.0) - _weighted_percentile(
        side_ratio_v, weights, 50.0
    )
    lead_coupling = _weighted_corr(intensity, side_ratio_v, weights).clamp(0.0, 1.0)
    lead_bloom = torch.sqrt(torch.clamp(lead_range / 0.35, 0.0, 1.0) * lead_coupling)

    if frame_count > 1:
        width_delta = (side_ratio_m[1:] - side_ratio_m[:-1]).abs()
        balance_pos = (fr_m - fl_m) / (fr_m + fl_m + _EPS)
        balance_delta = (balance_pos[1:] - balance_pos[:-1]).abs()
        motion = torch.sqrt(width_delta.pow(2) + balance_delta.pow(2))
        motion_amount = torch.clamp(_weighted_mean(motion, weights[1:]) / 0.50, 0.0, 1.0)
    else:
        motion_amount = torch.tensor(0.0)

    raw = {
        "center_focus": center_focus,
        "center_lock": center_lock,
        "front_width": front_width,
        "surround_amount": surround_amount,
        "rear_depth": rear_depth,
        "height_amount": height_amount,
        "lfe_amount": lfe_amount,
        "ambience_amount": ambience_amount,
        "placement_sharpness": placement_sharpness,
        "spatial_contrast": spatial_contrast,
        "chorus_expansion": chorus_expansion,
        "lead_bloom": lead_bloom,
        "motion_amount": motion_amount,
    }
    return {name: _value(raw[name]) for name in MIX_STYLE_NAMES}


def compute_mix_style_raw(
    audio: torch.Tensor,
    sample_rate: int,
    channel_labels: list[str],
) -> dict[str, float]:
    """
    Compute raw target-derived mix-style controls from channel-first spatial audio.

    The returned values are song-level scalars in the fixed ``MIX_STYLE_NAMES`` order.
    They should still be corpus-normalized before training.
    """
    if audio.dim() != 2:
        raise ValueError(f"audio must be [C,S], got {tuple(audio.shape)}")
    if audio.shape[0] != len(channel_labels):
        raise ValueError(
            "channel_labels length must match audio channels "
            f"({len(channel_labels)} != {audio.shape[0]})"
        )
    if int(sample_rate) <= 0:
        raise ValueError("sample_rate must be > 0")

    device = torch.device("cpu")
    audio = audio.detach().to(device=device, dtype=torch.float32)
    sample_rate = int(sample_rate)
    labels = [str(label).upper() for label in channel_labels]
    if audio.shape[0] == 2 and set(labels) <= {"FL", "FR"}:
        return _compute_stereo_binaural_mix_style_raw(
            audio=audio,
            sample_rate=sample_rate,
            labels=labels,
        )
    energies = _compute_band_energies(audio, sample_rate)
    e_m = energies["M"]
    e_v = energies["V"]
    e_l = energies["L"]
    e_d = energies["D"]
    frame_count = int(e_m.shape[-1])

    fl_m = _channel_energy(e_m, labels, "FL")
    fr_m = _channel_energy(e_m, labels, "FR")
    fc_m = _channel_energy(e_m, labels, "FC")
    fl_v = _channel_energy(e_v, labels, "FL")
    fr_v = _channel_energy(e_v, labels, "FR")
    fc_v = _channel_energy(e_v, labels, "FC")
    lfe_l = _channel_energy(e_l, labels, "LFE")

    side = ("SL", "SR")
    back = ("BL", "BR")
    sur = ("BL", "BR", "SL", "SR")
    height = ("TFL", "TFR", "TBL", "TBR")
    spat = ("BL", "BR", "SL", "SR", "TFL", "TFR", "TBL", "TBR")
    nl = tuple(label for label in labels if label != "LFE")

    e_nl_m = _group_energy(e_m, labels, nl)
    e_nl_v = _group_energy(e_v, labels, nl)
    e_nl_l = _group_energy(e_l, labels, nl)
    e_sur_m = _group_energy(e_m, labels, sur)
    e_height_m = _group_energy(e_m, labels, height)
    e_spat_m = _group_energy(e_m, labels, spat)
    e_spat_v = _group_energy(e_v, labels, spat)
    e_back_m = _group_energy(e_m, labels, back)
    e_side_m = _group_energy(e_m, labels, side)

    intensity = torch.log(e_nl_m + _EPS)
    p10 = torch.quantile(intensity, 0.10)
    p95 = torch.quantile(intensity, 0.95)
    weights = ((intensity - p10) / (p95 - p10).clamp_min(_EPS)).clamp(0.0, 1.0).pow(2)
    if float(weights.sum().item()) <= _EPS:
        weights = torch.ones_like(weights)

    center_ratio = fc_v / (fl_v + fr_v + fc_v + _EPS)
    center_focus = _weighted_mean(center_ratio, weights)
    center_range = _weighted_percentile(center_ratio, weights, 90.0) - _weighted_percentile(
        center_ratio, weights, 10.0
    )
    center_lock = 1.0 - torch.clamp(center_range / 0.50, 0.0, 1.0)

    pair_indices: list[tuple[int, int]] = []
    fl_idx = _label_index(labels, "FL")
    fr_idx = _label_index(labels, "FR")
    if fl_idx is not None and fr_idx is not None:
        pair_indices.append((fl_idx, fr_idx))
    spat_indices = _group_indices(labels, spat)
    for left_pos, left_idx in enumerate(spat_indices):
        for right_idx in spat_indices[left_pos + 1 :]:
            pair_indices.append((left_idx, right_idx))
    corr_by_pair = _frame_correlations(
        audio=audio,
        pairs=pair_indices,
        sample_rate=sample_rate,
        target_frames=frame_count,
    )
    rho_lr = (
        corr_by_pair.get((fl_idx, fr_idx), torch.zeros(frame_count))
        if fl_idx is not None and fr_idx is not None
        else torch.zeros(frame_count)
    )
    rho_lr = rho_lr.to(dtype=e_m.dtype)
    d_lr = ((1.0 - rho_lr) / 2.0).clamp(0.0, 1.0)
    b_lr = 1.0 - (fl_m - fr_m).abs() / (fl_m + fr_m + _EPS)
    p_lr = (fl_m + fr_m) / (fl_m + fr_m + fc_m + _EPS)
    front_width = _weighted_mean(d_lr * b_lr.clamp(0.0, 1.0) * torch.sqrt(p_lr), weights)

    r_sur = e_sur_m / (e_nl_m + _EPS)
    r_height = e_height_m / (e_nl_m + _EPS)
    r_spat = (e_sur_m + e_height_m) / (e_nl_m + _EPS)
    surround_amount = _weighted_mean(r_sur, weights)
    height_amount = _weighted_mean(r_height, weights)
    lfe_amount = _weighted_mean(lfe_l / (lfe_l + e_nl_l + _EPS), weights)

    rear_weight = weights * r_sur
    rear_depth = _weighted_mean(e_back_m / (e_back_m + e_side_m + _EPS), rear_weight)
    if _value(surround_amount) < 0.02:
        rear_depth = torch.tensor(DEFAULT_MIX_STYLE_VALUE)

    diff_num = torch.zeros(frame_count, dtype=torch.float32)
    diff_den = torch.zeros(frame_count, dtype=torch.float32)
    for left_pos, left_idx in enumerate(spat_indices):
        for right_idx in spat_indices[left_pos + 1 :]:
            corr = corr_by_pair.get((left_idx, right_idx))
            if corr is None:
                continue
            q = torch.sqrt(e_d[left_idx] * e_d[right_idx]).float()
            decor = 1.0 - corr.float().abs().clamp(0.0, 1.0)
            diff_num = diff_num + q * decor
            diff_den = diff_den + q
    diff_spat = diff_num / diff_den.clamp_min(_EPS)
    ambience_amount = _weighted_mean(r_spat * diff_spat.to(dtype=r_spat.dtype), weights)

    if spat_indices:
        spat_energy = e_m[spat_indices]
        probs = spat_energy / e_spat_m.clamp_min(_EPS)[None, :]
        entropy = -(probs * torch.log(probs + _EPS)).sum(dim=0) / math.log(
            max(2, len(spat_indices))
        )
        sharpness = 1.0 - entropy.clamp(0.0, 1.0)
    else:
        sharpness = torch.full_like(r_spat, DEFAULT_MIX_STYLE_VALUE)
    spatial_weight = weights * r_spat
    placement_sharpness = _weighted_mean(sharpness, spatial_weight)
    if _value(_weighted_mean(r_spat, weights)) < 0.02:
        placement_sharpness = torch.tensor(DEFAULT_MIX_STYLE_VALUE)

    spatial_range = _weighted_percentile(r_spat, weights, 90.0) - _weighted_percentile(
        r_spat, weights, 10.0
    )
    spatial_contrast = torch.clamp(spatial_range / 0.50, 0.0, 1.0)
    chorus_expansion = _weighted_corr(intensity, r_spat, weights).clamp(0.0, 1.0)
    chorus_expansion = chorus_expansion * torch.clamp(spatial_range / 0.35, 0.0, 1.0)

    b_lead = e_spat_v / (e_spat_v + fc_v + _EPS)
    u = weights * fc_v / (e_nl_v + _EPS)
    lead_range = _weighted_percentile(b_lead, u, 90.0) - _weighted_percentile(
        b_lead, u, 50.0
    )
    lead_coupling = _weighted_corr(intensity, b_lead, u).clamp(0.0, 1.0)
    lead_bloom = torch.sqrt(torch.clamp(lead_range / 0.35, 0.0, 1.0) * lead_coupling)

    centroid = torch.zeros((frame_count, 3), dtype=torch.float32)
    for label, position in _POSITIONS.items():
        idx = _label_index(labels, label)
        if idx is None or idx >= e_m.shape[0]:
            continue
        p_c = (e_m[idx] / (e_nl_m + _EPS)).float()
        pos = torch.tensor(position, dtype=torch.float32)
        centroid = centroid + p_c[:, None] * pos[None, :]
    if frame_count > 1:
        motion = torch.linalg.vector_norm(centroid[1:] - centroid[:-1], dim=1)
        motion_weight = weights[1:] * torch.maximum(r_spat[1:], r_spat[:-1])
        motion_amount = torch.clamp(_weighted_mean(motion, motion_weight) / 0.50, 0.0, 1.0)
    else:
        motion_amount = torch.tensor(0.0)

    raw = {
        "center_focus": center_focus,
        "center_lock": center_lock,
        "front_width": front_width,
        "surround_amount": surround_amount,
        "rear_depth": rear_depth,
        "height_amount": height_amount,
        "lfe_amount": lfe_amount,
        "ambience_amount": ambience_amount,
        "placement_sharpness": placement_sharpness,
        "spatial_contrast": spatial_contrast,
        "chorus_expansion": chorus_expansion,
        "lead_bloom": lead_bloom,
        "motion_amount": motion_amount,
    }
    return {name: _value(raw[name]) for name in MIX_STYLE_NAMES}


__all__ = [
    "DEFAULT_MIX_STYLE_VALUE",
    "DEFAULT_MIX_STYLE_VECTOR",
    "MIX_STYLE_PRESETS",
    "MIX_STYLE_NAMES",
    "compute_mix_style_raw",
    "mix_style_active_names",
    "mix_style_dict_to_vector",
    "mix_style_inactive_names_for_layout",
    "mix_style_preset_description",
    "mix_style_preset_names",
    "mix_style_preset_values",
    "normalize_mix_style_raw",
]
