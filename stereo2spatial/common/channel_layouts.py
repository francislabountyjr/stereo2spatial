"""Shared speaker layout metadata for preprocessing, training, and inference."""

from __future__ import annotations

import math
import re

SQRT_HALF = 1.0 / math.sqrt(2.0)

CHANNEL_ORDER_MONO = ["FC"]
CHANNEL_ORDER_STEREO = ["FL", "FR"]
CHANNEL_ORDER_HEADPHONE_VIRTUALIZER = ["FL", "FR"]
CHANNEL_ORDER_5_1_REAR = ["FL", "FR", "FC", "LFE", "BL", "BR"]
CHANNEL_ORDER_5_1_SIDE = ["FL", "FR", "FC", "LFE", "SL", "SR"]
CHANNEL_ORDER_7_1_4 = [
    "FL",
    "FR",
    "FC",
    "LFE",
    "BL",
    "BR",
    "SL",
    "SR",
    "TFL",
    "TFR",
    "TBL",
    "TBR",
]

SPEAKER_MASK_BITS: dict[str, int] = {
    "FL": 0x1,
    "FR": 0x2,
    "FC": 0x4,
    "LFE": 0x8,
    "BL": 0x10,
    "BR": 0x20,
    "FLC": 0x40,
    "FRC": 0x80,
    "BC": 0x100,
    "SL": 0x200,
    "SR": 0x400,
    "TC": 0x800,
    "TFL": 0x1000,
    "TFC": 0x2000,
    "TFR": 0x4000,
    "TBL": 0x8000,
    "TBC": 0x10000,
    "TBR": 0x20000,
}

AC3_DOWNMIX_COEFFICIENTS: dict[str, tuple[float, float]] = {
    "FL": (1.0, 0.0),
    "FR": (0.0, 1.0),
    "FC": (SQRT_HALF, SQRT_HALF),
    "LFE": (0.5, 0.5),
    "LFE2": (0.5, 0.5),
    "BL": (SQRT_HALF, 0.0),
    "BR": (0.0, SQRT_HALF),
    "SL": (SQRT_HALF, 0.0),
    "SR": (0.0, SQRT_HALF),
    "BC": (0.5, 0.5),
    "FLC": (SQRT_HALF, 0.0),
    "FRC": (0.0, SQRT_HALF),
    "TFL": (0.5, 0.0),
    "TFR": (0.0, 0.5),
    "TBL": (0.5, 0.0),
    "TBR": (0.0, 0.5),
    "TFC": (0.3535533905932738, 0.3535533905932738),
    "TC": (0.3535533905932738, 0.3535533905932738),
    "TBC": (0.3535533905932738, 0.3535533905932738),
}

LAYOUT_CHANNELS: dict[str, list[str]] = {
    "mono": CHANNEL_ORDER_MONO,
    "stereo": CHANNEL_ORDER_STEREO,
    "2.0": CHANNEL_ORDER_STEREO,
    "binaural": CHANNEL_ORDER_HEADPHONE_VIRTUALIZER,
    "binauralstereo": CHANNEL_ORDER_HEADPHONE_VIRTUALIZER,
    "headphone": CHANNEL_ORDER_HEADPHONE_VIRTUALIZER,
    "headphones": CHANNEL_ORDER_HEADPHONE_VIRTUALIZER,
    "headphonevirtualizer": CHANNEL_ORDER_HEADPHONE_VIRTUALIZER,
    "headphonevirtualiser": CHANNEL_ORDER_HEADPHONE_VIRTUALIZER,
    "headphone virtualizer": CHANNEL_ORDER_HEADPHONE_VIRTUALIZER,
    "headphone virtualiser": CHANNEL_ORDER_HEADPHONE_VIRTUALIZER,
    "2.1": ["FL", "FR", "LFE"],
    "3.0": ["FL", "FR", "FC"],
    "3.0(back)": ["FL", "FR", "BC"],
    "3.1": ["FL", "FR", "FC", "LFE"],
    "4.0": ["FL", "FR", "FC", "BC"],
    "quad": ["FL", "FR", "BL", "BR"],
    "quad(side)": ["FL", "FR", "SL", "SR"],
    "5.0": ["FL", "FR", "FC", "BL", "BR"],
    "5.0(rear)": ["FL", "FR", "FC", "BL", "BR"],
    "5.0(back)": ["FL", "FR", "FC", "BL", "BR"],
    "5.0(side)": ["FL", "FR", "FC", "SL", "SR"],
    "5.1": CHANNEL_ORDER_5_1_REAR,
    "5.1(rear)": CHANNEL_ORDER_5_1_REAR,
    "5.1(back)": CHANNEL_ORDER_5_1_REAR,
    "5.1(side)": CHANNEL_ORDER_5_1_SIDE,
    "6.1": ["FL", "FR", "FC", "LFE", "BC", "SL", "SR"],
    "6.1(back)": ["FL", "FR", "FC", "LFE", "BL", "BR", "BC"],
    "7.0": ["FL", "FR", "FC", "BL", "BR", "SL", "SR"],
    "7.0(front)": ["FL", "FR", "FC", "FLC", "FRC", "SL", "SR"],
    "7.1": ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
    "7.1(wide)": ["FL", "FR", "FC", "LFE", "FLC", "FRC", "SL", "SR"],
    "7.1.2": ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR", "TFL", "TFR"],
    "7.1.4": CHANNEL_ORDER_7_1_4,
}

CHANNEL_COUNT_FALLBACKS: dict[int, list[str]] = {
    1: CHANNEL_ORDER_MONO,
    2: CHANNEL_ORDER_STEREO,
    3: ["FL", "FR", "FC"],
    4: ["FL", "FR", "BL", "BR"],
    5: ["FL", "FR", "FC", "BL", "BR"],
    6: CHANNEL_ORDER_5_1_REAR,
    8: ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
    10: ["FL", "FR", "FC", "LFE", "SL", "SR", "TFL", "TFR", "TBL", "TBR"],
    12: CHANNEL_ORDER_7_1_4,
}


def _layout_key_candidates(layout: str) -> list[str]:
    key = layout.strip().lower()
    condensed = key.replace(" ", "")
    compact = re.sub(r"[\s_]+", "", key)
    dotted = key.replace("_", ".")
    return [
        key,
        condensed,
        compact,
        dotted,
        condensed.replace("_", "."),
        compact.replace("rear", "(rear)"),
        compact.replace("back", "(back)"),
        compact.replace("side", "(side)"),
        dotted.replace("rear", "(rear)"),
        dotted.replace("back", "(back)"),
        dotted.replace("side", "(side)"),
    ]


def channel_labels_for_layout(layout: str, num_channels: int) -> list[str]:
    """Resolve ordered speaker labels for a named layout or explicit label list."""
    for candidate in _layout_key_candidates(layout):
        if candidate in LAYOUT_CHANNELS:
            labels = LAYOUT_CHANNELS[candidate]
            if len(labels) == num_channels:
                return list(labels)
            break

    if "+" in layout:
        split_labels = [part.strip().upper() for part in layout.split("+")]
        if len(split_labels) == num_channels and all(split_labels):
            return split_labels

    fallback = CHANNEL_COUNT_FALLBACKS.get(num_channels)
    if fallback is not None:
        return list(fallback)
    return [f"C{i}" for i in range(num_channels)]


def channel_mask_for_order(channel_order: list[str] | tuple[str, ...]) -> int | None:
    """Return a WAVEX speaker mask for known labels, or None for unknown labels."""
    mask = 0
    for raw_label in channel_order:
        label = str(raw_label).upper()
        bit = SPEAKER_MASK_BITS.get(label)
        if bit is None:
            return None
        mask |= bit
    return mask


def channel_mask_for_layout(layout: str, num_channels: int) -> int | None:
    """Resolve a WAVEX speaker mask for a layout name and channel count."""
    return channel_mask_for_order(channel_labels_for_layout(layout, num_channels))


def is_headphone_virtualizer_layout(layout: str) -> bool:
    """Return whether a layout name refers to a binaural headphone render."""
    candidates = set(_layout_key_candidates(layout))
    headphone_keys = {
        "binaural",
        "binauralstereo",
        "headphone",
        "headphones",
        "headphonevirtualizer",
        "headphonevirtualiser",
        "headphone virtualizer",
        "headphone virtualiser",
    }
    return bool(candidates & headphone_keys)


def layout_to_suffix(layout: str) -> str:
    """Return a filesystem-friendly suffix for a layout label."""
    suffix = re.sub(r"[^0-9A-Za-z]+", "_", layout.strip()).strip("_")
    if not suffix:
        raise ValueError(f"Invalid layout value: {layout!r}")
    return suffix.lower()
