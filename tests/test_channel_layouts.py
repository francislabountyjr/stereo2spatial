from __future__ import annotations

from stereo2spatial.common.channel_layouts import (
    channel_labels_for_layout,
    channel_mask_for_layout,
    channel_mask_for_order,
    is_headphone_virtualizer_layout,
)
from stereo2spatial.common.mix_style import (
    MIX_STYLE_NAMES,
    mix_style_active_names,
    mix_style_inactive_names_for_layout,
)


def test_5_1_rear_aliases_resolve_to_back_surround_order() -> None:
    expected = ["FL", "FR", "FC", "LFE", "BL", "BR"]

    assert channel_labels_for_layout("5.1", 6) == expected
    assert channel_labels_for_layout("5.1 rear", 6) == expected
    assert channel_labels_for_layout("5.1_rear", 6) == expected
    assert channel_labels_for_layout("5.1 back", 6) == expected


def test_5_1_rear_and_side_have_distinct_wavex_masks() -> None:
    assert channel_mask_for_layout("5.1 rear", 6) == 0x3F
    assert channel_mask_for_layout("5.1 side", 6) == 0x60F
    assert channel_mask_for_order(["FL", "FR", "FC", "LFE", "BL", "BR"]) == 0x3F


def test_headphone_virtualizer_resolves_as_binaural_stereo() -> None:
    assert channel_labels_for_layout("Headphone Virtualizer", 2) == ["FL", "FR"]
    assert channel_mask_for_layout("Headphone Virtualizer", 2) == 0x3
    assert is_headphone_virtualizer_layout("Headphone Virtualizer")
    assert is_headphone_virtualizer_layout("binaural")


def test_layout_inactive_mix_style_names_are_omitted() -> None:
    rear_inactive = mix_style_inactive_names_for_layout(
        target_channels=6,
        is_binaural_stereo=False,
        channel_labels=["FL", "FR", "FC", "LFE", "BL", "BR"],
    )
    rear_active = mix_style_active_names(inactive_names=rear_inactive)
    assert rear_inactive == ("rear_depth", "height_amount")
    assert len(rear_active) == len(MIX_STYLE_NAMES) - 2
    assert "rear_depth" not in rear_active
    assert "height_amount" not in rear_active

    headphone_inactive = mix_style_inactive_names_for_layout(
        target_channels=2,
        is_binaural_stereo=True,
        channel_labels=["FL", "FR"],
    )
    headphone_active = mix_style_active_names(inactive_names=headphone_inactive)
    assert headphone_inactive == ("rear_depth", "height_amount", "lfe_amount")
    assert len(headphone_active) == len(MIX_STYLE_NAMES) - 3
    assert "lfe_amount" not in headphone_active
