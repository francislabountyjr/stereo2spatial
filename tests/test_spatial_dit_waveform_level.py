from __future__ import annotations

import pytest
import torch

from stereo2spatial.modeling import SpatialDiT
from stereo2spatial.modeling.layers import WaveformTransformerBlock


def _build_model(
    *,
    waveform_level_depth: int,
    activation_checkpointing: bool = False,
) -> SpatialDiT:
    return SpatialDiT(
        target_channels=2,
        cond_channels=2,
        patch_size=8,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        timestep_embed_dim=16,
        timestep_scale=1000.0,
        max_period=10000.0,
        num_memory_tokens=0,
        waveform_level_depth=waveform_level_depth,
        waveform_micro_patch_size=2,
        waveform_hidden_dim=8,
        waveform_num_heads=4,
        waveform_mlp_ratio=2.0,
        activation_checkpointing=activation_checkpointing,
    )


def test_waveform_level_branch_starts_as_small_residual_refiner() -> None:
    torch.manual_seed(123)
    coarse_model = _build_model(waveform_level_depth=0)
    torch.manual_seed(123)
    waveform_model = _build_model(waveform_level_depth=1)

    zt = torch.randn(2, 2, 8, 5)
    zc = torch.randn(2, 2, 8, 5)
    mask = torch.ones(2, 5, dtype=torch.bool)
    t = torch.tensor([0.25, 0.75])

    coarse = coarse_model(zt=zt, t=t, z_cond=zc, valid_mask=mask)
    refined = waveform_model(zt=zt, t=t, z_cond=zc, valid_mask=mask)

    assert refined.shape == zt.shape
    residual = refined - coarse
    assert torch.isfinite(residual).all()
    assert residual.abs().amax().item() < 0.05
    assert residual.abs().sum().item() > 0.0


def test_waveform_level_branch_receives_output_gradient() -> None:
    torch.manual_seed(123)
    model = _build_model(waveform_level_depth=1)
    zt = torch.randn(2, 2, 8, 5)
    zc = torch.randn(2, 2, 8, 5)
    mask = torch.ones(2, 5, dtype=torch.bool)
    t = torch.tensor([0.25, 0.75])

    loss = model(zt=zt, t=t, z_cond=zc, valid_mask=mask).square().mean()
    loss.backward()

    assert model.waveform_out is not None
    assert model.waveform_in is not None
    weight_grad = model.waveform_out.weight.grad
    input_grad = model.waveform_in.weight.grad
    assert weight_grad is not None
    assert input_grad is not None
    assert torch.isfinite(weight_grad).all()
    assert torch.isfinite(input_grad).all()
    assert weight_grad.abs().sum().item() > 0.0
    assert input_grad.abs().sum().item() > 0.0


def test_activation_checkpointing_preserves_backward() -> None:
    torch.manual_seed(123)
    model = _build_model(
        waveform_level_depth=1,
        activation_checkpointing=True,
    )
    model.train()
    zt = torch.randn(2, 2, 8, 5, requires_grad=True)
    zc = torch.randn(2, 2, 8, 5)
    mask = torch.ones(2, 5, dtype=torch.bool)
    t = torch.tensor([0.25, 0.75])

    loss = model(zt=zt, t=t, z_cond=zc, valid_mask=mask).square().mean()
    loss.backward()

    assert zt.grad is not None
    assert torch.isfinite(zt.grad).all()
    assert model.target_in.weight.grad is not None
    assert model.waveform_out is not None
    assert model.waveform_out.weight.grad is not None


def test_conditioned_final_output_block_works_without_waveform_level() -> None:
    torch.manual_seed(123)
    model = _build_model(waveform_level_depth=0)
    zt = torch.randn(2, 2, 8, 5)
    zc = torch.randn(2, 2, 8, 5)
    mask = torch.ones(2, 5, dtype=torch.bool)
    t = torch.tensor([0.25, 0.75])

    out = model(zt=zt, t=t, z_cond=zc, valid_mask=mask)
    loss = out.square().mean()
    loss.backward()

    assert out.shape == zt.shape
    assert torch.isfinite(out).all()
    assert model.final_output.conv.weight.grad is not None
    assert model.final_output.adaLN_modulation[-1].weight.grad is not None
    assert model.cond_in.weight.grad is not None
    assert model.cond_in.weight.grad.abs().sum().item() > 0.0


def test_inference_conditioning_cache_matches_uncached_forward() -> None:
    torch.manual_seed(123)
    model = SpatialDiT(
        target_channels=2,
        cond_channels=2,
        patch_size=8,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        timestep_embed_dim=16,
        timestep_scale=1000.0,
        max_period=10000.0,
        num_memory_tokens=3,
        waveform_level_depth=1,
        waveform_micro_patch_size=2,
        waveform_hidden_dim=8,
        waveform_num_heads=4,
        waveform_mlp_ratio=2.0,
    )
    model.eval()
    zt = torch.randn(2, 2, 8, 5)
    zc = torch.randn(2, 2, 8, 5)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ],
        dtype=torch.bool,
    )
    t = torch.tensor([0.25, 0.75])
    mem = model.init_memory(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)

    with torch.inference_mode():
        uncached, uncached_mem = model(
            zt=zt,
            t=t,
            z_cond=zc,
            valid_mask=mask,
            mem=mem,
            return_mem=True,
        )
        cache = model.build_conditioning_cache(z_cond=zc, valid_mask=mask)
        cached, cached_mem = model(
            zt=zt,
            t=t,
            z_cond=zc,
            valid_mask=mask,
            mem=mem,
            return_mem=True,
            conditioning_cache=cache,
        )

    assert torch.allclose(cached, uncached, atol=1.0e-6, rtol=1.0e-6)
    assert uncached_mem is not None
    assert cached_mem is not None
    assert torch.allclose(cached_mem, uncached_mem, atol=1.0e-6, rtol=1.0e-6)


def test_conditioned_final_output_block_uses_source_tokens() -> None:
    torch.manual_seed(123)
    model = _build_model(waveform_level_depth=0)
    zt = torch.randn(1, 2, 8, 5)
    zc_a = torch.zeros(1, 2, 8, 5)
    zc_b = torch.randn(1, 2, 8, 5)
    mask = torch.ones(1, 5, dtype=torch.bool)
    t = torch.tensor([0.5])

    out_a = model(zt=zt, t=t, z_cond=zc_a, valid_mask=mask)
    out_b = model(zt=zt, t=t, z_cond=zc_b, valid_mask=mask)

    assert not torch.allclose(out_a, out_b)


def test_waveform_level_branch_uses_conditioning_microtokens() -> None:
    torch.manual_seed(123)
    model = _build_model(waveform_level_depth=1)
    assert model.waveform_out is not None
    with torch.no_grad():
        waveform_block = model.waveform_blocks[0]
        assert isinstance(waveform_block, WaveformTransformerBlock)
        bias = waveform_block.semantic_mod.bias.view(4, 6, 8)
        bias[:, 2, :] = 1.0
        model.waveform_out.weight.fill_(0.05)
        model.waveform_out.bias.zero_()

    zt = torch.zeros(1, 2, 8, 3)
    zc_a = torch.zeros(1, 2, 8, 3)
    zc_b = torch.ones(1, 2, 8, 3)
    semantic = torch.zeros(1, 3, model.hidden_dim)
    time_context = torch.zeros(1, model.hidden_dim)
    pos = torch.zeros(3, model.hidden_dim)
    coarse = torch.zeros(1, 3, model.target_channels * model.patch_size)
    mask = torch.ones(1, 3, dtype=torch.bool)

    out_a = model._waveform_refinement(
        zt=zt,
        z_cond=zc_a,
        semantic_tokens=semantic,
        time_context=time_context,
        memory_tokens=None,
        pos_embed=pos,
        coarse_tokens=coarse,
        frame_pad_mask=None,
        frame_keep_mask=mask,
        rope_self=None,
        rope_frames=None,
    )
    out_b = model._waveform_refinement(
        zt=zt,
        z_cond=zc_b,
        semantic_tokens=semantic,
        time_context=time_context,
        memory_tokens=None,
        pos_embed=pos,
        coarse_tokens=coarse,
        frame_pad_mask=None,
        frame_keep_mask=mask,
        rope_self=None,
        rope_frames=None,
    )

    assert not torch.allclose(out_a, out_b)


def test_waveform_level_branch_refines_coarse_prediction_microtokens() -> None:
    torch.manual_seed(123)
    model = _build_model(waveform_level_depth=1)
    assert model.waveform_out is not None
    with torch.no_grad():
        waveform_block = model.waveform_blocks[0]
        assert isinstance(waveform_block, WaveformTransformerBlock)
        bias = waveform_block.semantic_mod.bias.view(4, 6, 8)
        bias[:, 2, :] = 1.0
        model.waveform_out.weight.fill_(0.05)
        model.waveform_out.bias.zero_()

    zt = torch.zeros(1, 2, 8, 3)
    zc = torch.zeros(1, 2, 8, 3)
    semantic = torch.zeros(1, 3, model.hidden_dim)
    time_context = torch.zeros(1, model.hidden_dim)
    pos = torch.zeros(3, model.hidden_dim)
    coarse_a = torch.zeros(1, 3, model.target_channels * model.patch_size)
    coarse_b = torch.ones(1, 3, model.target_channels * model.patch_size)
    mask = torch.ones(1, 3, dtype=torch.bool)

    out_a = model._waveform_refinement(
        zt=zt,
        z_cond=zc,
        semantic_tokens=semantic,
        time_context=time_context,
        memory_tokens=None,
        pos_embed=pos,
        coarse_tokens=coarse_a,
        frame_pad_mask=None,
        frame_keep_mask=mask,
        rope_self=None,
        rope_frames=None,
    )
    out_b = model._waveform_refinement(
        zt=zt,
        z_cond=zc,
        semantic_tokens=semantic,
        time_context=time_context,
        memory_tokens=None,
        pos_embed=pos,
        coarse_tokens=coarse_b,
        frame_pad_mask=None,
        frame_keep_mask=mask,
        rope_self=None,
        rope_frames=None,
    )

    residual_a = out_a - coarse_a
    residual_b = out_b - coarse_b
    assert not torch.allclose(residual_a, residual_b)


def test_waveform_level_conditioning_receives_gradient_when_active() -> None:
    torch.manual_seed(123)
    model = _build_model(waveform_level_depth=1)
    assert model.waveform_out is not None
    assert model.waveform_cond_in is not None
    with torch.no_grad():
        waveform_block = model.waveform_blocks[0]
        assert isinstance(waveform_block, WaveformTransformerBlock)
        bias = waveform_block.semantic_mod.bias.view(4, 6, 8)
        bias[:, 2, :] = 1.0
        model.waveform_out.weight.fill_(0.05)
        model.waveform_out.bias.zero_()

    zt = torch.randn(2, 2, 8, 5)
    zc = torch.randn(2, 2, 8, 5)
    mask = torch.ones(2, 5, dtype=torch.bool)
    t = torch.tensor([0.25, 0.75])

    loss = model(zt=zt, t=t, z_cond=zc, valid_mask=mask).square().mean()
    loss.backward()

    cond_grad = model.waveform_cond_in.weight.grad
    cond_film_grad = waveform_block.cond_mod.weight.grad
    assert cond_grad is not None
    assert cond_film_grad is not None
    assert torch.isfinite(cond_grad).all()
    assert torch.isfinite(cond_film_grad).all()
    assert cond_grad.abs().sum().item() > 0.0
    assert cond_film_grad.abs().sum().item() > 0.0


def test_waveform_block_attends_to_memory_tokens() -> None:
    torch.manual_seed(123)
    block = WaveformTransformerBlock(
        semantic_dim=16,
        waveform_hidden_dim=8,
        num_micro_tokens=4,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    with torch.no_grad():
        bias = block.semantic_mod.bias.view(4, 6, 8)
        bias[:, 2, :] = 1.0

    waveform_tokens = torch.randn(1, 3, 4, 8)
    semantic_tokens = torch.zeros(1, 3, 16)
    memory_a = torch.zeros(1, 2, 16)
    memory_b = torch.randn(1, 2, 16)

    out_a = block(
        waveform_tokens=waveform_tokens,
        semantic_tokens=semantic_tokens,
        memory_tokens=memory_a,
    )
    out_b = block(
        waveform_tokens=waveform_tokens,
        semantic_tokens=semantic_tokens,
        memory_tokens=memory_b,
    )

    assert not torch.allclose(out_a, out_b)


def test_waveform_level_requires_divisible_micro_patch_size() -> None:
    with pytest.raises(ValueError, match="divisible"):
        SpatialDiT(
            target_channels=2,
            cond_channels=2,
            patch_size=10,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            timestep_embed_dim=16,
            timestep_scale=1000.0,
            max_period=10000.0,
            waveform_level_depth=1,
            waveform_micro_patch_size=4,
            waveform_hidden_dim=8,
            waveform_num_heads=4,
        )


def test_spatial_dit_requires_hidden_dim_divisible_by_num_heads() -> None:
    with pytest.raises(ValueError, match="num_heads"):
        SpatialDiT(
            target_channels=2,
            cond_channels=2,
            patch_size=8,
            hidden_dim=16,
            num_layers=1,
            num_heads=3,
            mlp_ratio=2.0,
            dropout=0.0,
            timestep_embed_dim=16,
            timestep_scale=1000.0,
            max_period=10000.0,
        )
