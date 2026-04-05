"""GAN step helpers for the training loop."""

from __future__ import annotations

import torch
from accelerate import Accelerator

from .discriminator import (
    MultiScaleDiscriminator,
    d_hinge_loss,
    g_hinge_loss,
    r1_penalty,
    set_requires_grad,
)
from .loss_terms import _channel_correlation_l1_loss, _channel_routing_kl_loss


def compute_channel_aux_losses(
    *,
    gan_aux: dict[str, torch.Tensor] | None,
    routing_kl_weight: float,
    routing_kl_temperature: float,
    routing_kl_eps: float,
    corr_weight: float,
    corr_eps: float,
    corr_offdiag_only: bool,
    corr_use_correlation: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute optional channel-structure auxiliary losses from cached GAN tensors."""
    if gan_aux is None:
        raise RuntimeError(
            "Channel routing/correlation losses were enabled but auxiliary tensors were not collected."
        )

    loss_route_step: torch.Tensor | None = None
    loss_corr_step: torch.Tensor | None = None
    with torch.no_grad():
        aux_pred = gan_aux["fake"].detach()
        aux_target = gan_aux["real"].detach()
        aux_mask = gan_aux["mask"][:, 0, :, :].detach()
        if routing_kl_weight > 0.0:
            loss_route_step = _channel_routing_kl_loss(
                prediction_x1=aux_pred,
                target_x1=aux_target,
                mask_dt=aux_mask,
                temperature=routing_kl_temperature,
                eps=routing_kl_eps,
            )
        if corr_weight > 0.0:
            loss_corr_step = _channel_correlation_l1_loss(
                prediction_x1=aux_pred,
                target_x1=aux_target,
                mask_dt=aux_mask,
                eps=corr_eps,
                offdiag_only=corr_offdiag_only,
                use_correlation=corr_use_correlation,
            )

    return loss_route_step, loss_corr_step


def run_gan_step(
    *,
    accelerator: Accelerator,
    discriminator: MultiScaleDiscriminator | torch.nn.Module | None,
    gan_aux: dict[str, torch.Tensor] | None,
    gan_use_mask_channel: bool,
    global_step: int,
    gan_ms_w_fine: float,
    gan_ms_w_coarse: float,
    gan_r1_gamma: float,
    gan_r1_every: int,
    gan_lambda_adv_max: float,
    gan_adv_warmup_steps: int,
    loss_fm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Run discriminator + generator adversarial loss backward passes.

    Returns:
      - total generator loss (flow matching + weighted adversarial term)
      - detached discriminator step loss
      - detached generator adversarial loss
      - adversarial lambda used for this step
    """
    if discriminator is None:
        raise RuntimeError("GAN enabled but discriminator is missing.")
    if gan_aux is None:
        raise RuntimeError("GAN enabled but GAN auxiliary tensors are missing.")

    cond_for_gan = gan_aux["cond"]
    real_for_gan = gan_aux["real"]
    fake_for_gan = gan_aux["fake"]
    if gan_use_mask_channel:
        mask_for_gan = gan_aux["mask"]
        real_disc_in = torch.cat([cond_for_gan, real_for_gan, mask_for_gan], dim=1)
        fake_disc_in_detached = torch.cat(
            [cond_for_gan, fake_for_gan.detach(), mask_for_gan], dim=1
        )
    else:
        real_disc_in = torch.cat([cond_for_gan, real_for_gan], dim=1)
        fake_disc_in_detached = torch.cat([cond_for_gan, fake_for_gan.detach()], dim=1)

    set_requires_grad(discriminator, True)

    outs_real = discriminator(real_disc_in)
    outs_fake = discriminator(fake_disc_in_detached)
    loss_d_fine = d_hinge_loss(outs_real["fine"], outs_fake["fine"])
    loss_d_coarse = d_hinge_loss(outs_real["coarse"], outs_fake["coarse"])
    loss_d_total = gan_ms_w_fine * loss_d_fine + gan_ms_w_coarse * loss_d_coarse
    if gan_r1_gamma > 0.0 and (global_step % gan_r1_every) == 0:
        with torch.autocast(
            device_type=real_disc_in.device.type,
            enabled=False,
        ):
            real_disc_in_r1 = real_disc_in.detach().float().requires_grad_(True)
            outs_real_r1 = discriminator(real_disc_in_r1)
            loss_r1 = (gan_r1_gamma * 0.5) * r1_penalty(
                outs_real_r1["coarse"], real_disc_in_r1
            )
        loss_d_total = loss_d_total + loss_r1
    accelerator.backward(loss_d_total)

    set_requires_grad(discriminator, False)
    if gan_use_mask_channel:
        mask_for_gan = gan_aux["mask"]
        fake_disc_in = torch.cat([cond_for_gan, fake_for_gan, mask_for_gan], dim=1)
    else:
        fake_disc_in = torch.cat([cond_for_gan, fake_for_gan], dim=1)

    outs_fake_for_g = discriminator(fake_disc_in)
    loss_adv_fine = g_hinge_loss(outs_fake_for_g["fine"])
    loss_adv_coarse = g_hinge_loss(outs_fake_for_g["coarse"])
    loss_adv = gan_ms_w_fine * loss_adv_fine + gan_ms_w_coarse * loss_adv_coarse
    gan_lambda_adv_step = gan_lambda_adv_max * min(
        1.0,
        float(global_step) / float(max(1, gan_adv_warmup_steps)),
    )
    loss = loss_fm + float(gan_lambda_adv_step) * loss_adv
    accelerator.backward(loss)
    return (
        loss,
        loss_d_total.detach(),
        loss_adv.detach(),
        gan_lambda_adv_step,
    )
