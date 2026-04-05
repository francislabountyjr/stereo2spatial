"""PatchGAN discriminators and GAN loss primitives for latent-space training."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sn(module: nn.Module) -> nn.Module:
    """Apply spectral normalization to a module."""
    return nn.utils.spectral_norm(module)


class PatchDiscriminator2D(nn.Module):
    """
    Latent-space PatchGAN discriminator.

    Input: [B, C, F, T] where C = cond_channels + target_channels.
    Output: [B, 1, F', T'] patch-level logits.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        num_layers: int = 4,
        use_spectral_norm: bool = True,
        downsample_freq: bool = False,
    ) -> None:
        super().__init__()
        conv_ctor = (
            (lambda *args, **kwargs: _sn(nn.Conv2d(*args, **kwargs)))
            if use_spectral_norm
            else nn.Conv2d
        )

        layers: list[nn.Module] = []
        channels = int(base_channels)
        layers.extend(
            [
                conv_ctor(
                    int(in_channels),
                    channels,
                    kernel_size=(3, 7),
                    stride=(1, 2),
                    padding=(1, 3),
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )

        for _ in range(int(num_layers)):
            next_channels = min(channels * 2, 512)
            stride = (2, 2) if downsample_freq else (1, 2)
            layers.extend(
                [
                    conv_ctor(
                        channels,
                        next_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            channels = next_channels

        layers.append(
            conv_ctor(channels, 1, kernel_size=3, stride=1, padding=1)
        )
        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute patch-level discriminator logits for one latent batch."""
        return cast(torch.Tensor, self.net(x))


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale latent-space discriminator.

    - fine: time-only downsampling for local/detail texture.
    - coarse: frequency+time downsampling for broader structure.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        fine_layers: int = 3,
        coarse_layers: int = 4,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.fine = PatchDiscriminator2D(
            in_channels=in_channels,
            base_channels=base_channels,
            num_layers=fine_layers,
            use_spectral_norm=use_spectral_norm,
            downsample_freq=False,
        )
        self.coarse = PatchDiscriminator2D(
            in_channels=in_channels,
            base_channels=base_channels,
            num_layers=coarse_layers,
            use_spectral_norm=use_spectral_norm,
            downsample_freq=True,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return fine/coarse discriminator outputs for the same latent batch."""
        return {
            "fine": self.fine(x),
            "coarse": self.coarse(x),
        }


def d_hinge_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Discriminator hinge objective on real and fake logits."""
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()


def g_hinge_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """Generator hinge objective against discriminator fake logits."""
    return (-d_fake).mean()


def r1_penalty(d_out: torch.Tensor, x_in: torch.Tensor) -> torch.Tensor:
    """R1 gradient penalty for discriminator regularization."""
    grad = torch.autograd.grad(
        outputs=d_out.sum(),
        inputs=x_in,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad.pow(2).reshape(grad.shape[0], -1).sum(dim=1).mean()


@torch.no_grad()
def set_requires_grad(module: nn.Module, flag: bool) -> None:
    """Enable or disable gradient tracking for all parameters in a module."""
    for parameter in module.parameters():
        parameter.requires_grad_(bool(flag))
