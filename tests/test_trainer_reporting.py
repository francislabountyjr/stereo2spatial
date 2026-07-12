from __future__ import annotations

import torch

from stereo2spatial.training.trainer_reporting import _build_log_postfix


def test_log_postfix_reports_dynamic_loss_terms_without_unused_route_corr() -> None:
    postfix = _build_log_postfix(
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=0.01),
        step_loss_value=1.5,
        avg_loss=1.25,
        t_eff=8,
        num_windows=2,
        use_gan=False,
        loss_d_step=None,
        loss_adv_step=None,
        gan_lambda_adv_step=0.0,
        loss_route_step=None,
        loss_corr_step=None,
        loss_terms_step={
            "charb": torch.tensor(0.5),
            "mrstft": torch.tensor(0.25),
        },
        avg_d_loss=None,
        avg_adv_loss=None,
        avg_route_loss=0.0,
        avg_corr_loss=0.0,
        avg_loss_terms={"charb": 0.4, "mrstft": 0.2},
    )

    assert postfix["charb_step"] == "0.500000"
    assert postfix["mrstft_step"] == "0.250000"
    assert postfix["charb_avg"] == "0.400000"
    assert postfix["mrstft_avg"] == "0.200000"
    assert "route_avg" not in postfix
    assert "corr_avg" not in postfix
