from __future__ import annotations

import torch

from stereo2spatial.common.checkpoints import adapt_state_dict_keys_for_model


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)


def test_adapt_state_dict_keys_strips_module_prefix() -> None:
    model = _Tiny()
    prefixed = {
        f"module.{name}": tensor.clone()
        for name, tensor in model.state_dict().items()
    }

    adapted = adapt_state_dict_keys_for_model(model, prefixed)

    assert set(adapted.keys()) == set(model.state_dict().keys())


def test_adapt_state_dict_keys_returns_original_when_already_matching() -> None:
    model = _Tiny()
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    adapted = adapt_state_dict_keys_for_model(model, state)

    assert adapted == state
