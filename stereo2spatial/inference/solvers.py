"""Solver-controller abstractions for batched inference scheduling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch

SolverName = str
CLEAN_PREDICTION_EPS = 1.0e-4
INTEGRATION_T_END = 1.0 - CLEAN_PREDICTION_EPS


@dataclass(frozen=True)
class ModelQuery:
    """One model-forward request produced by a solver controller."""

    request_id: str
    window_index: int
    query_index: int
    zt: torch.Tensor
    z_cond: torch.Tensor
    valid_mask: torch.Tensor | None
    t_value: float
    mem: torch.Tensor | None = None
    mix_style: torch.Tensor | None = None
    amplitude_gain: torch.Tensor | None = None
    return_mem: bool = False
    conditioning_cache: dict[str, object] | None = None


class SolverController(Protocol):
    """State machine that integrates clean predictions into an accepted state."""

    def next_query(self) -> ModelQuery | None:
        """Return the next model query needed by this controller."""
        ...

    def accept_output(
        self,
        clean_prediction: torch.Tensor,
        mem_out: torch.Tensor | None = None,
    ) -> None:
        """Consume one model output for the most recent query."""
        ...

    @property
    def is_done(self) -> bool:
        """Whether the controller has completed its integration state."""
        ...

    def result(self) -> torch.Tensor:
        """Return the final accepted clean prediction."""
        ...


def clean_prediction_to_velocity(
    clean_prediction: torch.Tensor,
    z_state: torch.Tensor,
    t_value: float,
) -> torch.Tensor:
    """Convert a clean endpoint prediction into rectified-flow velocity."""
    denom = max(1.0 - float(t_value), CLEAN_PREDICTION_EPS)
    return (clean_prediction - z_state) / denom


def res6s_tableau(
    step_size: float,
) -> tuple[list[float], list[list[float]], list[float]]:
    """Return a consistent six-stage RK tableau for the ``res6s`` API.

    The historical implementation copied an exponential-integrator tableau but
    applied it as a generic explicit Runge--Kutta method. Its weights summed to
    ``phi_1(-h)`` rather than one, so even a constant velocity field was integrated
    incorrectly. ``res6s`` now composes two Bogacki--Shampine RK3 half-steps into
    one six-stage third-order step. This is valid for a generic rectified-flow ODE,
    and its largest stage time is ``7/8`` so clean-to-velocity conversion never
    probes the ill-conditioned end of a step.

    ``step_size`` remains part of the public API even though an ordinary RK
    tableau is step-size independent.
    """
    if float(step_size) <= 0.0:
        raise ValueError("step_size must be > 0")

    c = [0.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 2.0, 3.0 / 4.0, 7.0 / 8.0]
    a = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 3.0 / 8.0, 0.0, 0.0, 0.0, 0.0],
        [1.0 / 9.0, 1.0 / 6.0, 2.0 / 9.0, 0.0, 0.0, 0.0],
        [1.0 / 9.0, 1.0 / 6.0, 2.0 / 9.0, 1.0 / 4.0, 0.0, 0.0],
        [
            1.0 / 9.0,
            1.0 / 6.0,
            2.0 / 9.0,
            0.0,
            3.0 / 8.0,
            0.0,
        ],
    ]
    b = [1.0 / 9.0, 1.0 / 6.0, 2.0 / 9.0, 1.0 / 9.0, 1.0 / 6.0, 2.0 / 9.0]
    return c, a, b


def weighted_sum(weights: list[float], values: list[torch.Tensor]) -> torch.Tensor:
    """Return a tensor weighted sum with the same shape as ``values[0]``."""
    result = torch.zeros_like(values[0])
    for weight, value in zip(weights, values):
        if weight != 0.0:
            result = result + float(weight) * value
    return result


def stack_model_queries(queries: list[ModelQuery]) -> dict[str, Any]:
    """Stack compatible singleton-batch model queries into one model call."""
    if not queries:
        raise ValueError("queries cannot be empty")

    first = queries[0]
    return_mem = first.return_mem

    def _require_same_optional(name: str) -> None:
        first_value = getattr(first, name)
        first_present = first_value is not None
        first_shape = None if first_value is None else tuple(first_value.shape)
        for query in queries[1:]:
            value = getattr(query, name)
            if (value is not None) != first_present:
                raise ValueError(f"all queries must agree on {name} presence")
            if value is not None and tuple(value.shape) != first_shape:
                raise ValueError(f"all queries must share {name} shape")

    for query in queries:
        if query.return_mem != return_mem:
            raise ValueError("all queries in one model batch must agree on return_mem")
        if query.zt.shape[0] != 1:
            raise ValueError("query.zt must include a singleton batch dimension")
        if query.z_cond.shape[0] != 1:
            raise ValueError("query.z_cond must include a singleton batch dimension")
        if query.valid_mask is not None and query.valid_mask.shape[0] != 1:
            raise ValueError(
                "query.valid_mask must include a singleton batch dimension"
            )
        if query.zt.shape[2:] != first.zt.shape[2:]:
            raise ValueError("all queries must share target patch/window shape")
        if query.z_cond.shape[2:] != first.z_cond.shape[2:]:
            raise ValueError("all queries must share conditioning patch/window shape")
        if (query.valid_mask is not None) != (first.valid_mask is not None):
            raise ValueError("all queries must agree on valid_mask presence")
        if (
            query.valid_mask is not None
            and first.valid_mask is not None
            and query.valid_mask.shape[1:] != first.valid_mask.shape[1:]
        ):
            raise ValueError("all queries must share valid_mask shape")

    _require_same_optional("mem")
    _require_same_optional("mix_style")
    _require_same_optional("amplitude_gain")
    first_cache_present = first.conditioning_cache is not None
    if any(
        (query.conditioning_cache is not None) != first_cache_present
        for query in queries
    ):
        raise ValueError("all queries must agree on conditioning_cache presence")

    batch: dict[str, object] = {
        "zt": torch.cat([query.zt for query in queries], dim=0),
        "z_cond": torch.cat([query.z_cond for query in queries], dim=0),
        "t": torch.tensor(
            [float(query.t_value) for query in queries],
            # Time remains float32 even when model activations use fp16/bf16 so
            # values just below one do not round to the singular endpoint.
            dtype=torch.float32,
            device=first.zt.device,
        ),
    }
    if first.valid_mask is not None:
        batch["valid_mask"] = torch.cat(
            [query.valid_mask for query in queries if query.valid_mask is not None],
            dim=0,
        )
    if first.mem is not None:
        batch["mem"] = torch.cat(
            [query.mem for query in queries if query.mem is not None],
            dim=0,
        )
    if first.mix_style is not None:
        batch["mix_style"] = torch.cat(
            [query.mix_style for query in queries if query.mix_style is not None],
            dim=0,
        )
    if first.amplitude_gain is not None:
        batch["amplitude_gain"] = torch.cat(
            [
                query.amplitude_gain
                for query in queries
                if query.amplitude_gain is not None
            ],
            dim=0,
        )
    if first.conditioning_cache is not None:
        batch["conditioning_cache"] = _stack_conditioning_caches(
            [
                query.conditioning_cache
                for query in queries
                if query.conditioning_cache is not None
            ]
        )
    return batch


def _stack_optional_tensors(values: list[object]) -> torch.Tensor | None:
    if values[0] is None:
        if any(value is not None for value in values):
            raise ValueError("all cached optional tensors must agree on presence")
        return None
    tensors = [cast(torch.Tensor, value) for value in values]
    return torch.cat(tensors, dim=0)


def _stack_optional_rope(
    values: list[object],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if values[0] is None:
        if any(value is not None for value in values):
            raise ValueError("all cached RoPE values must agree on presence")
        return None
    first = cast(tuple[torch.Tensor, torch.Tensor], values[0])
    for value in values[1:]:
        current = cast(tuple[torch.Tensor, torch.Tensor], value)
        if current[0].shape != first[0].shape or current[1].shape != first[1].shape:
            raise ValueError("all cached RoPE values must share shape")
    return first


def _stack_tensor_dicts(
    items: Sequence[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    keys = set(items[0])
    if any(set(item) != keys for item in items):
        raise ValueError("all cached tensor dicts must share keys")
    return {
        key: torch.cat([item[key] for item in items], dim=0) for key in sorted(keys)
    }


def _stack_conditioning_caches(
    caches: list[dict[str, object]],
) -> dict[str, object]:
    if not caches:
        raise ValueError("conditioning caches cannot be empty")
    result: dict[str, object] = {
        "cond_tokens": torch.cat(
            [cast(torch.Tensor, cache["cond_tokens"]) for cache in caches],
            dim=0,
        ),
        "pos": cast(torch.Tensor, caches[0]["pos"]),
        "frame_pad_mask": _stack_optional_tensors(
            [cache["frame_pad_mask"] for cache in caches]
        ),
        "frame_keep_mask": _stack_optional_tensors(
            [cache["frame_keep_mask"] for cache in caches]
        ),
        "rope_x": _stack_optional_rope([cache["rope_x"] for cache in caches]),
        "rope_frames": _stack_optional_rope([cache["rope_frames"] for cache in caches]),
        "transformer_caches": [
            _stack_tensor_dicts(layer_items)
            for layer_items in zip(
                *[
                    cast(list[dict[str, torch.Tensor]], cache["transformer_caches"])
                    for cache in caches
                ]
            )
        ],
    }
    if "waveform_cond_tokens" in caches[0]:
        result["waveform_cond_tokens"] = torch.cat(
            [cast(torch.Tensor, cache["waveform_cond_tokens"]) for cache in caches],
            dim=0,
        )
        result["waveform_rope_self"] = _stack_optional_rope(
            [cache["waveform_rope_self"] for cache in caches]
        )
        result["waveform_rope_frames"] = _stack_optional_rope(
            [cache["waveform_rope_frames"] for cache in caches]
        )
        result["waveform_block_caches"] = [
            _stack_tensor_dicts(layer_items)
            for layer_items in zip(
                *[
                    cast(list[dict[str, torch.Tensor]], cache["waveform_block_caches"])
                    for cache in caches
                ]
            )
        ]
    return result


class FixedStepSolverController:
    """Query-driven controller for fixed-step clean-prediction solvers."""

    def __init__(
        self,
        *,
        request_id: str,
        window_index: int,
        solver: SolverName,
        solver_steps: int,
        z0_chunk: torch.Tensor,
        z_cond: torch.Tensor,
        valid_mask: torch.Tensor | None,
        mem: torch.Tensor | None = None,
        mix_style: torch.Tensor | None = None,
        amplitude_gain: torch.Tensor | None = None,
        conditioning_cache: dict[str, object] | None = None,
    ) -> None:
        if solver_steps <= 0:
            raise ValueError("solver_steps must be > 0")
        if z0_chunk.shape[0] != 1:
            raise ValueError("z0_chunk must include a singleton batch dimension")
        if z_cond.shape[0] != 1:
            raise ValueError("z_cond must include a singleton batch dimension")
        if valid_mask is not None and valid_mask.shape[0] != 1:
            raise ValueError("valid_mask must include a singleton batch dimension")
        if z0_chunk.shape[-1] != z_cond.shape[-1]:
            raise ValueError("z0_chunk and z_cond must share window frames")
        if valid_mask is not None and valid_mask.shape[-1] != z_cond.shape[-1]:
            raise ValueError("valid_mask must share window frames")

        solver_name = str(solver).strip().lower()
        if solver_name == "res_6s":
            solver_name = "res6s"
        if solver_name in {"midpoint", "midpoint-rk2", "rk2"}:
            solver_name = "midpoint_rk2"
        if solver_name not in {"euler", "heun", "midpoint_rk2", "res6s"}:
            raise ValueError(
                "FixedStepSolverController supports euler, heun, midpoint_rk2, res6s"
            )

        self.request_id = str(request_id)
        self.window_index = int(window_index)
        self.solver = solver_name
        self.solver_steps = int(solver_steps)
        self.z_cond = z_cond
        self.valid_mask = valid_mask
        self.mem = mem
        self.mix_style = mix_style
        self.amplitude_gain = amplitude_gain
        self.conditioning_cache = conditioning_cache
        self.dt = INTEGRATION_T_END / float(self.solver_steps)
        self.z_state = z0_chunk
        self.step_idx = 0
        self.query_index = 0
        self._result: torch.Tensor | None = None
        self.mem_out: torch.Tensor | None = None
        self._phase = "velocity"
        self._last_query: ModelQuery | None = None

        self._heun_z_start: torch.Tensor | None = None
        self._heun_v0: torch.Tensor | None = None
        self._midpoint_z_start: torch.Tensor | None = None
        self._midpoint_v0: torch.Tensor | None = None
        self._res_stage_idx = 0
        self._res_step_start = z0_chunk
        self._res_velocities: list[torch.Tensor] = []
        if self.solver == "res6s":
            self._res_c, self._res_a, self._res_b = res6s_tableau(self.dt)
        else:
            self._res_c, self._res_a, self._res_b = [], [], []

    @property
    def is_done(self) -> bool:
        return self._phase == "done"

    def result(self) -> torch.Tensor:
        if self._result is None:
            raise RuntimeError("solver controller is not done")
        return self._result

    def next_query(self) -> ModelQuery | None:
        if self.is_done:
            return None
        if self._phase == "final_clean":
            query = self._make_query(
                zt=self.z_state,
                t_value=INTEGRATION_T_END,
                return_mem=self.mem is not None,
            )
        elif self.solver == "euler":
            query = self._make_query(
                zt=self.z_state,
                t_value=float(self.step_idx) * self.dt,
                return_mem=False,
            )
        elif self.solver == "heun":
            query = self._next_heun_query()
        elif self.solver == "midpoint_rk2":
            query = self._next_midpoint_query()
        elif self.solver == "res6s":
            query = self._next_res6s_query()
        else:
            raise RuntimeError(f"Unsupported solver state: {self.solver}")

        self._last_query = query
        self.query_index += 1
        return query

    def accept_output(
        self,
        clean_prediction: torch.Tensor,
        mem_out: torch.Tensor | None = None,
    ) -> None:
        if self._last_query is None:
            raise RuntimeError("accept_output called before next_query")
        query = self._last_query

        if self._phase == "final_clean":
            self._result = clean_prediction.clone()
            if query.return_mem:
                if mem_out is None:
                    raise RuntimeError(
                        "final clean query requested memory but returned none"
                    )
                self.mem_out = mem_out.clone()
            self._phase = "done"
            self._last_query = None
            return

        if self.solver == "euler":
            velocity = clean_prediction_to_velocity(
                clean_prediction,
                self.z_state,
                query.t_value,
            )
            self.z_state = self.z_state + self.dt * velocity
            self._advance_or_finish_step()
        elif self.solver == "heun":
            self._accept_heun_output(clean_prediction, query)
        elif self.solver == "midpoint_rk2":
            self._accept_midpoint_output(clean_prediction, query)
        elif self.solver == "res6s":
            self._accept_res6s_output(clean_prediction, query)
        else:
            raise RuntimeError(f"Unsupported solver state: {self.solver}")
        self._last_query = None

    def _make_query(
        self,
        *,
        zt: torch.Tensor,
        t_value: float,
        return_mem: bool,
    ) -> ModelQuery:
        return ModelQuery(
            request_id=self.request_id,
            window_index=self.window_index,
            query_index=self.query_index,
            zt=zt,
            z_cond=self.z_cond,
            valid_mask=self.valid_mask,
            t_value=float(t_value),
            mem=self.mem,
            mix_style=self.mix_style,
            amplitude_gain=self.amplitude_gain,
            return_mem=bool(return_mem),
            conditioning_cache=self.conditioning_cache,
        )

    def _advance_or_finish_step(self) -> None:
        self.step_idx += 1
        if self.step_idx >= self.solver_steps:
            self._phase = "final_clean"

    def _next_heun_query(self) -> ModelQuery:
        if self._phase == "velocity":
            self._heun_z_start = self.z_state
            return self._make_query(
                zt=self.z_state,
                t_value=float(self.step_idx) * self.dt,
                return_mem=False,
            )
        if self._phase == "heun_v1":
            if self._heun_z_start is None or self._heun_v0 is None:
                raise RuntimeError("heun_v1 requested before v0")
            terminal_step = self.step_idx == self.solver_steps - 1
            stage_fraction = 0.5 if terminal_step else 1.0
            z_euler = self._heun_z_start + stage_fraction * self.dt * self._heun_v0
            return self._make_query(
                zt=z_euler,
                t_value=(float(self.step_idx) + stage_fraction) * self.dt,
                return_mem=False,
            )
        raise RuntimeError(f"Unexpected heun phase: {self._phase}")

    def _accept_heun_output(
        self,
        clean_prediction: torch.Tensor,
        query: ModelQuery,
    ) -> None:
        if self._phase == "velocity":
            self._heun_z_start = self.z_state
            self._heun_v0 = clean_prediction_to_velocity(
                clean_prediction,
                self.z_state,
                query.t_value,
            )
            self._phase = "heun_v1"
            return

        if self._phase != "heun_v1":
            raise RuntimeError(f"Unexpected heun phase: {self._phase}")
        if self._heun_z_start is None or self._heun_v0 is None:
            raise RuntimeError("heun_v1 accepted before v0")
        v1 = clean_prediction_to_velocity(
            clean_prediction,
            query.zt,
            query.t_value,
        )
        if self.step_idx == self.solver_steps - 1:
            # The clean-to-velocity conversion is ill-conditioned as t -> 1.
            # Use midpoint RK2 for the terminal step so the final velocity probe
            # stays half a step away while preserving second-order accuracy.
            self.z_state = self._heun_z_start + self.dt * v1
        else:
            self.z_state = self._heun_z_start + 0.5 * self.dt * (self._heun_v0 + v1)
        self._heun_z_start = None
        self._heun_v0 = None
        self._phase = "velocity"
        self._advance_or_finish_step()

    def _next_midpoint_query(self) -> ModelQuery:
        if self._phase == "velocity":
            self._midpoint_z_start = self.z_state
            return self._make_query(
                zt=self.z_state,
                t_value=float(self.step_idx) * self.dt,
                return_mem=False,
            )
        if self._phase == "midpoint_vmid":
            if self._midpoint_z_start is None or self._midpoint_v0 is None:
                raise RuntimeError("midpoint_vmid requested before v0")
            z_mid = self._midpoint_z_start + 0.5 * self.dt * self._midpoint_v0
            return self._make_query(
                zt=z_mid,
                t_value=min(
                    (float(self.step_idx) + 0.5) * self.dt,
                    INTEGRATION_T_END,
                ),
                return_mem=False,
            )
        raise RuntimeError(f"Unexpected midpoint phase: {self._phase}")

    def _accept_midpoint_output(
        self,
        clean_prediction: torch.Tensor,
        query: ModelQuery,
    ) -> None:
        if self._phase == "velocity":
            self._midpoint_z_start = self.z_state
            self._midpoint_v0 = clean_prediction_to_velocity(
                clean_prediction,
                self.z_state,
                query.t_value,
            )
            self._phase = "midpoint_vmid"
            return

        if self._phase != "midpoint_vmid":
            raise RuntimeError(f"Unexpected midpoint phase: {self._phase}")
        if self._midpoint_z_start is None or self._midpoint_v0 is None:
            raise RuntimeError("midpoint_vmid accepted before v0")
        v_mid = clean_prediction_to_velocity(
            clean_prediction,
            query.zt,
            query.t_value,
        )
        self.z_state = self._midpoint_z_start + self.dt * v_mid
        self._midpoint_z_start = None
        self._midpoint_v0 = None
        self._phase = "velocity"
        self._advance_or_finish_step()

    def _next_res6s_query(self) -> ModelQuery:
        t_base = float(self.step_idx) * self.dt
        stage_idx = self._res_stage_idx
        if stage_idx == 0:
            z_stage = self._res_step_start
        else:
            z_stage = self._res_step_start + self.dt * weighted_sum(
                self._res_a[stage_idx][:stage_idx],
                self._res_velocities,
            )
        t_stage = min(t_base + self._res_c[stage_idx] * self.dt, INTEGRATION_T_END)
        return self._make_query(
            zt=z_stage,
            t_value=t_stage,
            return_mem=False,
        )

    def _accept_res6s_output(
        self,
        clean_prediction: torch.Tensor,
        query: ModelQuery,
    ) -> None:
        velocity = clean_prediction_to_velocity(
            clean_prediction,
            query.zt,
            query.t_value,
        )
        self._res_velocities.append(velocity)
        if self._res_stage_idx < len(self._res_c) - 1:
            self._res_stage_idx += 1
            return

        self.z_state = self._res_step_start + self.dt * weighted_sum(
            self._res_b,
            self._res_velocities,
        )
        self.step_idx += 1
        self._res_stage_idx = 0
        self._res_velocities = []
        self._res_step_start = self.z_state
        if self.step_idx >= self.solver_steps:
            self._phase = "final_clean"
