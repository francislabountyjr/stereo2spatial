"""Dynamic model-query batching for inference solver controllers."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch

from stereo2spatial.inference.cuda_graphs import CudaGraphModelRunner
from stereo2spatial.inference.solvers import (
    ModelQuery,
    SolverController,
    stack_model_queries,
)


@dataclass(frozen=True)
class SchedulerStats:
    """Summary of one dynamic scheduler run."""

    completed_controllers: int
    model_batches: int
    model_queries: int
    max_observed_batch_size: int
    max_observed_active_controllers: int
    batch_size_counts: dict[int, int] = field(default_factory=dict)

    @property
    def average_batch_size(self) -> float:
        """Average model-query batch size over executed model forwards."""
        if self.model_batches <= 0:
            return 0.0
        return float(self.model_queries) / float(self.model_batches)


def model_query_compatibility_key(query: ModelQuery) -> tuple[Any, ...]:
    """Return the model-forward compatibility key for one query."""

    def _optional_shape(value: torch.Tensor | None) -> tuple[int, ...] | None:
        return None if value is None else tuple(value.shape[1:])

    device = query.zt.device
    return (
        tuple(query.zt.shape[1:]),
        tuple(query.z_cond.shape[1:]),
        _optional_shape(query.valid_mask),
        query.zt.dtype,
        device.type,
        device.index,
        _optional_shape(query.mem),
        _optional_shape(query.mix_style),
        _optional_shape(query.amplitude_gain),
        query.conditioning_cache is not None,
        query.return_mem,
    )


def group_compatible_queries(
    queries: Iterable[ModelQuery],
    *,
    max_batch_size: int,
) -> list[list[ModelQuery]]:
    """Group compatible queries into batches up to ``max_batch_size``."""
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be > 0")

    grouped: dict[tuple[Any, ...], list[ModelQuery]] = {}
    ordered_keys: list[tuple[Any, ...]] = []
    for query in queries:
        key = model_query_compatibility_key(query)
        if key not in grouped:
            grouped[key] = []
            ordered_keys.append(key)
        grouped[key].append(query)

    batches: list[list[ModelQuery]] = []
    for key in ordered_keys:
        group = grouped[key]
        for start in range(0, len(group), max_batch_size):
            batches.append(group[start : start + max_batch_size])
    return batches


@torch.inference_mode()
def run_dynamic_controller_scheduler(
    *,
    controllers: Iterable[SolverController],
    model: torch.nn.Module,
    max_batch_size: int,
    max_active_controllers: int | None = None,
    cuda_graph_runner: CudaGraphModelRunner | None = None,
) -> SchedulerStats:
    """Run solver controllers to completion using batched model-forward calls."""
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be > 0")
    if max_active_controllers is not None and max_active_controllers <= 0:
        raise ValueError("max_active_controllers must be > 0 when set")

    pending = deque(controllers)
    active: list[SolverController] = []
    completed = 0
    model_batches = 0
    model_queries = 0
    max_observed_batch_size = 0
    max_observed_active = 0
    batch_size_counts: dict[int, int] = {}

    def fill_active() -> None:
        nonlocal max_observed_active
        limit = max_active_controllers if max_active_controllers is not None else None
        while pending and (limit is None or len(active) < limit):
            active.append(pending.popleft())
        max_observed_active = max(max_observed_active, len(active))

    fill_active()
    while active:
        queries_by_controller: list[tuple[SolverController, ModelQuery]] = []
        for controller in active:
            if controller.is_done:
                continue
            query = controller.next_query()
            if query is None:
                raise RuntimeError("active controller returned no query before completion")
            queries_by_controller.append((controller, query))

        if not queries_by_controller:
            raise RuntimeError("scheduler made no progress")

        query_to_controller = {
            id(query): controller for controller, query in queries_by_controller
        }
        for batch_queries in group_compatible_queries(
            [query for _, query in queries_by_controller],
            max_batch_size=max_batch_size,
        ):
            batch = stack_model_queries(batch_queries)
            kwargs: dict[str, Any] = {
                "zt": batch["zt"],
                "t": batch["t"],
                "z_cond": batch["z_cond"],
                "valid_mask": batch.get("valid_mask"),
            }
            if "mem" in batch:
                kwargs["mem"] = batch["mem"]
            if "mix_style" in batch:
                kwargs["mix_style"] = batch["mix_style"]
            if "amplitude_gain" in batch:
                kwargs["amplitude_gain"] = batch["amplitude_gain"]
            if "conditioning_cache" in batch:
                kwargs["conditioning_cache"] = batch["conditioning_cache"]
            if batch_queries[0].return_mem:
                kwargs["return_mem"] = True

            output = (
                cuda_graph_runner.run(kwargs, actual_batch_size=len(batch_queries))
                if cuda_graph_runner is not None
                else model(**kwargs)
            )
            mem_batch = None
            if batch_queries[0].return_mem:
                if not isinstance(output, tuple) or len(output) != 2:
                    raise TypeError("return_mem model query must return (prediction, mem)")
                prediction_batch, mem_batch = output
            else:
                prediction_batch = output
            if not isinstance(prediction_batch, torch.Tensor):
                raise TypeError("model must return a tensor prediction")

            model_batches += 1
            model_queries += len(batch_queries)
            max_observed_batch_size = max(max_observed_batch_size, len(batch_queries))
            batch_size_counts[len(batch_queries)] = (
                batch_size_counts.get(len(batch_queries), 0) + 1
            )

            for item_idx, query in enumerate(batch_queries):
                controller = query_to_controller[id(query)]
                mem_out = (
                    None
                    if mem_batch is None
                    else mem_batch[item_idx : item_idx + 1].contiguous()
                )
                controller.accept_output(
                    prediction_batch[item_idx : item_idx + 1].contiguous(),
                    mem_out=mem_out,
                )

        still_active: list[SolverController] = []
        for controller in active:
            if controller.is_done:
                completed += 1
            else:
                still_active.append(controller)
        active = still_active
        fill_active()

    return SchedulerStats(
        completed_controllers=completed,
        model_batches=model_batches,
        model_queries=model_queries,
        max_observed_batch_size=max_observed_batch_size,
        max_observed_active_controllers=max_observed_active,
        batch_size_counts=batch_size_counts,
    )
