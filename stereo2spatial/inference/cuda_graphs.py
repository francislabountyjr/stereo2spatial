"""CUDA Graph replay helpers for fixed-shape inference model calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import torch

TensorTree: TypeAlias = (
    torch.Tensor
    | dict[str, "TensorTree"]
    | list["TensorTree"]
    | tuple["TensorTree", ...]
    | None
)


def _tensor_tree_key(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            tuple(value.shape),
            str(value.dtype),
            str(value.device),
        )
    if value is None:
        return ("none",)
    if isinstance(value, dict):
        return (
            "dict",
            tuple((key, _tensor_tree_key(value[key])) for key in sorted(value)),
        )
    if isinstance(value, list):
        return ("list", tuple(_tensor_tree_key(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_tensor_tree_key(item) for item in value))
    return ("constant", type(value).__name__, value)


def _empty_like_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return torch.empty_like(value)
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _empty_like_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_empty_like_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_empty_like_tree(item) for item in value)
    return value


def _copy_tree_(destination: Any, source: Any) -> None:
    if isinstance(destination, torch.Tensor):
        if not isinstance(source, torch.Tensor):
            raise TypeError("tensor graph input received non-tensor source")
        if destination.shape != source.shape:
            raise ValueError(
                "CUDA graph input shape changed: "
                f"expected {tuple(destination.shape)}, got {tuple(source.shape)}"
            )
        destination.copy_(source, non_blocking=True)
        return
    if destination is None:
        if source is not None:
            raise ValueError("CUDA graph input changed from None to non-None")
        return
    if isinstance(destination, dict):
        if not isinstance(source, dict) or destination.keys() != source.keys():
            raise ValueError("CUDA graph dict input structure changed")
        for key in destination:
            _copy_tree_(destination[key], source[key])
        return
    if isinstance(destination, list):
        if not isinstance(source, list) or len(destination) != len(source):
            raise ValueError("CUDA graph list input structure changed")
        for dest_item, source_item in zip(destination, source):
            _copy_tree_(dest_item, source_item)
        return
    if isinstance(destination, tuple):
        if not isinstance(source, tuple) or len(destination) != len(source):
            raise ValueError("CUDA graph tuple input structure changed")
        for dest_item, source_item in zip(destination, source):
            _copy_tree_(dest_item, source_item)
        return
    if destination != source:
        raise ValueError("CUDA graph constant input changed")


def _pad_batch_tensor(
    tensor: torch.Tensor,
    *,
    bucket_size: int,
    actual_batch_size: int,
) -> torch.Tensor:
    if tensor.shape[0] != actual_batch_size:
        return tensor
    if actual_batch_size == bucket_size:
        return tensor
    padded = torch.empty(
        (bucket_size, *tensor.shape[1:]),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    padded[:actual_batch_size].copy_(tensor, non_blocking=True)
    if actual_batch_size < bucket_size:
        padded[actual_batch_size:].zero_()
    return padded


def _pad_batch_tree(
    value: Any,
    *,
    bucket_size: int,
    actual_batch_size: int,
) -> Any:
    if isinstance(value, torch.Tensor):
        return _pad_batch_tensor(
            value,
            bucket_size=bucket_size,
            actual_batch_size=actual_batch_size,
        )
    if value is None:
        return None
    if isinstance(value, dict):
        return {
            key: _pad_batch_tree(
                item,
                bucket_size=bucket_size,
                actual_batch_size=actual_batch_size,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _pad_batch_tree(
                item,
                bucket_size=bucket_size,
                actual_batch_size=actual_batch_size,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _pad_batch_tree(
                item,
                bucket_size=bucket_size,
                actual_batch_size=actual_batch_size,
            )
            for item in value
        )
    return value


def pad_model_kwargs_to_bucket(
    kwargs: dict[str, Any],
    *,
    bucket_size: int,
    actual_batch_size: int,
) -> dict[str, Any]:
    """Return model kwargs padded along batch dimension to ``bucket_size``."""
    return {
        key: _pad_batch_tree(
            value,
            bucket_size=bucket_size,
            actual_batch_size=actual_batch_size,
        )
        for key, value in kwargs.items()
    }


def _slice_output_batch(value: Any, actual_batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value[:actual_batch_size].clone()
    if isinstance(value, tuple):
        return tuple(_slice_output_batch(item, actual_batch_size) for item in value)
    if isinstance(value, list):
        return [_slice_output_batch(item, actual_batch_size) for item in value]
    if isinstance(value, dict):
        return {
            key: _slice_output_batch(item, actual_batch_size)
            for key, item in value.items()
        }
    return value


def _view_output_batch(value: Any, actual_batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value[:actual_batch_size]
    if isinstance(value, tuple):
        return tuple(_view_output_batch(item, actual_batch_size) for item in value)
    if isinstance(value, list):
        return [_view_output_batch(item, actual_batch_size) for item in value]
    if isinstance(value, dict):
        return {
            key: _view_output_batch(item, actual_batch_size)
            for key, item in value.items()
        }
    return value


@dataclass
class _CapturedModelCall:
    graph: torch.cuda.CUDAGraph
    static_kwargs: dict[str, Any]
    static_output: Any


class CudaGraphModelRunner:
    """Replay fixed-shape inference model calls with CUDA Graphs.

    The runner buckets smaller dynamic batches up to configured graph sizes,
    copies inputs into static CUDA buffers, replays the captured graph, and
    returns cloned outputs for the real batch rows.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        bucket_sizes: tuple[int, ...],
        capture_warmup: int = 2,
        clone_outputs: bool = True,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Graph inference requires CUDA")
        buckets = tuple(sorted({int(size) for size in bucket_sizes if int(size) > 0}))
        if not buckets:
            raise ValueError("at least one positive CUDA graph bucket is required")
        self.model = model
        self.bucket_sizes = buckets
        self.capture_warmup = max(0, int(capture_warmup))
        self.clone_outputs = bool(clone_outputs)
        self._captured: dict[Any, _CapturedModelCall] = {}

    def bucket_for_batch(self, actual_batch_size: int) -> int:
        """Return the smallest configured bucket that can hold the batch."""
        actual = int(actual_batch_size)
        for bucket in self.bucket_sizes:
            if actual <= bucket:
                return bucket
        raise ValueError(
            f"batch size {actual} exceeds largest CUDA graph bucket {self.bucket_sizes[-1]}"
        )

    def run(
        self,
        kwargs: dict[str, Any],
        *,
        actual_batch_size: int,
    ) -> Any:
        """Replay or capture a CUDA graph for one fixed-shape model call."""
        bucket_size = self.bucket_for_batch(actual_batch_size)
        padded_kwargs = pad_model_kwargs_to_bucket(
            kwargs,
            bucket_size=bucket_size,
            actual_batch_size=int(actual_batch_size),
        )
        key = tuple((name, _tensor_tree_key(value)) for name, value in sorted(padded_kwargs.items()))
        captured = self._captured.get(key)
        if captured is None:
            captured = self._capture(padded_kwargs, key=key)
        else:
            for name, value in padded_kwargs.items():
                _copy_tree_(captured.static_kwargs[name], value)

        captured.graph.replay()
        if self.clone_outputs:
            return _slice_output_batch(captured.static_output, int(actual_batch_size))
        return _view_output_batch(captured.static_output, int(actual_batch_size))

    def _capture(self, padded_kwargs: dict[str, Any], *, key: Any) -> _CapturedModelCall:
        static_kwargs = {
            name: _empty_like_tree(value)
            for name, value in padded_kwargs.items()
        }
        for name, value in padded_kwargs.items():
            _copy_tree_(static_kwargs[name], value)

        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(self.capture_warmup):
                self.model(**static_kwargs)
        torch.cuda.current_stream().wait_stream(side_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = self.model(**static_kwargs)

        captured = _CapturedModelCall(
            graph=graph,
            static_kwargs=static_kwargs,
            static_output=static_output,
        )
        self._captured[key] = captured
        return captured


def parse_cuda_graph_buckets(raw: str | None, *, fallback_max_batch_size: int) -> tuple[int, ...]:
    """Parse a comma/semicolon separated bucket list."""
    if raw is None or not str(raw).strip():
        return (int(fallback_max_batch_size),)
    buckets = []
    for item in str(raw).replace(";", ",").split(","):
        clean = item.strip()
        if clean:
            buckets.append(int(clean))
    if not buckets:
        return (int(fallback_max_batch_size),)
    return tuple(sorted(set(buckets)))


__all__ = [
    "CudaGraphModelRunner",
    "pad_model_kwargs_to_bucket",
    "parse_cuda_graph_buckets",
]
