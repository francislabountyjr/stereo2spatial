from __future__ import annotations

from typing import Any, cast

import pytest

from stereo2spatial.training.dataset_types import ConditioningSource, EpochSegment
from stereo2spatial.training.song_local_sampler import SongLocalBatchSampler


def _segment(song_index: int, start_frame: int) -> EpochSegment:
    return EpochSegment(
        song_index=song_index,
        start_frame=start_frame,
        num_valid_frames=1,
        conditioning_source=ConditioningSource.STEREO,
    )


def _dataset(song_counts: list[int]) -> Any:
    segments: list[EpochSegment] = []
    for song_index, count in enumerate(song_counts):
        for start_frame in range(count):
            segments.append(_segment(song_index, start_frame))
    return cast(Any, type("DatasetStub", (), {"_segments": segments})())


def test_song_local_batch_sampler_keeps_batches_within_one_song() -> None:
    dataset = _dataset([5, 4, 3])
    sampler = SongLocalBatchSampler(
        dataset,
        batch_size=2,
        num_workers=2,
        drop_last=False,
    )

    batches = list(sampler)

    assert len(batches) == 7
    for batch in batches:
        song_ids = {dataset._segments[index].song_index for index in batch}
        assert len(song_ids) == 1


def test_song_local_batch_sampler_interleaves_worker_lanes() -> None:
    dataset = _dataset([4, 4, 4])
    sampler = SongLocalBatchSampler(
        dataset,
        batch_size=2,
        num_workers=2,
        drop_last=True,
    )

    batch_song_ids = [
        dataset._segments[batch[0]].song_index
        for batch in sampler
    ]

    assert batch_song_ids == [0, 1, 0, 1, 2, 2]


def test_song_local_batch_sampler_drops_partial_batches() -> None:
    dataset = _dataset([5, 4])
    sampler = SongLocalBatchSampler(
        dataset,
        batch_size=2,
        num_workers=1,
        drop_last=True,
    )

    batches = list(sampler)

    assert len(batches) == len(sampler) == 4
    assert all(len(batch) == 2 for batch in batches)


def test_song_local_batch_sampler_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        SongLocalBatchSampler(
            _dataset([1]),
            batch_size=0,
            num_workers=0,
            drop_last=False,
        )
