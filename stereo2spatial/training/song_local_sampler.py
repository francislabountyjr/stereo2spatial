"""Batch samplers that improve HDD locality for song-crop datasets."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator

from torch.utils.data import Sampler

from .dataset import WaveformSongDataset


class SongLocalBatchSampler(Sampler[list[int]]):
    """
    Yield crop-index batches grouped by song and interleaved into worker lanes.

    PyTorch's map-style DataLoader enqueues batch-index tasks to workers. By yielding
    lane0, lane1, ..., laneN batches repeatedly, each worker tends to keep receiving
    crops for the same active song, which improves per-worker cache reuse and reduces
    repeated HDD reads of large sample bundles.
    """

    def __init__(
        self,
        dataset: WaveformSongDataset,
        *,
        batch_size: int,
        num_workers: int,
        drop_last: bool,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.drop_last = bool(drop_last)

    def _song_batches(self) -> list[list[list[int]]]:
        groups: OrderedDict[int, list[int]] = OrderedDict()
        for index, segment in enumerate(self.dataset._segments):
            groups.setdefault(int(segment.song_index), []).append(index)

        songs: list[list[list[int]]] = []
        for indices in groups.values():
            batches: list[list[int]] = []
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
            if batches:
                songs.append(batches)
        return songs

    def __iter__(self) -> Iterator[list[int]]:
        songs = self._song_batches()
        lane_count = max(1, self.num_workers)
        lanes: list[list[list[list[int]]]] = [[] for _ in range(lane_count)]
        for song_idx, batches in enumerate(songs):
            lanes[song_idx % lane_count].append(batches)

        song_positions = [0 for _ in range(lane_count)]
        batch_positions = [0 for _ in range(lane_count)]
        active = True
        while active:
            active = False
            for lane_idx, lane in enumerate(lanes):
                while song_positions[lane_idx] < len(lane):
                    song_batches = lane[song_positions[lane_idx]]
                    batch_pos = batch_positions[lane_idx]
                    if batch_pos < len(song_batches):
                        batch_positions[lane_idx] += 1
                        active = True
                        yield song_batches[batch_pos]
                        break
                    song_positions[lane_idx] += 1
                    batch_positions[lane_idx] = 0

    def __len__(self) -> int:
        return sum(len(song_batches) for song_batches in self._song_batches())


__all__ = ["SongLocalBatchSampler"]
