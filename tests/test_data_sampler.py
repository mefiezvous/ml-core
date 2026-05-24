# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :mod:`mlcore.data.sampler`."""

from __future__ import annotations

from typing import Any

import pytest

from mlcore.data.sampler import MultiTaskBalancedSampler, SuccessOnlySampler


class _FakeDataset:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._records[idx]


def _multitask_dataset(per_task: int = 25, n_tasks: int = 4) -> _FakeDataset:
    """Build a 100-frame, 4-task dataset (25 frames per task by default)."""
    records: list[dict[str, Any]] = []
    for t in range(n_tasks):
        for _ in range(per_task):
            records.append({"task_id": f"task_{t}", "success": True})
    return _FakeDataset(records)


@pytest.mark.unit
def test_multitask_sampler_each_batch_uses_exactly_tasks_per_batch_tasks() -> None:
    ds = _multitask_dataset()
    sampler = MultiTaskBalancedSampler(
        ds, tasks_per_batch=2, batch_size=8, seed=123
    )

    indices = list(sampler)
    # 100 frames / 8 batch_size = 12 full batches → 96 indices.
    assert len(indices) == 96
    assert len(sampler) == 96

    # Each batch must draw from exactly 2 distinct task_ids.
    batch_size = 8
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        task_ids = {ds[i]["task_id"] for i in batch}
        assert len(task_ids) == 2, f"Batch starting at {start} drew from {task_ids}"


@pytest.mark.unit
def test_multitask_sampler_is_deterministic_given_seed() -> None:
    ds = _multitask_dataset()
    a = list(MultiTaskBalancedSampler(ds, tasks_per_batch=2, batch_size=8, seed=7))
    b = list(MultiTaskBalancedSampler(ds, tasks_per_batch=2, batch_size=8, seed=7))
    c = list(MultiTaskBalancedSampler(ds, tasks_per_batch=2, batch_size=8, seed=8))
    assert a == b
    assert a != c


@pytest.mark.unit
def test_multitask_sampler_rejects_indivisible_batch_size() -> None:
    ds = _multitask_dataset()
    with pytest.raises(ValueError, match="divisible"):
        MultiTaskBalancedSampler(ds, tasks_per_batch=3, batch_size=8)


@pytest.mark.unit
def test_multitask_sampler_rejects_more_tasks_than_available() -> None:
    ds = _multitask_dataset(n_tasks=2)
    with pytest.raises(ValueError, match="exceeds number"):
        MultiTaskBalancedSampler(ds, tasks_per_batch=4, batch_size=8)


@pytest.mark.unit
def test_multitask_sampler_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        MultiTaskBalancedSampler(_FakeDataset([]), tasks_per_batch=1, batch_size=1)


@pytest.mark.unit
def test_multitask_sampler_handles_small_bucket_with_replacement() -> None:
    """When a task's bucket is smaller than per_task, sample with replacement."""
    records: list[dict[str, Any]] = [
        {"task_id": "rare", "success": True},  # only 1 frame
        {"task_id": "rare", "success": True},
        *({"task_id": "common", "success": True} for _ in range(10)),
    ]
    ds = _FakeDataset(records)
    sampler = MultiTaskBalancedSampler(
        ds, tasks_per_batch=2, batch_size=8, seed=0, num_samples=8
    )
    indices = list(sampler)
    # All indices must be valid (no IndexError raised).
    assert all(0 <= i < len(ds) for i in indices)
    assert len(indices) == 8


@pytest.mark.unit
def test_success_only_sampler_yields_only_success_indices() -> None:
    records: list[dict[str, Any]] = [
        {"task_id": "t", "success": True},   # 0
        {"task_id": "t", "success": False},  # 1
        {"task_id": "t", "success": True},   # 2
        {"task_id": "t", "success": False},  # 3
        {"task_id": "t", "success": True},   # 4
    ]
    ds = _FakeDataset(records)
    sampler = SuccessOnlySampler(ds, seed=0)
    indices = list(sampler)
    assert sorted(indices) == [0, 2, 4]
    assert len(sampler) == 3


@pytest.mark.unit
def test_success_only_sampler_is_deterministic_given_seed() -> None:
    records: list[dict[str, Any]] = [
        {"task_id": "t", "success": True} for _ in range(10)
    ]
    ds = _FakeDataset(records)
    a = list(SuccessOnlySampler(ds, seed=1))
    b = list(SuccessOnlySampler(ds, seed=1))
    c = list(SuccessOnlySampler(ds, seed=2))
    assert a == b
    # Probabilistically distinct shuffles (10! orderings).
    assert a != c


@pytest.mark.unit
def test_success_only_sampler_empty_when_no_successes() -> None:
    records: list[dict[str, Any]] = [
        {"task_id": "t", "success": False} for _ in range(5)
    ]
    sampler = SuccessOnlySampler(_FakeDataset(records))
    assert list(sampler) == []
    assert len(sampler) == 0
