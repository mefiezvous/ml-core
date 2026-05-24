# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :mod:`mlcore.data.filter`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from torch.utils.data import Subset

from mlcore.data.filter import SuccessOnlyFilter, TaskFilter


class _FakeDataset:
    """Tiny in-memory dataset of dict frames for filter/sampler tests."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._records[idx]


@pytest.mark.unit
def test_success_only_filter_keeps_only_successes() -> None:
    records: list[dict[str, Any]] = [
        {"success": np.array([True]), "task_id": "t1"},
        {"success": np.array([False]), "task_id": "t1"},
        {"success": np.array([True]), "task_id": "t2"},
        {"success": np.array([False]), "task_id": "t2"},
    ]
    ds = _FakeDataset(records)

    subset = SuccessOnlyFilter()(ds)

    assert isinstance(subset, Subset)
    assert sorted(subset.indices) == [0, 2]
    # Subset items round-trip back to the originals.
    assert subset[0]["task_id"] == "t1"
    assert subset[1]["task_id"] == "t2"


@pytest.mark.unit
def test_success_only_filter_accepts_scalar_bool() -> None:
    """Scalar bools (not just shape-(1,) arrays) are honoured for in-memory fakes."""
    records: list[dict[str, Any]] = [
        {"success": True, "task_id": "t1"},
        {"success": False, "task_id": "t1"},
    ]
    subset = SuccessOnlyFilter()(_FakeDataset(records))
    assert list(subset.indices) == [0]


@pytest.mark.unit
def test_success_only_filter_empty_input_returns_empty_subset() -> None:
    subset = SuccessOnlyFilter()(_FakeDataset([]))
    assert isinstance(subset, Subset)
    assert list(subset.indices) == []


@pytest.mark.unit
def test_task_filter_includes_allow_list_only() -> None:
    records: list[dict[str, Any]] = [
        {"success": True, "task_id": "a"},
        {"success": True, "task_id": "b"},
        {"success": True, "task_id": "c"},
        {"success": True, "task_id": "a"},
    ]
    subset = TaskFilter(task_ids=["a", "c"])(_FakeDataset(records))
    assert isinstance(subset, Subset)
    assert sorted(subset.indices) == [0, 2, 3]


@pytest.mark.unit
def test_task_filter_rejects_empty_allow_list() -> None:
    with pytest.raises(ValueError, match="at least one task_id"):
        TaskFilter(task_ids=[])


@pytest.mark.unit
def test_task_filter_chained_with_success_filter() -> None:
    """Filters compose: TaskFilter ∘ SuccessOnlyFilter returns the intersection."""
    records: list[dict[str, Any]] = [
        {"success": True, "task_id": "a"},
        {"success": False, "task_id": "a"},
        {"success": True, "task_id": "b"},
        {"success": True, "task_id": "c"},
    ]
    ds = _FakeDataset(records)
    only_success = SuccessOnlyFilter()(ds)
    only_ab_success = TaskFilter(task_ids=["a", "b"])(only_success)

    # The chained Subset indices are *into the SuccessOnly Subset* (not ds);
    # but the items themselves must satisfy both predicates.
    materialised = [only_ab_success[i] for i in range(len(only_ab_success))]
    assert all(bool(rec["success"]) for rec in materialised)
    assert all(rec["task_id"] in {"a", "b"} for rec in materialised)
    assert len(materialised) == 2  # records 0 and 2
