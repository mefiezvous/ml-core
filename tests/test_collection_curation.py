# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``mlcore.collection.curation``."""

from __future__ import annotations

import numpy as np
import pytest

from mlcore.collection import (
    Episode,
    balance_by_task,
    dedupe_by_trajectory_hash,
    filter_by_success,
)


def _ep(task_id: str, success: bool, action_seed: int = 0, n_steps: int = 3) -> Episode:
    rng = np.random.default_rng(action_seed)
    actions = tuple(rng.standard_normal(8).astype(np.float32) for _ in range(n_steps))
    return Episode(
        observations=tuple({} for _ in range(n_steps)),
        actions=actions,
        rewards=tuple(0.0 for _ in range(n_steps)),
        task_id=task_id,
        task_description=task_id,
        success=success,
    )


@pytest.mark.unit
def test_filter_by_success_drops_failures() -> None:
    eps = [_ep("a", True, 0), _ep("a", False, 1), _ep("a", True, 2)]
    out = filter_by_success(eps)
    assert len(out) == 2
    assert all(ep.success for ep in out)


@pytest.mark.unit
def test_filter_by_success_returns_empty_when_all_fail() -> None:
    eps = [_ep("a", False, 0), _ep("b", False, 1)]
    assert filter_by_success(eps) == []


@pytest.mark.unit
def test_balance_by_task_uses_min_count_when_target_none() -> None:
    eps = [
        _ep("a", True, 0),
        _ep("a", True, 1),
        _ep("a", True, 2),
        _ep("b", True, 3),
        _ep("b", True, 4),
    ]
    out = balance_by_task(eps)
    counts = {"a": 0, "b": 0}
    for ep in out:
        counts[ep.task_id] += 1
    assert counts == {"a": 2, "b": 2}


@pytest.mark.unit
def test_balance_by_task_respects_explicit_target() -> None:
    eps = [
        _ep("a", True, 0),
        _ep("a", True, 1),
        _ep("a", True, 2),
        _ep("b", True, 3),
        _ep("b", True, 4),
    ]
    out = balance_by_task(eps, target_per_task=1)
    assert len(out) == 2
    assert {ep.task_id for ep in out} == {"a", "b"}


@pytest.mark.unit
def test_balance_by_task_preserves_relative_order() -> None:
    eps = [
        _ep("a", True, 0),
        _ep("b", True, 1),
        _ep("a", True, 2),
        _ep("b", True, 3),
    ]
    out = balance_by_task(eps, target_per_task=1)
    # First-seen wins per task → order is [a@0, b@1]
    assert [ep.task_id for ep in out] == ["a", "b"]


@pytest.mark.unit
def test_balance_by_task_empty_input() -> None:
    assert balance_by_task([]) == []


@pytest.mark.unit
def test_dedupe_drops_duplicate_action_sequences() -> None:
    a = _ep("t", True, 0)
    b_dup = _ep("t", True, 0)  # same seed → same actions
    c = _ep("t", True, 99)
    out = dedupe_by_trajectory_hash([a, b_dup, c])
    assert len(out) == 2
    assert out[0] is a
    assert out[2 - 1] is c  # second kept item is c


@pytest.mark.unit
def test_dedupe_preserves_unique_episodes() -> None:
    eps = [_ep("t", True, i) for i in range(5)]
    out = dedupe_by_trajectory_hash(eps)
    assert len(out) == 5


@pytest.mark.unit
def test_dedupe_handles_empty_action_episodes() -> None:
    empty = Episode(
        observations=(),
        actions=(),
        rewards=(),
        task_id="t",
        task_description="d",
    )
    out = dedupe_by_trajectory_hash([empty, empty])
    assert len(out) == 1
