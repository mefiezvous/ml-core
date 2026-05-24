# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``mlcore.collection.interfaces``."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from mlcore.collection import Collector, Episode, ScriptedCollector, ScriptedReachPolicy
from mlcore.collection.interfaces import EnvLike


@pytest.mark.unit
def test_episode_is_frozen() -> None:
    """Episode is a frozen dataclass — attribute writes raise."""
    ep = Episode(
        observations=(),
        actions=(),
        rewards=(),
        task_id="t",
        task_description="d",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ep.task_id = "other"  # type: ignore[misc]


@pytest.mark.unit
def test_episode_is_hashable_when_payload_is_hashable() -> None:
    """An Episode with empty payload is hashable (frozen + tuple fields)."""
    ep = Episode(
        observations=(),
        actions=(),
        rewards=(),
        task_id="t",
        task_description="d",
    )
    # Calling hash() must not raise; result is an int.
    assert isinstance(hash(ep), int)


@pytest.mark.unit
def test_episode_defaults_success_to_false() -> None:
    ep = Episode(
        observations=(),
        actions=(),
        rewards=(),
        task_id="t",
        task_description="d",
    )
    assert ep.success is False


@pytest.mark.unit
def test_collector_protocol_detects_scripted_collector() -> None:
    """ScriptedCollector matches the Collector Protocol structurally."""
    coll = ScriptedCollector(
        policy=ScriptedReachPolicy(),
        task_id="t",
        task_description="d",
    )
    assert isinstance(coll, Collector)


@pytest.mark.unit
def test_envlike_protocol_detects_minimal_env() -> None:
    """A duck-typed env satisfies EnvLike."""

    class _MiniEnv:
        def reset(self, seed: int) -> tuple[dict[str, NDArray[np.floating[Any]]], dict[str, Any]]:
            return {}, {}

        def step(
            self, action: NDArray[np.floating[Any]]
        ) -> tuple[
            dict[str, NDArray[np.floating[Any]]],
            float,
            bool,
            bool,
            dict[str, Any],
        ]:
            return {}, 0.0, True, False, {}

        def close(self) -> None:
            return None

    assert isinstance(_MiniEnv(), EnvLike)
