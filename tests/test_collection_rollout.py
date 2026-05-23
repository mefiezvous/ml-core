# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``mlcore.collection.rollout``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from mlcore.collection import PolicyRolloutCollector


class _FakeEnv:
    def __init__(self, steps_per_episode: int = 2, success: bool = True) -> None:
        self._steps_per_episode = steps_per_episode
        self._success = success
        self._step_count = 0
        self.closed = False

    def _obs(self) -> dict[str, NDArray[np.floating[Any]]]:
        return {"ee_pos": np.zeros(3, dtype=np.float32)}

    def reset(
        self, seed: int
    ) -> tuple[dict[str, NDArray[np.floating[Any]]], dict[str, Any]]:
        self._step_count = 0
        return self._obs(), {"seed": seed}

    def step(
        self, action: NDArray[np.floating[Any]]
    ) -> tuple[
        dict[str, NDArray[np.floating[Any]]],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        self._step_count += 1
        terminated = self._step_count >= self._steps_per_episode
        info: dict[str, Any] = {"success": self._success} if terminated else {}
        return self._obs(), 0.5, terminated, False, info

    def close(self) -> None:
        self.closed = True


@pytest.mark.unit
def test_policy_rollout_collector_calls_supplied_policy() -> None:
    calls: list[dict[str, Any]] = []

    def policy(obs: dict[str, Any]) -> NDArray[np.floating[Any]]:
        calls.append(obs)
        return np.full(8, 0.1, dtype=np.float32)

    env = _FakeEnv(steps_per_episode=3, success=True)
    coll = PolicyRolloutCollector(
        policy=policy,
        task_id="rollout",
        task_description="rollout task",
    )

    episodes = coll.collect(env, n_episodes=2)

    assert len(episodes) == 2
    assert len(calls) == 6  # 2 episodes × 3 steps
    for ep in episodes:
        assert ep.task_id == "rollout"
        assert ep.task_description == "rollout task"
        assert ep.success is True
        assert len(ep.actions) == 3
        np.testing.assert_allclose(ep.actions[0], 0.1 * np.ones(8, dtype=np.float32))
    assert env.closed is True


@pytest.mark.unit
def test_policy_rollout_collector_handles_failure() -> None:
    def policy(obs: dict[str, Any]) -> NDArray[np.floating[Any]]:
        return np.zeros(8, dtype=np.float32)

    env = _FakeEnv(steps_per_episode=1, success=False)
    coll = PolicyRolloutCollector(
        policy=policy,
        task_id="t",
        task_description="d",
    )
    episodes = coll.collect(env, n_episodes=1)
    assert episodes[0].success is False
