# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``mlcore.collection.scripted``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from mlcore.collection import ScriptedCollector, ScriptedReachPolicy


class _FakeEnv:
    """Deterministic fake EnvAdapter for collector tests.

    Returns an obs dict with ``ee_pos``, ``cube_pos``, ``joint_positions``.
    Terminates after ``steps_per_episode`` ``step`` calls; the final info
    dict carries ``success`` set from the constructor.
    """

    def __init__(self, steps_per_episode: int = 3, success: bool = True) -> None:
        self._steps_per_episode = steps_per_episode
        self._success = success
        self._step_count = 0
        self.closed = False
        self.reset_seeds: list[int] = []

    def _obs(self) -> dict[str, NDArray[np.floating[Any]]]:
        return {
            "ee_pos": np.zeros(3, dtype=np.float32),
            "cube_pos": np.ones(3, dtype=np.float32),
            "joint_positions": np.zeros(7, dtype=np.float32),
        }

    def reset(
        self, seed: int
    ) -> tuple[dict[str, NDArray[np.floating[Any]]], dict[str, Any]]:
        self.reset_seeds.append(seed)
        self._step_count = 0
        return self._obs(), {}

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
        return self._obs(), 1.0, terminated, False, info

    def close(self) -> None:
        self.closed = True


@pytest.mark.unit
def test_scripted_reach_policy_returns_8d_clipped_action() -> None:
    policy = ScriptedReachPolicy(gain=10.0, noise_scale=0.0)
    obs: dict[str, Any] = {
        "ee_pos": np.zeros(3, dtype=np.float32),
        "cube_pos": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }
    action = policy.select_action(obs)
    assert action.shape == (8,)
    assert action.dtype == np.float32
    assert (action <= 1.0).all() and (action >= -1.0).all()


@pytest.mark.unit
def test_scripted_reach_policy_with_noise_adds_perturbation() -> None:
    policy = ScriptedReachPolicy(gain=0.0, noise_scale=0.1, seed=1)
    obs: dict[str, Any] = {
        "ee_pos": np.zeros(3, dtype=np.float32),
        "cube_pos": np.zeros(3, dtype=np.float32),
    }
    action = policy.select_action(obs)
    # Noise should make the action non-zero even with zero gain.
    assert not np.allclose(action, 0.0)


@pytest.mark.unit
def test_scripted_reach_policy_handles_missing_keys() -> None:
    """Missing ``ee_pos`` / ``cube_pos`` default to zeros — action is just noise."""
    policy = ScriptedReachPolicy(gain=2.0, noise_scale=0.0)
    action = policy.select_action({})
    np.testing.assert_array_equal(action, np.zeros(8, dtype=np.float32))


@pytest.mark.unit
def test_scripted_reach_policy_custom_action_dim() -> None:
    policy = ScriptedReachPolicy(action_dim=12, noise_scale=0.0)
    obs: dict[str, Any] = {
        "ee_pos": np.zeros(3, dtype=np.float32),
        "cube_pos": np.zeros(3, dtype=np.float32),
    }
    assert policy.select_action(obs).shape == (12,)


@pytest.mark.unit
def test_scripted_collector_returns_n_episodes_with_metadata() -> None:
    env = _FakeEnv(steps_per_episode=4, success=True)
    policy = ScriptedReachPolicy(noise_scale=0.0)
    collector = ScriptedCollector(
        policy=policy,
        task_id="cube_reach_v1",
        task_description="Reach the red cube",
        seed=100,
    )

    episodes = collector.collect(env, n_episodes=3)

    assert len(episodes) == 3
    for ep in episodes:
        assert ep.task_id == "cube_reach_v1"
        assert ep.task_description == "Reach the red cube"
        assert ep.success is True
        assert len(ep.observations) == 4
        assert len(ep.actions) == 4
        assert len(ep.rewards) == 4

    assert env.reset_seeds == [100, 101, 102]
    assert env.closed is True


@pytest.mark.unit
def test_scripted_collector_records_failure_when_env_reports_no_success() -> None:
    env = _FakeEnv(steps_per_episode=2, success=False)
    collector = ScriptedCollector(
        policy=ScriptedReachPolicy(noise_scale=0.0),
        task_id="t",
        task_description="d",
    )
    episodes = collector.collect(env, n_episodes=1)
    assert episodes[0].success is False
