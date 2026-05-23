# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Policy-rollout collector.

Drives a learned (or otherwise opaque) policy through an environment and
records the resulting trajectories as :class:`Episode` instances. Useful
for DAgger, off-policy evaluation, and bootstrapped data augmentation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from mlcore.collection.interfaces import EnvLike, Episode

PolicyFn = Callable[[dict[str, Any]], NDArray[np.floating[Any]]]


class PolicyRolloutCollector:
    """Collect rollouts driven by a generic policy callable.

    Args:
        policy: Callable mapping an observation dict to an action array.
        task_id: Stable identifier stamped on every collected episode.
        task_description: Natural-language label stamped on every episode.
        seed: Base RNG seed; episode ``i`` uses ``seed + i`` for env reset.
    """

    def __init__(
        self,
        policy: PolicyFn,
        task_id: str,
        task_description: str,
        seed: int = 42,
    ) -> None:
        self._policy = policy
        self._task_id = task_id
        self._task_description = task_description
        self._seed = seed

    def collect(self, env: EnvLike, n_episodes: int) -> list[Episode]:
        """Roll out ``policy`` for ``n_episodes`` and return Episodes.

        Args:
            env: Environment satisfying :class:`EnvLike`.
            n_episodes: Number of episodes to collect.

        Returns:
            List of :class:`Episode` instances.
        """
        episodes: list[Episode] = []

        for i in range(n_episodes):
            obs, _ = env.reset(seed=self._seed + i)
            ep_obs: list[dict[str, Any]] = []
            ep_actions: list[NDArray[np.floating[Any]]] = []
            ep_rewards: list[float] = []
            success = False
            done = False

            while not done:
                action = np.asarray(self._policy(obs), dtype=np.float32)
                ep_obs.append(obs)
                ep_actions.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_rewards.append(float(reward))
                done = bool(terminated or truncated)
                success = bool(info.get("success", False))

            episodes.append(
                Episode(
                    observations=tuple(ep_obs),
                    actions=tuple(ep_actions),
                    rewards=tuple(ep_rewards),
                    task_id=self._task_id,
                    task_description=self._task_description,
                    success=success,
                )
            )
            logger.info(
                f"Rollout episode {i + 1}/{n_episodes} done | "
                f"steps={len(ep_rewards)} | success={success}"
            )

        env.close()
        return episodes
