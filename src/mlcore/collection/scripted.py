# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Scripted reach policy + collector.

Migrated and generalised from
``lerobot-playground-portfolio/src/playground/data/pipeline.py``. The
collector is now task-agnostic: ``task_id`` and ``task_description`` are
constructor arguments rather than hardcoded defaults, and the env is
passed directly (no name resolution via a registry) so this module has
zero coupling to ``playground`` or ``robotics_platform``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from mlcore.collection.interfaces import EnvLike, Episode


class ScriptedReachPolicy:
    """Proportional reach controller for scripted demo collection.

    Moves the end-effector toward a target with a simple P-controller in
    Cartesian space, padded to a full ``action_dim``-DOF action vector.

    Args:
        gain: Proportional gain applied to the EE-target delta.
        noise_scale: Std-dev of Gaussian noise added to the action. Use
            ``0.0`` to disable.
        seed: RNG seed for reproducible noise.
        action_dim: Total action vector length. Defaults to 8 (7 joints
            + 1 gripper for a Franka-class arm).
        ee_key: Observation key for the end-effector XYZ position.
        target_key: Observation key for the target XYZ position.
    """

    def __init__(
        self,
        gain: float = 2.0,
        noise_scale: float = 0.05,
        seed: int = 42,
        action_dim: int = 8,
        ee_key: str = "ee_pos",
        target_key: str = "cube_pos",
    ) -> None:
        self._gain = gain
        self._noise_scale = noise_scale
        self._rng = np.random.default_rng(seed)
        self._action_dim = action_dim
        self._ee_key = ee_key
        self._target_key = target_key

    def select_action(self, obs: dict[str, Any]) -> NDArray[np.floating[Any]]:
        """Return an action that moves the EE toward the target.

        Args:
            obs: Raw observation dict. Must contain ``ee_key`` and
                ``target_key``; defaults to zeros if missing.

        Returns:
            Action of shape ``(action_dim,)`` and dtype ``float32``,
            clipped to ``[-1, 1]``.
        """
        ee_pos = np.asarray(obs.get(self._ee_key, np.zeros(3)), dtype=np.float32)
        target_pos = np.asarray(obs.get(self._target_key, np.zeros(3)), dtype=np.float32)
        delta = target_pos - ee_pos

        action = np.zeros(self._action_dim, dtype=np.float32)
        action[:3] = self._gain * delta[:3]
        if self._noise_scale > 0.0:
            noise = self._rng.normal(0.0, self._noise_scale, self._action_dim).astype(
                np.float32
            )
            action += noise
        return np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)


class ScriptedCollector:
    """Collect episodes by rolling out a :class:`ScriptedReachPolicy`.

    Args:
        policy: Scripted policy producing actions from observations.
        task_id: Stable identifier stamped on every collected episode.
        task_description: Natural-language label stamped on every episode.
        seed: Base RNG seed; episode ``i`` uses ``seed + i`` for env reset.
    """

    def __init__(
        self,
        policy: ScriptedReachPolicy,
        task_id: str,
        task_description: str,
        seed: int = 42,
    ) -> None:
        self._policy = policy
        self._task_id = task_id
        self._task_description = task_description
        self._seed = seed

    def collect(self, env: EnvLike, n_episodes: int) -> list[Episode]:
        """Run the scripted policy for ``n_episodes`` and return Episodes.

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
                action = self._policy.select_action(obs)
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
                f"Episode {i + 1}/{n_episodes} done | "
                f"steps={len(ep_rewards)} | success={success}"
            )

        env.close()
        return episodes
