# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Core data model and Protocols for the collection package.

The :class:`Episode` dataclass is **frozen** and uses tuples for its
sequence fields so instances are immutable and hashable. The
:class:`Collector` Protocol captures the structural contract every
collector must satisfy.

The :class:`EnvLike` Protocol mirrors
``robotics_platform.envs.interfaces.EnvAdapter`` structurally so that
``ml-core`` can type-check ``Collector.collect(env, ...)`` calls without
taking a hard import dependency on the template layer (per the
import-isolation rule documented in ``ml-core/CLAUDE.md``). Any concrete
``EnvAdapter`` satisfies :class:`EnvLike` via duck typing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Episode:
    """Single collected trajectory.

    All sequence fields are ``tuple`` (not ``list``) so the dataclass is
    safely hashable and immutable. Pass lists or tuples to the constructor;
    wrap explicitly in ``tuple(...)`` if you start from a list.

    Attributes:
        observations: Raw observation dicts from the env, one per step.
        actions: Per-step actions, each a ``np.ndarray``.
        rewards: Per-step scalar rewards.
        task_id: Short, stable identifier for the task (e.g. ``"cube_reach_v1"``).
        task_description: Natural-language description of the task.
        success: ``True`` if the episode ended successfully.
    """

    observations: tuple[dict[str, Any], ...]
    actions: tuple[NDArray[np.floating[Any]], ...]
    rewards: tuple[float, ...]
    task_id: str
    task_description: str
    success: bool = False


@runtime_checkable
class EnvLike(Protocol):
    """Structural counterpart to ``robotics_platform.envs.interfaces.EnvAdapter``.

    Defined locally to avoid importing from ``robotics_platform_template``.
    Any class that conforms to this Protocol — including a concrete
    ``EnvAdapter`` — is a valid argument to :meth:`Collector.collect`.
    """

    def reset(
        self, seed: int
    ) -> tuple[dict[str, NDArray[np.floating[Any]]], dict[str, Any]]:
        """Reset to an initial state and return ``(obs, info)``."""
        ...

    def step(
        self, action: NDArray[np.floating[Any]]
    ) -> tuple[
        dict[str, NDArray[np.floating[Any]]],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """Step the env and return ``(obs, reward, terminated, truncated, info)``."""
        ...

    def close(self) -> None:
        """Release underlying resources."""
        ...


@runtime_checkable
class Collector(Protocol):
    """Structural contract for any episode collector.

    Implementations consume an :class:`EnvLike` and produce a list of
    :class:`Episode` instances.
    """

    def collect(self, env: EnvLike, n_episodes: int) -> list[Episode]:
        """Collect ``n_episodes`` trajectories from ``env``.

        Args:
            env: An environment satisfying :class:`EnvLike`.
            n_episodes: Number of episodes to collect.

        Returns:
            A list of :class:`Episode` instances.
        """
        ...
