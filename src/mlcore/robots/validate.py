# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Runtime validation: ensure a ``RobotSpec`` matches an ``EnvAdapter`` shape.

This catches startup-time misconfigurations (wrong action dim, missing obs keys)
before any training step is taken, so the user gets a clear, actionable error
instead of a cryptic crash deep inside the policy or trainer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from loguru import logger

from mlcore.robots.base import RobotSpec

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    pass


@runtime_checkable
class _EnvAdapterShape(Protocol):
    """Minimal structural view of an EnvAdapter used for validation.

    We do not import :class:`robotics_platform.envs.interfaces.EnvAdapter`
    directly to avoid forcing ``mlcore`` to depend on the platform package at
    type-check time for this single function — the structural Protocol below
    is sufficient and keeps ``mlcore`` decoupled.
    """

    @property
    def obs_space_keys(self) -> list[str]: ...

    @property
    def action_dim(self) -> int: ...


class RobotSpecMismatch(ValueError):
    """Raised when a ``RobotSpec`` is incompatible with an ``EnvAdapter``."""


def validate_spec_against_env(spec: RobotSpec, env: Any) -> None:
    """Verify that ``env`` exposes everything ``spec`` declares.

    Args:
        spec: The robot specification expected by the policy/pipeline.
        env: An :class:`~robotics_platform.envs.interfaces.EnvAdapter`-compatible
            object (structural; only ``obs_space_keys`` and ``action_dim`` are
            inspected here).

    Raises:
        RobotSpecMismatch: If ``spec.action_dim`` differs from
            ``env.action_dim`` or if any key in ``spec.obs_keys`` is absent
            from ``env.obs_space_keys``.
    """
    env_name = type(env).__name__

    if spec.action_dim != env.action_dim:
        raise RobotSpecMismatch(
            f"Action dim mismatch: spec='{spec.name}' declares "
            f"action_dim={spec.action_dim} but env exposes "
            f"action_dim={env.action_dim}"
        )

    available = set(env.obs_space_keys)
    required = set(spec.obs_keys)
    missing = required - available
    if missing:
        raise RobotSpecMismatch(
            f"Missing obs keys in env '{env_name}': spec requires "
            f"{sorted(missing)} but env only provides {sorted(available)}"
        )

    logger.info(
        "RobotSpec '{}' validated against env '{}' (action_dim={}, obs_keys OK)",
        spec.name,
        env_name,
        spec.action_dim,
    )
