# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""RobotSpec — declarative descriptor for a robot configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotSpec:
    """Immutable descriptor for a robot's kinematics and observation layout.

    Args:
        name: Unique snake_case identifier used as namespace key.
        n_joints: Degrees of freedom (DOF) of the robot.
        obs_keys: Observation dict keys consumed by policies.
        action_dim: Dimension of the action vector.
        target_pos_key: Key in obs dict for target XYZ position. Mandatory —
            every task must declare its target explicitly (no implicit cube).
        success_threshold: Distance (metres) below which the episode is a success.
        max_episode_steps: Hard step cap per episode.
        ee_pos_key: Key in obs dict for end-effector XYZ position.
        extra_obs_keys: Optional task-specific obs keys (gripper, force, etc.).
            Tuple (not list) because the dataclass is frozen.
        relational_features: Pairs ``(b_key, a_key)`` for which the pipeline
            should compute the delta ``obs[b_key] - obs[a_key]``. For example
            ``(("cube_pos", "ee_pos"),)`` yields an ``ee_to_cube`` feature.
            Tuple of tuples for hashability.

    Raises:
        ValueError: If any field invariant is violated (MLC-009).

    Note:
        The global ``_REGISTRY`` (module-level dict) is **not thread-safe**.
        Registering specs from multiple threads concurrently is unsafe; in
        practice specs are only registered at import time, so this is fine.
    """

    name: str
    n_joints: int
    obs_keys: list[str]
    action_dim: int
    target_pos_key: str
    success_threshold: float = 0.05
    max_episode_steps: int = 200
    ee_pos_key: str = "ee_pos"
    extra_obs_keys: tuple[str, ...] = ()
    relational_features: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Validate field invariants (MLC-009)."""
        if not self.name:
            raise ValueError("RobotSpec.name must be a non-empty string")
        if self.n_joints <= 0:
            raise ValueError(f"RobotSpec.n_joints must be > 0, got {self.n_joints}")
        if self.action_dim <= 0:
            raise ValueError(f"RobotSpec.action_dim must be > 0, got {self.action_dim}")
        if not self.obs_keys:
            raise ValueError("RobotSpec.obs_keys must contain at least one key")
        if not (0.0 < self.success_threshold < 1.0):
            raise ValueError(
                f"RobotSpec.success_threshold must be in (0, 1), got {self.success_threshold}"
            )
        if self.max_episode_steps <= 0:
            raise ValueError(
                f"RobotSpec.max_episode_steps must be > 0, got {self.max_episode_steps}"
            )
