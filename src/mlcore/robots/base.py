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
