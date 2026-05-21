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
        success_threshold: Distance (metres) below which the episode is a success.
        max_episode_steps: Hard step cap per episode.
        ee_pos_key: Key in obs dict for end-effector XYZ position.
        target_pos_key: Key in obs dict for target XYZ position.
    """

    name: str
    n_joints: int
    obs_keys: list[str]
    action_dim: int
    success_threshold: float = 0.05
    max_episode_steps: int = 200
    ee_pos_key: str = "ee_pos"
    target_pos_key: str = "cube_pos"
