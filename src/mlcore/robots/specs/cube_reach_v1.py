# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Spec for the CubeReachV1 Franka Panda task."""

from __future__ import annotations

from mlcore.robots.base import RobotSpec
from mlcore.robots.registry import register

CUBE_REACH_V1 = RobotSpec(
    name="cube_reach_v1",
    n_joints=7,
    obs_keys=["ee_pos", "cube_pos", "joints"],
    action_dim=8,  # 7 joints + 1 gripper
    ee_pos_key="ee_pos",
    target_pos_key="cube_pos",
)
register(CUBE_REACH_V1)
