# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Build Hydra-compatible training config dicts from RobotSpec."""

from __future__ import annotations

from mlcore.robots.base import RobotSpec


def build_train_config(spec: RobotSpec, policy_type: str) -> dict[str, object]:
    """Generate a Hydra-compatible training config dict for a given robot spec.

    Args:
        spec: Robot descriptor.
        policy_type: Policy identifier, e.g. ``"act"`` or ``"diffusion"``.

    Returns:
        Dict with keys robot_name, policy_type, env, training.
    """
    return {
        "robot_name": spec.name,
        "policy_type": policy_type,
        "env": {
            "n_joints": spec.n_joints,
            "obs_keys": list(spec.obs_keys),
        },
        "training": {
            "max_episode_steps": spec.max_episode_steps,
        },
    }
