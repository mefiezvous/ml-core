# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Generic proportional reach controller parameterised by RobotSpec."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlcore.robots.base import RobotSpec


class GenericScriptedPolicy:
    """Proportional reach controller paramétré par RobotSpec.

    Computes a Cartesian delta toward the target position, scales it by gain,
    clips to [-1, 1], then pads or truncates to spec.action_dim.

    Args:
        spec: Robot descriptor that defines obs keys and action dimension.
        gain: Proportional gain applied to the EE–target delta.
    """

    def __init__(self, spec: RobotSpec, gain: float = 5.0) -> None:
        self._spec = spec
        self._gain = gain

    def select_action(self, obs: dict[str, Any]) -> np.ndarray[Any, np.dtype[Any]]:
        """Return an action of shape (spec.action_dim,) that moves EE toward target.

        Args:
            obs: Observation dict containing at least ee_pos_key and target_pos_key.

        Returns:
            Float32 array of shape (action_dim,), clipped to [-1, 1].
        """
        ee_pos = np.asarray(obs[self._spec.ee_pos_key], dtype=np.float32)
        target = np.asarray(obs[self._spec.target_pos_key], dtype=np.float32)
        delta = target - ee_pos
        raw = self._gain * delta

        action = np.zeros(self._spec.action_dim, dtype=np.float32)
        n = min(len(raw), self._spec.action_dim)
        action[:n] = raw[:n]
        return np.clip(action, -1.0, 1.0)
