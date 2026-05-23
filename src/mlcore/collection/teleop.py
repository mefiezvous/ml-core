# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Teleoperation collector stub.

Ships the interface for Phase 4 so the import graph and Collector
registry remain intact. The implementation lives behind a
``NotImplementedError`` and will be wired when the real HAL — LeRobot
teleop scripts or a leader-follower hardware setup — becomes available.
"""

from __future__ import annotations

from typing import Any

from mlcore.collection.interfaces import EnvLike, Episode


class TeleopCollector:
    """Stub for human-teleoperation collection.

    The constructor accepts arbitrary ``*args`` / ``**kwargs`` to remain
    forward-compatible with the real implementation; nothing is stored.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs  # explicit "intentionally ignored" for ruff/mypy

    def collect(self, env: EnvLike, n_episodes: int) -> list[Episode]:
        """Raise ``NotImplementedError`` — see class docstring.

        Args:
            env: Ignored.
            n_episodes: Ignored.

        Raises:
            NotImplementedError: Always.
        """
        del env, n_episodes
        raise NotImplementedError(
            "TeleopCollector is a stub. Implement when LeRobot teleop scripts "
            "or hardware leader-follower setup is wired."
        )
