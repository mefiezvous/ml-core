# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Global registry for RobotSpec instances."""

from __future__ import annotations

from mlcore.robots.base import RobotSpec

_REGISTRY: dict[str, RobotSpec] = {}


def register(spec: RobotSpec) -> None:
    """Add spec to the global registry under spec.name."""
    _REGISTRY[spec.name] = spec


def get(name: str) -> RobotSpec:
    """Return the RobotSpec registered under name.

    Raises:
        KeyError: If no spec with that name has been registered.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown robot spec: {name!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_specs() -> list[str]:
    """Return names of all registered specs."""
    return list(_REGISTRY.keys())
