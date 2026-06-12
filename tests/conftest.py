# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for ml-core tests."""

from __future__ import annotations

import shutil
from collections.abc import Generator
from pathlib import Path

import pytest

from mlcore.robots import RobotSpec, register
from mlcore.robots.registry import _REGISTRY

# Reference spec formerly shipped as the built-in ``mlcore.robots.specs.cube_reach_v1``
# production module. Removed in P2 (ml-core is robot-agnostic; concrete specs are now
# data-driven via YAML downstream). Kept here as a TEST fixture so the existing suite,
# which depends on ``get("cube_reach_v1")``, stays green without production
# auto-registration.
_CUBE_REACH_V1 = RobotSpec(
    name="cube_reach_v1",
    n_joints=7,
    obs_keys=["ee_pos", "cube_pos", "joints"],
    action_dim=8,  # 7 joints + 1 gripper
    ee_pos_key="ee_pos",
    target_pos_key="cube_pos",
    relational_features=(("cube_pos", "ee_pos"),),
)


@pytest.fixture(autouse=True)
def register_cube_reach_v1() -> Generator[None, None, None]:
    """Register the reference ``cube_reach_v1`` spec for the duration of each test.

    Replaces the production import-time auto-registration removed in P2. Snapshots
    and restores the registry so the fixture cannot leak the spec into tests that
    assert on registry contents.
    """
    snapshot = dict(_REGISTRY)
    register(_CUBE_REACH_V1)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


@pytest.fixture(autouse=True)
def clean_checkpoints() -> Generator[None, None, None]:
    """Remove any leftover checkpoint directories before and after each test."""
    ckpt_root = Path("checkpoints")
    if ckpt_root.exists():
        shutil.rmtree(ckpt_root)
    yield
    if ckpt_root.exists():
        shutil.rmtree(ckpt_root)
