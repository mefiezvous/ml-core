# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``mlcore.robots.validate.validate_spec_against_env``."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from mlcore.robots import RobotSpec, RobotSpecMismatch, validate_spec_against_env


@dataclass
class FakeEnv:
    """Minimal structural EnvAdapter for validation tests."""

    action_dim: int
    obs_space_keys: list[str] = field(default_factory=list)


def _spec(
    *,
    name: str = "test_spec",
    action_dim: int = 6,
    obs_keys: tuple[str, ...] = ("ee_pos", "target_pos"),
) -> RobotSpec:
    target_pos_key = "target_pos" if "target_pos" in obs_keys else obs_keys[0]
    return RobotSpec(
        name=name,
        n_joints=action_dim,
        obs_keys=list(obs_keys),
        action_dim=action_dim,
        target_pos_key=target_pos_key,
    )


@pytest.mark.unit
def test_validate_passes_on_exact_match() -> None:
    spec = _spec(action_dim=6, obs_keys=("ee_pos", "target_pos"))
    env = FakeEnv(action_dim=6, obs_space_keys=["ee_pos", "target_pos"])
    # Should return None without raising.
    assert validate_spec_against_env(spec, env) is None


@pytest.mark.unit
def test_validate_raises_on_action_dim_mismatch() -> None:
    spec = _spec(name="action_mismatch_spec", action_dim=6)
    env = FakeEnv(action_dim=8, obs_space_keys=["ee_pos", "target_pos"])
    with pytest.raises(RobotSpecMismatch, match=r"action_mismatch_spec.*6.*8"):
        validate_spec_against_env(spec, env)


@pytest.mark.unit
def test_validate_raises_on_missing_obs_keys() -> None:
    spec = _spec(action_dim=6, obs_keys=("ee_pos", "target_pos", "joints"))
    env = FakeEnv(action_dim=6, obs_space_keys=["ee_pos"])
    with pytest.raises(RobotSpecMismatch, match=r"joints"):
        validate_spec_against_env(spec, env)


@pytest.mark.unit
def test_validate_raises_message_lists_all_missing_keys() -> None:
    spec = _spec(action_dim=6, obs_keys=("ee_pos", "target_pos", "joints"))
    env = FakeEnv(action_dim=6, obs_space_keys=["ee_pos"])
    with pytest.raises(RobotSpecMismatch) as exc_info:
        validate_spec_against_env(spec, env)
    msg = str(exc_info.value)
    assert "joints" in msg
    assert "target_pos" in msg


@pytest.mark.unit
def test_validate_accepts_env_with_extra_obs_keys() -> None:
    spec = _spec(action_dim=6, obs_keys=("ee_pos",))
    env = FakeEnv(action_dim=6, obs_space_keys=["ee_pos", "extra", "more"])
    # Spec subset of env is OK — no raise.
    assert validate_spec_against_env(spec, env) is None


@pytest.mark.unit
def test_robotspec_mismatch_is_value_error() -> None:
    assert issubclass(RobotSpecMismatch, ValueError)
