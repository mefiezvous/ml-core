# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for RobotSpec, registry, GenericScriptedPolicy, and build_train_config."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mlcore.robots import (
    GenericScriptedPolicy,
    RobotSpec,
    build_train_config,
    get,
    list_specs,
    register,
)


@pytest.mark.unit
def test_robot_spec_creation() -> None:
    spec = RobotSpec(
        name="cube_reach",
        n_joints=7,
        obs_keys=["ee_pos", "cube_pos"],
        action_dim=7,
        target_pos_key="cube_pos",
    )
    assert spec.name == "cube_reach"
    assert spec.action_dim == 7
    assert spec.n_joints == 7


@pytest.mark.unit
def test_robot_spec_requires_target_pos_key() -> None:
    """Every spec must declare its target explicitly — no implicit cube default."""
    with pytest.raises(TypeError):
        RobotSpec(  # type: ignore[call-arg]
            name="missing_target",
            n_joints=7,
            obs_keys=["ee_pos"],
            action_dim=7,
        )


@pytest.mark.unit
def test_robot_spec_extra_obs_keys_default_empty_tuple() -> None:
    spec = RobotSpec(
        name="extra_default",
        n_joints=7,
        obs_keys=["ee_pos"],
        action_dim=7,
        target_pos_key="ee_pos",
    )
    assert spec.extra_obs_keys == ()
    assert isinstance(spec.extra_obs_keys, tuple)


@pytest.mark.unit
def test_robot_spec_relational_features_default_empty_tuple() -> None:
    spec = RobotSpec(
        name="relational_default",
        n_joints=7,
        obs_keys=["ee_pos"],
        action_dim=7,
        target_pos_key="ee_pos",
    )
    assert spec.relational_features == ()
    assert isinstance(spec.relational_features, tuple)


@pytest.mark.unit
def test_robot_spec_accepts_extra_obs_keys_and_relational_features() -> None:
    spec = RobotSpec(
        name="rich",
        n_joints=7,
        obs_keys=["ee_pos", "cube_pos"],
        action_dim=8,
        target_pos_key="cube_pos",
        extra_obs_keys=("gripper", "force"),
        relational_features=(("cube_pos", "ee_pos"),),
    )
    assert spec.extra_obs_keys == ("gripper", "force")
    assert spec.relational_features == (("cube_pos", "ee_pos"),)


@pytest.mark.unit
def test_robot_spec_is_frozen() -> None:
    spec = RobotSpec(
        name="frozen_test",
        n_joints=7,
        obs_keys=["ee_pos"],
        action_dim=7,
        target_pos_key="ee_pos",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "mutated"  # type: ignore[misc]


@pytest.mark.unit
def test_spec_registry_register_and_get() -> None:
    spec = RobotSpec(
        name="reg_test",
        n_joints=6,
        obs_keys=["ee_pos"],
        action_dim=6,
        target_pos_key="ee_pos",
    )
    register(spec)
    assert get("reg_test") is spec


@pytest.mark.unit
def test_spec_registry_list() -> None:
    spec = RobotSpec(
        name="list_test",
        n_joints=6,
        obs_keys=["ee_pos"],
        action_dim=6,
        target_pos_key="ee_pos",
    )
    register(spec)
    assert "list_test" in list_specs()


@pytest.mark.unit
def test_spec_registry_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get("unknown_spec_xyz")


@pytest.mark.unit
def test_generic_scripted_policy_shape() -> None:
    spec = RobotSpec(
        name="shape_test",
        n_joints=7,
        obs_keys=["ee_pos", "cube_pos"],
        action_dim=7,
        target_pos_key="cube_pos",
    )
    policy = GenericScriptedPolicy(spec)
    action = policy.select_action({"ee_pos": np.zeros(3), "cube_pos": np.ones(3)})
    assert action.shape == (7,)


@pytest.mark.unit
def test_generic_scripted_policy_direction() -> None:
    spec = RobotSpec(
        name="dir_test",
        n_joints=7,
        obs_keys=["ee_pos", "cube_pos"],
        action_dim=7,
        target_pos_key="cube_pos",
    )
    policy = GenericScriptedPolicy(spec, gain=5.0)
    ee_pos = np.zeros(3, dtype=np.float32)
    cube_pos = np.array([0.5, 0.0, 0.0], dtype=np.float32)

    def _reward(ep: np.ndarray, cp: np.ndarray) -> float:
        return float(1.0 - np.tanh(5.0 * np.linalg.norm(cp - ep)))

    first_reward = _reward(ee_pos, cube_pos)
    for _ in range(50):
        action = policy.select_action({"ee_pos": ee_pos.copy(), "cube_pos": cube_pos.copy()})
        ee_pos = np.clip(ee_pos + action[:3] * 0.1, -10.0, 10.0)
    assert _reward(ee_pos, cube_pos) > first_reward


@pytest.mark.unit
def test_config_builder_generates_yaml() -> None:
    spec = RobotSpec(
        name="yaml_test",
        n_joints=6,
        obs_keys=["ee_pos"],
        action_dim=6,
        target_pos_key="ee_pos",
    )
    cfg = build_train_config(spec, policy_type="act")
    assert cfg["robot_name"] == "yaml_test"
    assert cfg["policy_type"] == "act"
    assert "env" in cfg
    assert "training" in cfg


@pytest.mark.unit
def test_cube_reach_v1_registered() -> None:
    spec = get("cube_reach_v1")
    assert spec.n_joints == 7
    assert spec.action_dim == 8


@pytest.mark.unit
def test_cube_reach_v1_has_relational_feature() -> None:
    spec = get("cube_reach_v1")
    assert spec.relational_features == (("cube_pos", "ee_pos"),)
