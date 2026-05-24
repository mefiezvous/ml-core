# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for ObservationBuilder."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mlcore.observation import ObservationBuilder
from mlcore.robots import get

# Legacy reference implementation copied verbatim from
# playground/data/pipeline.py::_build_state (lerobot-playground-portfolio).
_LEGACY_STATE_NAMES = [
    "ee_x",
    "ee_y",
    "ee_z",
    "cube_x",
    "cube_y",
    "cube_z",
    "q0",
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "dx",
    "dy",
    "dz",
]


def _legacy_build_state(obs: dict[str, np.ndarray]) -> np.ndarray:
    ee_pos = np.asarray(obs["ee_pos"], dtype=np.float32)
    cube_pos = np.asarray(obs["cube_pos"], dtype=np.float32)
    joint_pos = np.asarray(obs["joint_positions"], dtype=np.float32)
    ee_to_cube = cube_pos - ee_pos
    return np.concatenate([ee_pos, cube_pos, joint_pos, ee_to_cube])


@pytest.fixture
def cube_obs() -> dict[str, np.ndarray]:
    return {
        "ee_pos": np.ones(3, dtype=np.float32),
        "cube_pos": 2.0 * np.ones(3, dtype=np.float32),
        "joint_positions": np.ones(7, dtype=np.float32),
    }


@pytest.mark.unit
def test_builds_legacy_16dim_vector_identically(cube_obs: dict[str, np.ndarray]) -> None:
    builder = ObservationBuilder(
        keys=("ee_pos", "cube_pos", "joint_positions"),
        relational=(("cube_pos", "ee_pos"),),
        key_shapes=(3, 3, 7),
    )
    out = builder.build(cube_obs)
    expected = _legacy_build_state(cube_obs)
    assert out.shape == (16,)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, expected)


@pytest.mark.unit
def test_state_dim_matches_output_shape(cube_obs: dict[str, np.ndarray]) -> None:
    builder = ObservationBuilder(
        keys=("ee_pos", "cube_pos", "joint_positions"),
        relational=(("cube_pos", "ee_pos"),),
        key_shapes=(3, 3, 7),
    )
    assert builder.state_dim == 16
    assert builder.build(cube_obs).shape[0] == builder.state_dim


@pytest.mark.unit
def test_state_names_length_matches_state_dim() -> None:
    builder = ObservationBuilder(
        keys=("ee_pos", "cube_pos", "joint_positions"),
        relational=(("cube_pos", "ee_pos"),),
        key_shapes=(3, 3, 7),
    )
    assert len(builder.state_names) == builder.state_dim


@pytest.mark.unit
def test_state_names_match_legacy_for_cube_reach() -> None:
    builder = ObservationBuilder(
        keys=("ee_pos", "cube_pos", "joint_positions"),
        relational=(("cube_pos", "ee_pos"),),
        key_shapes=(3, 3, 7),
    )
    assert builder.state_names == _LEGACY_STATE_NAMES


@pytest.mark.unit
def test_missing_key_raises_key_error() -> None:
    builder = ObservationBuilder(
        keys=("ee_pos", "cube_pos"),
        key_shapes=(3, 3),
    )
    with pytest.raises(KeyError, match="cube_pos"):
        builder.build({"ee_pos": np.zeros(3, dtype=np.float32)})


@pytest.mark.unit
def test_relational_mismatched_shapes_raises_value_error() -> None:
    builder = ObservationBuilder(
        keys=("ee_pos", "joint_positions"),
        relational=(("joint_positions", "ee_pos"),),
        key_shapes=(3, 7),
    )
    with pytest.raises(ValueError, match="shape"):
        builder.build(
            {
                "ee_pos": np.zeros(3, dtype=np.float32),
                "joint_positions": np.zeros(7, dtype=np.float32),
            }
        )


@pytest.mark.unit
def test_relational_pair_must_be_in_keys() -> None:
    with pytest.raises(ValueError, match="not in keys"):
        ObservationBuilder(
            keys=("ee_pos",),
            relational=(("cube_pos", "ee_pos"),),
            key_shapes=(3,),
        )


@pytest.mark.unit
def test_key_shapes_length_must_match_keys() -> None:
    with pytest.raises(ValueError, match="key_shapes"):
        ObservationBuilder(
            keys=("ee_pos", "cube_pos"),
            key_shapes=(3,),
        )


@pytest.mark.unit
def test_frozen_invariant() -> None:
    builder = ObservationBuilder(keys=("ee_pos",), key_shapes=(3,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        builder.keys = ("other",)  # type: ignore[misc]


@pytest.mark.unit
def test_from_spec_round_trips_cube_reach_v1(cube_obs: dict[str, np.ndarray]) -> None:
    spec = get("cube_reach_v1")
    # The spec uses ``joints`` as joint key name; remap the fixture so we
    # exercise the same key contract as the spec.
    obs_for_spec = {
        "ee_pos": cube_obs["ee_pos"],
        "cube_pos": cube_obs["cube_pos"],
        "joints": cube_obs["joint_positions"],
    }
    builder = ObservationBuilder.from_spec(spec, obs_for_spec)
    out = builder.build(obs_for_spec)
    # ee_pos(3) + cube_pos(3) + joints(7) + (cube_pos - ee_pos)(3) = 16
    assert out.shape == (16,)
    # Verify the relational block matches cube - ee
    np.testing.assert_array_equal(out[-3:], cube_obs["cube_pos"] - cube_obs["ee_pos"])


@pytest.mark.unit
def test_from_spec_handles_extra_obs_keys() -> None:
    from mlcore.robots import RobotSpec

    spec = RobotSpec(
        name="with_extra",
        n_joints=7,
        obs_keys=["ee_pos", "cube_pos"],
        action_dim=8,
        target_pos_key="cube_pos",
        extra_obs_keys=("gripper",),
        relational_features=(("cube_pos", "ee_pos"),),
    )
    obs = {
        "ee_pos": np.zeros(3, dtype=np.float32),
        "cube_pos": np.ones(3, dtype=np.float32),
        "gripper": np.array([0.5], dtype=np.float32),
    }
    builder = ObservationBuilder.from_spec(spec, obs)
    out = builder.build(obs)
    # 3 + 3 + 1 + 3 = 10
    assert out.shape == (10,)
    assert "gripper" in builder.keys
