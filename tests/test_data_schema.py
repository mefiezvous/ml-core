# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``mlcore.data.schema``."""

from __future__ import annotations

import pytest

from mlcore.data.schema import (
    ACTION_KEY,
    IMAGE_KEY_PREFIX,
    OBS_STATE_KEY,
    SUCCESS_KEY,
    TASK_DESCRIPTION_KEY,
    TASK_ID_KEY,
    build_features,
    image_key,
)


@pytest.mark.unit
def test_image_key_prefix_constant() -> None:
    """The image-key prefix constant matches the LeRobotDataset convention."""
    assert IMAGE_KEY_PREFIX == "observation.images."


@pytest.mark.unit
def test_image_key_helper() -> None:
    """``image_key`` concatenates the prefix and the camera name."""
    assert image_key("front") == "observation.images.front"
    assert image_key("wrist") == "observation.images.wrist"


@pytest.mark.unit
def test_constants_match_lerobot_naming() -> None:
    """Module-level constants match the LeRobotDataset v3.0 naming."""
    assert OBS_STATE_KEY == "observation.state"
    assert ACTION_KEY == "action"
    assert TASK_ID_KEY == "task_id"
    assert TASK_DESCRIPTION_KEY == "task_description"
    assert SUCCESS_KEY == "success"


@pytest.mark.unit
def test_build_features_no_images_minimal_keys() -> None:
    """Without image keys, ``features`` contains exactly the expected schema entries."""
    state_names = [f"j{i}" for i in range(7)]
    features = build_features(state_dim=7, state_names=state_names)

    expected_keys = {
        OBS_STATE_KEY,
        ACTION_KEY,
        TASK_ID_KEY,
        TASK_DESCRIPTION_KEY,
        SUCCESS_KEY,
    }
    assert set(features.keys()) == expected_keys


@pytest.mark.unit
def test_build_features_state_entry_shape_dtype_names() -> None:
    """``observation.state`` entry has the right dtype, shape, and names."""
    state_names = ["a", "b", "c"]
    features = build_features(state_dim=3, state_names=state_names)

    state = features[OBS_STATE_KEY]
    assert state["dtype"] == "float32"
    assert state["shape"] == (3,)
    assert state["names"] == state_names


@pytest.mark.unit
def test_build_features_action_entry_default_dim() -> None:
    """``action`` defaults to dim 8, dtype float32."""
    features = build_features(state_dim=7, state_names=[f"j{i}" for i in range(7)])

    action = features[ACTION_KEY]
    assert action["dtype"] == "float32"
    assert action["shape"] == (8,)


@pytest.mark.unit
def test_build_features_action_custom_dim() -> None:
    """Custom ``action_dim`` is reflected in the shape tuple."""
    features = build_features(
        state_dim=7,
        state_names=[f"j{i}" for i in range(7)],
        action_dim=12,
    )
    assert features[ACTION_KEY]["shape"] == (12,)


@pytest.mark.unit
def test_build_features_task_and_success_entries() -> None:
    """Task labels are string-typed and success is a 1-element boolean."""
    features = build_features(state_dim=2, state_names=["x", "y"])

    assert features[TASK_ID_KEY]["dtype"] == "string"
    assert features[TASK_ID_KEY]["shape"] == ()
    assert features[TASK_DESCRIPTION_KEY]["dtype"] == "string"
    assert features[TASK_DESCRIPTION_KEY]["shape"] == ()
    assert features[SUCCESS_KEY]["dtype"] == "bool"
    assert features[SUCCESS_KEY]["shape"] == (1,)


@pytest.mark.unit
def test_build_features_with_image_keys() -> None:
    """Image keys are added under the ``observation.images.{cam}`` namespace."""
    features = build_features(
        state_dim=7,
        state_names=[f"j{i}" for i in range(7)],
        image_keys=["front", "wrist"],
    )

    assert "observation.images.front" in features
    assert "observation.images.wrist" in features

    front = features["observation.images.front"]
    assert front["dtype"] == "video"
    assert front["shape"] == (480, 640, 3)


@pytest.mark.unit
def test_build_features_custom_image_shape() -> None:
    """Custom ``image_shape`` is propagated to every image entry."""
    features = build_features(
        state_dim=2,
        state_names=["x", "y"],
        image_keys=["top"],
        image_shape=(240, 320, 3),
    )
    assert features["observation.images.top"]["shape"] == (240, 320, 3)


@pytest.mark.unit
def test_build_features_is_deterministic() -> None:
    """Two calls with the same args produce equal dicts."""
    args: dict[str, object] = {
        "state_dim": 7,
        "state_names": [f"j{i}" for i in range(7)],
        "action_dim": 8,
        "image_keys": ["front", "wrist"],
        "image_shape": (480, 640, 3),
    }
    a = build_features(**args)  # type: ignore[arg-type]
    b = build_features(**args)  # type: ignore[arg-type]
    assert a == b


@pytest.mark.unit
def test_build_features_rejects_mismatched_state_names() -> None:
    """A mismatch between ``state_dim`` and ``len(state_names)`` is an error."""
    with pytest.raises(ValueError):
        build_features(state_dim=3, state_names=["a", "b"])
