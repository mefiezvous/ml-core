# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""LeRobotDataset v3.0 feature-schema builders shared across consumers.

This module centralises the ``features`` dict passed to
``LeRobotDataset.create`` so PUBLIC (``lerobot-playground-portfolio``) and
PRIVATE (``my-robot-stack``) layers describe their datasets identically.

The schema is intentionally richer than the previous PUBLIC inline version:
it carries per-frame task labels (``task_id`` + natural-language
``task_description``) and a boolean ``success`` flag, both required by the
upcoming multi-task training pipeline.
"""

from __future__ import annotations

from typing import Any

OBS_STATE_KEY: str = "observation.state"
ACTION_KEY: str = "action"
TASK_ID_KEY: str = "task_id"
TASK_DESCRIPTION_KEY: str = "task_description"
SUCCESS_KEY: str = "success"
IMAGE_KEY_PREFIX: str = "observation.images."


def image_key(camera_name: str) -> str:
    """Return the LeRobotDataset feature key for a given camera.

    Args:
        camera_name: Short camera identifier, e.g. ``"front"`` or ``"wrist"``.

    Returns:
        The fully qualified feature key, e.g. ``"observation.images.front"``.
    """
    return f"{IMAGE_KEY_PREFIX}{camera_name}"


def build_features(
    state_dim: int,
    state_names: list[str],
    action_dim: int = 8,
    image_keys: list[str] | None = None,
    image_shape: tuple[int, int, int] = (480, 640, 3),
) -> dict[str, Any]:
    """Build a LeRobotDataset v3.0 ``features`` dict.

    The returned dict always contains ``observation.state``, ``action``,
    ``task_id``, ``task_description`` and ``success``. One
    ``observation.images.{cam}`` entry is added for every camera in
    ``image_keys``.

    Args:
        state_dim: Dimensionality of the proprioceptive state vector.
        state_names: Per-component labels for the state vector. Must have
            length ``state_dim``.
        action_dim: Dimensionality of the action vector. Defaults to 8
            (7 joints + 1 gripper for a Franka-class arm).
        image_keys: Short camera identifiers (without the
            ``observation.images.`` prefix). Defaults to no cameras.
        image_shape: ``(H, W, C)`` of every image entry. Defaults to
            ``(480, 640, 3)``.

    Returns:
        A dict suitable for passing to ``LeRobotDataset.create(features=...)``.

    Raises:
        ValueError: If ``len(state_names) != state_dim``.
    """
    if len(state_names) != state_dim:
        raise ValueError(f"state_names has length {len(state_names)} but state_dim is {state_dim}")

    cameras: list[str] = list(image_keys) if image_keys is not None else []

    features: dict[str, Any] = {
        OBS_STATE_KEY: {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": list(state_names),
        },
    }

    for cam in cameras:
        features[image_key(cam)] = {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        }

    features[ACTION_KEY] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": None,
    }
    features[TASK_ID_KEY] = {
        "dtype": "string",
        "shape": (),
        "names": None,
    }
    features[TASK_DESCRIPTION_KEY] = {
        "dtype": "string",
        "shape": (),
        "names": None,
    }
    features[SUCCESS_KEY] = {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    }

    return features
