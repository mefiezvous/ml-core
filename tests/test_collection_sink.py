# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``mlcore.collection.sink``."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mlcore.collection import Episode, HubSink
from mlcore.collection import sink as sink_module
from mlcore.data.schema import (
    ACTION_KEY,
    OBS_STATE_KEY,
    SUCCESS_KEY,
    TASK_DESCRIPTION_KEY,
    TASK_ID_KEY,
)
from mlcore.observation.builder import ObservationBuilder


def _make_episode(n_steps: int = 2, success: bool = True) -> Episode:
    obs: list[dict[str, Any]] = []
    actions: list[np.ndarray[Any, Any]] = []
    for _ in range(n_steps):
        obs.append(
            {
                "ee_pos": np.zeros(3, dtype=np.float32),
                "cube_pos": np.ones(3, dtype=np.float32),
                "joint_positions": np.zeros(7, dtype=np.float32),
            }
        )
        actions.append(np.zeros(8, dtype=np.float32))
    return Episode(
        observations=tuple(obs),
        actions=tuple(actions),
        rewards=tuple(0.0 for _ in range(n_steps)),
        task_id="cube_reach_v1",
        task_description="Reach the red cube",
        success=success,
    )


def _builder() -> ObservationBuilder:
    return ObservationBuilder(
        keys=("ee_pos", "cube_pos", "joint_positions"),
        key_shapes=(3, 3, 7),
        relational=(("cube_pos", "ee_pos"),),
    )


@pytest.mark.unit
def test_hubsink_raises_when_lerobot_unavailable(tmp_path: Path) -> None:
    sink = HubSink(repo_id="user/repo")
    with patch.object(sink_module, "LeRobotDataset", None):
        with pytest.raises(RuntimeError, match="lerobot is required"):
            sink.write(
                episodes=[_make_episode()],
                root=tmp_path,
                obs_builder=_builder(),
            )


@pytest.mark.unit
def test_hubsink_write_builds_features_and_calls_add_frame(tmp_path: Path) -> None:
    sink = HubSink(repo_id="user/repo", fps=30, action_dim=8)
    fake_ds = MagicMock()
    fake_cls = MagicMock()
    fake_cls.create.return_value = fake_ds

    episodes = [_make_episode(n_steps=2), _make_episode(n_steps=3)]

    with patch.object(sink_module, "LeRobotDataset", fake_cls):
        result = sink.write(
            episodes=episodes,
            root=tmp_path,
            obs_builder=_builder(),
            push_to_hub=False,
        )

    # create was called with the right repo_id / fps / root / features schema.
    fake_cls.create.assert_called_once()
    kwargs = fake_cls.create.call_args.kwargs
    assert kwargs["repo_id"] == "user/repo"
    assert kwargs["fps"] == 30
    assert kwargs["root"] == tmp_path

    features = kwargs["features"]
    assert OBS_STATE_KEY in features
    assert features[OBS_STATE_KEY]["shape"] == (16,)
    assert features[ACTION_KEY]["shape"] == (8,)
    assert TASK_ID_KEY in features
    assert TASK_DESCRIPTION_KEY in features
    assert SUCCESS_KEY in features

    # add_frame called once per step across episodes.
    assert fake_ds.add_frame.call_count == 5  # 2 + 3
    sample_payload = fake_ds.add_frame.call_args_list[0].args[0]
    assert OBS_STATE_KEY in sample_payload
    assert sample_payload[OBS_STATE_KEY].shape == (16,)
    assert sample_payload[ACTION_KEY].shape == (8,)
    assert sample_payload[TASK_ID_KEY] == "cube_reach_v1"
    assert sample_payload[TASK_DESCRIPTION_KEY] == "Reach the red cube"
    assert sample_payload[SUCCESS_KEY].dtype == np.bool_

    # save_episode called once per Episode with the description as task arg.
    assert fake_ds.save_episode.call_count == 2
    fake_ds.save_episode.assert_called_with(task="Reach the red cube")

    # push_to_hub disabled → never invoked.
    fake_ds.push_to_hub.assert_not_called()

    assert result is fake_ds


@pytest.mark.unit
def test_hubsink_push_to_hub_invokes_push(tmp_path: Path) -> None:
    sink = HubSink(repo_id="user/repo")
    fake_ds = MagicMock()
    fake_cls = MagicMock()
    fake_cls.create.return_value = fake_ds

    with patch.object(sink_module, "LeRobotDataset", fake_cls):
        sink.write(
            episodes=[_make_episode()],
            root=tmp_path,
            obs_builder=_builder(),
            push_to_hub=True,
        )

    fake_ds.push_to_hub.assert_called_once_with("user/repo")
