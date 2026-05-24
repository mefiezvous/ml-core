# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""LeRobotDataset v3.0 sink — persists episodes to disk + optional Hub push.

Generalisation of ``DemoCollector.to_lerobot_dataset`` from
``playground.data.pipeline``. The sink is decoupled from any specific
collector and accepts an :class:`ObservationBuilder` so each consumer can
control its own state-vector layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from mlcore.collection.interfaces import Episode
from mlcore.data.schema import (
    ACTION_KEY,
    OBS_STATE_KEY,
    SUCCESS_KEY,
    TASK_DESCRIPTION_KEY,
    TASK_ID_KEY,
    build_features,
)
from mlcore.observation.builder import ObservationBuilder


# Try both lerobot import paths (0.5.x reorg) so the sink works against any
# install. Fall back to ``None`` so import of this module never fails.
def _resolve_lerobot_dataset() -> Any:
    """Return the ``LeRobotDataset`` class or ``None`` if lerobot is missing."""
    try:  # pragma: no cover - import probing
        from lerobot.datasets.lerobot_dataset import LeRobotDataset as _Cls
    except ImportError:  # pragma: no cover
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as _Cls
        except ImportError:
            return None
    return _Cls


LeRobotDataset: Any = _resolve_lerobot_dataset()


class HubSink:
    """Persist collected episodes to a LeRobotDataset v3.0 on disk.

    Optionally pushes the dataset to the HuggingFace Hub under ``repo_id``.

    Args:
        repo_id: HuggingFace dataset repo id, e.g.
            ``"mefiezvous/cube-reach-v1-dataset"``.
        fps: Control frequency used to tag the LeRobotDataset.
        action_dim: Dimensionality of the action vector. Defaults to 8.
    """

    def __init__(self, repo_id: str, fps: int = 20, action_dim: int = 8) -> None:
        self._repo_id = repo_id
        self._fps = fps
        self._action_dim = action_dim

    def write(
        self,
        episodes: list[Episode],
        root: Path,
        obs_builder: ObservationBuilder,
        push_to_hub: bool = False,
    ) -> Any:
        """Materialise ``episodes`` as a LeRobotDataset v3.0 under ``root``.

        Args:
            episodes: Episodes to persist.
            root: Local directory where the dataset is written.
            obs_builder: Builder used to flatten each raw obs dict into
                the LeRobotDataset ``observation.state`` vector.
            push_to_hub: When ``True``, push the dataset to the Hub
                under ``self._repo_id`` after writing.

        Returns:
            The created ``LeRobotDataset`` instance.

        Raises:
            RuntimeError: If ``lerobot`` is not importable.
        """
        if LeRobotDataset is None:
            raise RuntimeError("lerobot is required for dataset export. Install with: uv sync")

        features: dict[str, Any] = build_features(
            state_dim=obs_builder.state_dim,
            state_names=obs_builder.state_names,
            action_dim=self._action_dim,
        )

        dataset = LeRobotDataset.create(
            repo_id=self._repo_id,
            fps=self._fps,
            root=root,
            features=features,
        )

        total_frames = 0
        for ep_idx, episode in enumerate(episodes):
            success_arr = np.array([episode.success], dtype=np.bool_)
            for obs, action in zip(episode.observations, episode.actions, strict=False):
                dataset.add_frame(
                    {
                        OBS_STATE_KEY: obs_builder.build(obs),
                        ACTION_KEY: np.asarray(action, dtype=np.float32),
                        TASK_ID_KEY: episode.task_id,
                        TASK_DESCRIPTION_KEY: episode.task_description,
                        SUCCESS_KEY: success_arr,
                    }
                )
                total_frames += 1
            dataset.save_episode(task=episode.task_description)
            logger.debug(f"Saved episode {ep_idx + 1}/{len(episodes)}")

        logger.info(f"Dataset saved to {root} | episodes={len(episodes)} | frames={total_frames}")

        if push_to_hub:
            dataset.push_to_hub(self._repo_id)
            logger.info(f"Dataset pushed to hub: {self._repo_id}")

        return dataset
