# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Data collection package.

Provides the :class:`Episode` data model, the :class:`Collector` Protocol,
concrete collectors (scripted, policy-rollout, teleop stub), pure curation
helpers, and a :class:`HubSink` to persist episodes to a LeRobotDataset v3.0
on disk and optionally push to the HuggingFace Hub.

This module is the canonical home for data-collection primitives — the
``playground.data.pipeline`` module in ``lerobot-playground-portfolio``
remains for backward compatibility but new consumers should depend on
``mlcore.collection`` directly.
"""

from mlcore.collection.curation import (
    balance_by_task as balance_by_task,
)
from mlcore.collection.curation import (
    dedupe_by_trajectory_hash as dedupe_by_trajectory_hash,
)
from mlcore.collection.curation import (
    filter_by_success as filter_by_success,
)
from mlcore.collection.interfaces import Collector as Collector
from mlcore.collection.interfaces import EnvLike as EnvLike
from mlcore.collection.interfaces import Episode as Episode
from mlcore.collection.rollout import PolicyRolloutCollector as PolicyRolloutCollector
from mlcore.collection.scripted import ScriptedCollector as ScriptedCollector
from mlcore.collection.scripted import ScriptedReachPolicy as ScriptedReachPolicy
from mlcore.collection.sink import HubSink as HubSink
from mlcore.collection.teleop import TeleopCollector as TeleopCollector
