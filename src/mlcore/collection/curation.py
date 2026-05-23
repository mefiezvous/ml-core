# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Pure curation helpers for collected episodes.

All functions are side-effect-free and deterministic. They take a list of
:class:`Episode` and return a new list — inputs are never mutated.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict, defaultdict

import numpy as np

from mlcore.collection.interfaces import Episode


def filter_by_success(episodes: list[Episode]) -> list[Episode]:
    """Return only episodes whose ``success`` flag is ``True``.

    Args:
        episodes: Input episodes.

    Returns:
        A new list containing only successful episodes, in original order.
    """
    return [ep for ep in episodes if ep.success]


def balance_by_task(
    episodes: list[Episode], target_per_task: int | None = None
) -> list[Episode]:
    """Cap the number of episodes per ``task_id``.

    Args:
        episodes: Input episodes.
        target_per_task: Maximum number of episodes to keep per task.
            When ``None``, uses the minimum count across all tasks (so
            every task is represented equally).

    Returns:
        A new list, deterministic in order: episodes appear in the same
        relative order as the input, but each ``task_id`` is truncated
        at ``target_per_task``.
    """
    if not episodes:
        return []

    counts: dict[str, int] = defaultdict(int)
    for ep in episodes:
        counts[ep.task_id] += 1

    cap = target_per_task if target_per_task is not None else min(counts.values())

    taken: dict[str, int] = defaultdict(int)
    out: list[Episode] = []
    for ep in episodes:
        if taken[ep.task_id] < cap:
            out.append(ep)
            taken[ep.task_id] += 1
    return out


def _trajectory_hash(ep: Episode) -> str:
    """Return a stable hex hash of the flattened action sequence."""
    if not ep.actions:
        return hashlib.sha256(b"").hexdigest()
    flat = np.concatenate([np.asarray(a, dtype=np.float32).reshape(-1) for a in ep.actions])
    return hashlib.sha256(flat.tobytes()).hexdigest()


def dedupe_by_trajectory_hash(episodes: list[Episode]) -> list[Episode]:
    """Drop episodes whose flattened action sequence repeats an earlier one.

    Args:
        episodes: Input episodes.

    Returns:
        A new list preserving original order, with later duplicates removed.
    """
    seen: OrderedDict[str, None] = OrderedDict()
    out: list[Episode] = []
    for ep in episodes:
        h = _trajectory_hash(ep)
        if h in seen:
            continue
        seen[h] = None
        out.append(ep)
    return out
