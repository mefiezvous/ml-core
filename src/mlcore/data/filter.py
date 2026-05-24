# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Dataset filters for multi-task / success-only training.

The classes here are intentionally tiny, pure-Python callables that, given
a dataset object exposing ``__len__`` and ``__getitem__`` returning a
``dict``-like record, return a ``torch.utils.data.Subset`` containing only
the frames that pass the predicate.

They are Hydra-instantiable via ``_target_`` and chainable (the resulting
``Subset`` is itself a dataset that can be filtered again or wrapped by a
sampler).

Assumptions:
    * Each frame ``dataset[i]`` is a mapping (e.g. a LeRobotDataset item
      dict).
    * ``success`` entries are array-like of shape ``(1,)`` (per the feature
      schema in :mod:`mlcore.data.schema`) but scalar booleans are also
      accepted.
    * ``task_id`` entries are strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from torch.utils.data import Dataset, Subset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is a runtime dep but guard anyway
    Dataset = None  # type: ignore[assignment,misc]
    Subset = None  # type: ignore[assignment,misc]
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    """Raise ``RuntimeError`` if ``torch`` is not importable."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "torch is required for mlcore.data.filter; install with 'uv sync'."
        )


def _coerce_success(value: Any) -> bool:
    """Coerce a ``success`` field to a Python ``bool``.

    The schema declares ``success`` as shape ``(1,)`` bool, but tests and
    in-memory fakes often store scalar booleans. We accept both.
    """
    if hasattr(value, "__len__") and not isinstance(value, str | bytes):
        if len(value) == 0:
            return False
        return bool(value[0])
    return bool(value)


@dataclass
class SuccessOnlyFilter:
    """Keep only frames whose ``success`` field evaluates truthy.

    Hydra usage::

        filter:
          _target_: mlcore.data.filter.SuccessOnlyFilter

    The callable returns a :class:`torch.utils.data.Subset` so it composes
    naturally with samplers and other filters.
    """

    def __call__(self, dataset: Any) -> Any:
        """Apply the filter and return a ``Subset`` of successful frames.

        Args:
            dataset: Any object supporting ``len(dataset)`` and
                ``dataset[i]`` returning a mapping with a ``"success"`` key.

        Returns:
            A :class:`torch.utils.data.Subset` containing only the frames
            for which ``success`` is truthy.

        Raises:
            RuntimeError: If ``torch`` is not importable.
            KeyError: If a frame is missing the ``"success"`` key.
        """
        _require_torch()
        indices: list[int] = [
            i for i in range(len(dataset)) if _coerce_success(dataset[i]["success"])
        ]
        return Subset(dataset, indices)


@dataclass
class TaskFilter:
    """Keep only frames whose ``task_id`` is in the allow-list.

    Hydra usage::

        filter:
          _target_: mlcore.data.filter.TaskFilter
          task_ids: [cube_reach_v1, cube_stack_v1]

    Args:
        task_ids: Allow-list of task identifiers. Must be a non-empty
            iterable of strings.
    """

    task_ids: list[str]

    def __post_init__(self) -> None:
        # Normalise (Hydra may pass a ListConfig/tuple).
        ids: list[str] = list(self.task_ids)
        if not ids:
            raise ValueError("TaskFilter requires at least one task_id.")
        self.task_ids = ids
        self._allow: frozenset[str] = frozenset(ids)

    def __call__(self, dataset: Any) -> Any:
        """Apply the filter and return a ``Subset`` of matching frames.

        Args:
            dataset: Any object supporting ``len(dataset)`` and
                ``dataset[i]`` returning a mapping with a ``"task_id"`` key.

        Returns:
            A :class:`torch.utils.data.Subset` containing only the frames
            whose ``task_id`` is in the allow-list.

        Raises:
            RuntimeError: If ``torch`` is not importable.
            KeyError: If a frame is missing the ``"task_id"`` key.
        """
        _require_torch()
        allow: frozenset[str] = self._allow
        indices: list[int] = [
            i for i in range(len(dataset)) if str(dataset[i]["task_id"]) in allow
        ]
        return Subset(dataset, indices)


__all__: tuple[str, ...] = ("SuccessOnlyFilter", "TaskFilter")
