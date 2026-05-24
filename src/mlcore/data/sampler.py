# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Smart samplers for multi-task and success-only training.

The samplers here are :class:`torch.utils.data.Sampler` subclasses that
work on any dataset whose items are mappings (e.g. ``LeRobotDataset``
items, or a :class:`torch.utils.data.Subset` thereof).

Multi-task balancing matters when one task dominates the dataset — naive
shuffling under-samples rare tasks and biases the policy. The
:class:`MultiTaskBalancedSampler` groups frames by ``task_id``, picks
``tasks_per_batch`` task_ids per batch, then draws an even slice from
each.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import torch
from loguru import logger
from torch.utils.data import Sampler

from mlcore.data.filter import SuccessOnlyFilter


def _read_task_id(dataset: Any, index: int) -> str:
    """Return the ``task_id`` field of ``dataset[index]`` as a string.

    LeRobotDataset items are dicts; ``task_id`` is stored as a string
    feature. Tensors / numpy scalars are coerced via :class:`str`.
    """
    raw: Any = dataset[index]["task_id"]
    if hasattr(raw, "item") and not isinstance(raw, str | bytes):
        try:
            raw = raw.item()
        except (ValueError, TypeError):
            pass
    return str(raw)


class MultiTaskBalancedSampler(Sampler[int]):
    """Yield indices such that each batch is balanced across task_ids.

    On every iteration the sampler picks ``tasks_per_batch`` distinct
    task_ids uniformly at random, then draws ``batch_size //
    tasks_per_batch`` frames per chosen task. The result is a flat stream
    of indices: when consumed by a :class:`torch.utils.data.DataLoader`
    with the same ``batch_size``, each batch contains exactly that
    distribution.

    Args:
        dataset: Any dataset whose items are mappings carrying a
            ``"task_id"`` string field. LeRobotDataset items satisfy this
            contract (``dataset[i]["task_id"]``); a
            :class:`torch.utils.data.Subset` of one also does.
        tasks_per_batch: How many distinct task_ids to draw from per
            batch. Must be ``>= 1`` and ``<=`` the number of unique
            task_ids in ``dataset``.
        batch_size: Size of each balanced batch. Must be divisible by
            ``tasks_per_batch``.
        num_samples: Total number of indices the sampler yields. Defaults
            to ``len(dataset)``, rounded down to a multiple of
            ``batch_size``.
        seed: Seed for the internal :class:`torch.Generator`.

    Raises:
        ValueError: If arguments are inconsistent (e.g.
            ``batch_size % tasks_per_batch != 0``, ``tasks_per_batch``
            exceeds the number of unique tasks, or the dataset is empty).
    """

    def __init__(
        self,
        dataset: Any,
        tasks_per_batch: int,
        batch_size: int,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if tasks_per_batch < 1:
            raise ValueError(f"tasks_per_batch must be >= 1, got {tasks_per_batch}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if batch_size % tasks_per_batch != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by "
                f"tasks_per_batch ({tasks_per_batch})"
            )
        n: int = len(dataset)
        if n == 0:
            raise ValueError("MultiTaskBalancedSampler requires a non-empty dataset.")

        self.dataset: Any = dataset
        self.tasks_per_batch: int = tasks_per_batch
        self.batch_size: int = batch_size
        self.per_task: int = batch_size // tasks_per_batch
        self.seed: int = seed

        # Build the task_id -> [indices] map once at construction time.
        buckets: dict[str, list[int]] = defaultdict(list)
        for i in range(n):
            buckets[_read_task_id(dataset, i)].append(i)
        self._buckets: dict[str, list[int]] = dict(buckets)
        self._task_ids: list[str] = sorted(self._buckets.keys())

        if tasks_per_batch > len(self._task_ids):
            raise ValueError(
                f"tasks_per_batch ({tasks_per_batch}) exceeds number of "
                f"unique task_ids ({len(self._task_ids)})."
            )

        total: int = num_samples if num_samples is not None else n
        # Round down to a whole number of batches so DataLoader(drop_last=True)
        # gives consistent shapes.
        self._num_samples: int = (total // batch_size) * batch_size
        if self._num_samples == 0:
            logger.warning(
                "MultiTaskBalancedSampler will yield 0 samples (num_samples={}, batch_size={}).",
                total,
                batch_size,
            )

    def __iter__(self) -> Iterator[int]:
        """Yield ``self._num_samples`` indices grouped into balanced batches."""
        gen: torch.Generator = torch.Generator()
        gen.manual_seed(self.seed)

        n_batches: int = self._num_samples // self.batch_size
        all_tasks: list[str] = self._task_ids
        n_tasks: int = len(all_tasks)

        for _ in range(n_batches):
            # Pick tasks_per_batch distinct task_ids uniformly at random.
            task_perm: torch.Tensor = torch.randperm(n_tasks, generator=gen)
            chosen: list[str] = [all_tasks[int(task_perm[k])] for k in range(self.tasks_per_batch)]

            for task in chosen:
                bucket: list[int] = self._buckets[task]
                m: int = len(bucket)
                # Sample per_task indices with replacement if bucket is small,
                # otherwise without replacement for diversity within the batch.
                if m >= self.per_task:
                    perm: torch.Tensor = torch.randperm(m, generator=gen)
                    for k in range(self.per_task):
                        yield bucket[int(perm[k])]
                else:
                    for _ in range(self.per_task):
                        idx: int = int(torch.randint(0, m, (1,), generator=gen).item())
                        yield bucket[idx]

    def __len__(self) -> int:
        return self._num_samples


class SuccessOnlySampler(Sampler[int]):
    """Wrap :class:`SuccessOnlyFilter`, then yield a shuffled permutation.

    Convenience sampler for the common case of training only on successful
    frames. Equivalent to constructing a DataLoader over
    ``SuccessOnlyFilter()(dataset)`` with ``shuffle=True``, but exposed as
    a Sampler so it slots into the same Hydra-driven plumbing as
    :class:`MultiTaskBalancedSampler`.

    Args:
        dataset: Any dataset whose items are mappings with a ``"success"``
            field (per :mod:`mlcore.data.schema`).
        seed: Seed for the shuffle :class:`torch.Generator`.
    """

    def __init__(self, dataset: Any, seed: int = 0) -> None:
        super().__init__()
        self.dataset: Any = dataset
        self.seed: int = seed
        subset: Any = SuccessOnlyFilter()(dataset)
        # ``Subset`` exposes ``.indices`` — these are the kept positions in
        # the *underlying* dataset, which is exactly what the consumer
        # DataLoader will index into.
        self._indices: list[int] = list(subset.indices)

    def __iter__(self) -> Iterator[int]:
        """Yield the kept indices in a deterministic shuffled order."""
        gen: torch.Generator = torch.Generator()
        gen.manual_seed(self.seed)
        n: int = len(self._indices)
        if n == 0:
            return
        perm: torch.Tensor = torch.randperm(n, generator=gen)
        for k in range(n):
            yield self._indices[int(perm[k])]

    def __len__(self) -> int:
        return len(self._indices)


__all__: tuple[str, ...] = ("MultiTaskBalancedSampler", "SuccessOnlySampler")
