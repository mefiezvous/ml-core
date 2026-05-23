# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Parameterizable observation builder.

Replaces the hardcoded ``_build_state`` in
``lerobot-playground-portfolio/src/playground/data/pipeline.py`` with a
task-agnostic, declarative concatenator from an observation dict to a flat
float32 vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mlcore.robots.base import RobotSpec


@dataclass(frozen=True)
class ObservationBuilder:
    """Parameterizable obs dict → flat state vector.

    Concatenates raw obs values (in ``keys`` order) followed by relational
    deltas (``obs[b] - obs[a]`` for each ``(b, a)`` in ``relational``).
    All outputs are float32.

    The ``key_shapes`` argument captures the flat length of each key in
    ``keys`` and is required at construction. This is **design option (a)** —
    explicit shapes — chosen over (b) lazy caching or (c) sample-obs
    sniffing, because:

    * It keeps the dataclass truly frozen and hashable (no mutable cache).
    * ``state_dim`` and ``state_names`` are pure properties available
      without ever calling ``build``, which matters for downstream config
      (LeRobotDataset feature schemas, policy input layers).
    * Construction-time validation catches misconfigurations before any
      observation flows through the pipeline.

    The ``from_spec`` classmethod handles the convenience case where shapes
    are resolved from a sample observation dict.

    Args:
        keys: Observation dict keys to extract, in concatenation order.
        relational: Pairs ``(b_key, a_key)``; each yields ``obs[b] - obs[a]``
            appended after the raw keys. Both names must appear in ``keys``.
        key_shapes: Flat length of each key, parallel to ``keys``. Required.

    Raises:
        ValueError: If ``key_shapes`` length differs from ``keys``,
            or if a relational pair references a key not in ``keys``.
    """

    keys: tuple[str, ...]
    key_shapes: tuple[int, ...]
    relational: tuple[tuple[str, str], ...] = field(default=())

    def __post_init__(self) -> None:
        if len(self.key_shapes) != len(self.keys):
            raise ValueError(
                f"key_shapes length ({len(self.key_shapes)}) must match "
                f"keys length ({len(self.keys)})"
            )
        key_set = set(self.keys)
        for b_key, a_key in self.relational:
            if b_key not in key_set or a_key not in key_set:
                raise ValueError(
                    f"relational pair ({b_key!r}, {a_key!r}) references a key "
                    f"not in keys {self.keys!r}"
                )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_spec(
        cls, spec: RobotSpec, obs_sample: dict[str, Any]
    ) -> ObservationBuilder:
        """Build from a :class:`RobotSpec` using ``obs_sample`` to resolve shapes.

        The ``keys`` are ``spec.obs_keys`` followed by ``spec.extra_obs_keys``,
        in that order. Shapes are read from the corresponding entries in
        ``obs_sample`` (each value is flattened via ``np.asarray(...).reshape(-1)``).

        Args:
            spec: Source robot specification.
            obs_sample: A representative observation dict; every key referenced
                by the spec must be present.

        Returns:
            A new immutable :class:`ObservationBuilder`.

        Raises:
            KeyError: If a key declared in the spec is missing from ``obs_sample``.
        """
        keys: tuple[str, ...] = tuple(spec.obs_keys) + tuple(spec.extra_obs_keys)
        shapes: list[int] = []
        for k in keys:
            if k not in obs_sample:
                raise KeyError(
                    f"key {k!r} declared in spec {spec.name!r} is missing from obs_sample"
                )
            shapes.append(int(np.asarray(obs_sample[k]).reshape(-1).shape[0]))
        return cls(
            keys=keys,
            key_shapes=tuple(shapes),
            relational=tuple(spec.relational_features),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, obs: dict[str, Any]) -> np.ndarray[Any, Any]:
        """Return the concatenated float32 vector for ``obs``.

        Args:
            obs: Observation dict. Must contain every key in ``self.keys``.

        Returns:
            1-D ``np.ndarray`` of dtype ``float32`` and length ``self.state_dim``.

        Raises:
            KeyError: If ``obs`` is missing any key in ``self.keys``.
            ValueError: If a relational pair's two values have different shapes.
        """
        parts: list[np.ndarray[Any, Any]] = []
        resolved: dict[str, np.ndarray[Any, Any]] = {}
        for key, expected_len in zip(self.keys, self.key_shapes, strict=True):
            if key not in obs:
                raise KeyError(key)
            arr = np.asarray(obs[key], dtype=np.float32).reshape(-1)
            if arr.shape[0] != expected_len:
                raise ValueError(
                    f"key {key!r}: expected length {expected_len}, got {arr.shape[0]}"
                )
            resolved[key] = arr
            parts.append(arr)

        for b_key, a_key in self.relational:
            b_arr = resolved[b_key]
            a_arr = resolved[a_key]
            if b_arr.shape != a_arr.shape:
                raise ValueError(
                    f"relational pair ({b_key!r}, {a_key!r}) has mismatched "
                    f"shape: {b_arr.shape} vs {a_arr.shape}"
                )
            parts.append(b_arr - a_arr)

        result: np.ndarray[Any, Any] = np.concatenate(parts).astype(np.float32, copy=False)
        return result

    @property
    def state_dim(self) -> int:
        """Total dimension of the output vector.

        Equals ``sum(key_shapes) + 3 * len(relational)`` when each relational
        pair operates on length-3 vectors. More generally it is
        ``sum(key_shapes) + sum(len(b)) for (b, _) in relational)``.
        """
        raw = sum(self.key_shapes)
        key_to_len = dict(zip(self.keys, self.key_shapes, strict=True))
        rel = sum(key_to_len[b] for b, _ in self.relational)
        return raw + rel

    @property
    def state_names(self) -> list[str]:
        """Per-dimension feature names for traceability.

        Naming rules:

        * For ``ee_pos`` / ``cube_pos`` of length 3 → ``ee_x``, ``ee_y``, ``ee_z``
          (drop the ``_pos`` suffix and append ``_x|_y|_z``).
        * For ``joint_positions`` / ``joints`` of length N → ``q0..q{N-1}``.
        * Otherwise → ``{key}_{i}``.
        * Relational pair ``(b, a)`` of length 3 between ``*_pos`` keys → ``dx``,
          ``dy``, ``dz``. Otherwise → ``d_{b}_{a}_{i}``.
        """
        names: list[str] = []
        for key, n in zip(self.keys, self.key_shapes, strict=True):
            names.extend(_per_key_names(key, n))
        key_to_len = dict(zip(self.keys, self.key_shapes, strict=True))
        for b_key, a_key in self.relational:
            names.extend(_relational_names(b_key, a_key, key_to_len[b_key]))
        return names


_XYZ = ("x", "y", "z")


def _per_key_names(key: str, n: int) -> list[str]:
    if key.endswith("_pos") and n == 3:
        prefix = key[: -len("_pos")]
        return [f"{prefix}_{axis}" for axis in _XYZ]
    if key in {"joint_positions", "joints"}:
        return [f"q{i}" for i in range(n)]
    return [f"{key}_{i}" for i in range(n)]


def _relational_names(b_key: str, a_key: str, n: int) -> list[str]:
    if n == 3 and b_key.endswith("_pos") and a_key.endswith("_pos"):
        return [f"d{axis}" for axis in _XYZ]
    return [f"d_{b_key}_{a_key}_{i}" for i in range(n)]
