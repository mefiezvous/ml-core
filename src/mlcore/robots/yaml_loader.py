# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Load and register RobotSpec instances from a directory of YAML spec files.

The directory path is provided by the consumer (e.g. lerobot-playground-portfolio)
via an env var or Hydra config — this module never hardcodes a location, keeping
ml-core agnostic of downstream repos.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from mlcore.robots.base import RobotSpec
from mlcore.robots.registry import register

# WS-03: ``robot_specs/`` is writable via the orchestrator API. Even though the
# API validates on write, this loader re-checks the id format so a hand-edited
# or second-writer YAML cannot register a spec under an arbitrary registry key.
_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


def load_specs_from_dir(specs_dir: Path) -> list[RobotSpec]:
    """Parse every ``*.yaml`` file in ``specs_dir``, register and return the specs.

    Args:
        specs_dir: Directory containing one YAML file per spec version. If it
            does not exist, returns an empty list (no-op).

    Returns:
        The list of loaded ``RobotSpec`` instances, in filename order.

    Raises:
        ValueError: If a YAML file is malformed or violates RobotSpec invariants.
    """
    if not specs_dir.is_dir():
        return []

    specs: list[RobotSpec] = []
    for path in sorted(specs_dir.glob("*.yaml")):
        spec = _entry_to_spec(_load_entry(path), path)
        register(spec)
        specs.append(spec)
        logger.debug(f"Loaded robot spec '{spec.name}' from {path}")
    return specs


def _load_entry(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Robot spec file {path} must contain a YAML mapping")
    return data


def _entry_to_spec(entry: dict[str, Any], path: Path) -> RobotSpec:
    try:
        fields = entry["spec"]
        spec_id = entry["id"]
        if not _ID_RE.fullmatch(str(spec_id)):
            raise ValueError(f"id {spec_id!r} must match {_ID_RE.pattern}")
        return RobotSpec(
            name=spec_id,
            n_joints=fields["n_joints"],
            obs_keys=list(fields["obs_keys"]),
            action_dim=fields["action_dim"],
            target_pos_key=fields["target_pos_key"],
            success_threshold=fields.get("success_threshold", 0.05),
            max_episode_steps=fields.get("max_episode_steps", 200),
            ee_pos_key=fields.get("ee_pos_key", "ee_pos"),
            extra_obs_keys=tuple(fields.get("extra_obs_keys", ())),
            relational_features=tuple(
                (str(pair[0]), str(pair[1])) for pair in fields.get("relational_features", ())
            ),
            lineage_name=entry.get("name"),
            parent_id=entry.get("parent_id"),
        )
    except KeyError as exc:
        raise ValueError(f"Robot spec file {path} is missing required field {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Robot spec file {path} is invalid: {exc}") from exc
