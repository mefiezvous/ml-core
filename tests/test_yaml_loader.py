# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``mlcore.robots.yaml_loader.load_specs_from_dir``."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlcore.robots import get
from mlcore.robots.yaml_loader import load_specs_from_dir

_MINIMAL_SPEC = """
id: yaml_loader_minimal
spec:
  n_joints: 6
  obs_keys: [ee_pos, target_pos]
  action_dim: 6
  target_pos_key: target_pos
"""

_FULL_SPEC = """
id: yaml_loader_full
name: yaml_loader_lineage
parent_id: yaml_loader_minimal
spec:
  n_joints: 7
  obs_keys: [ee_pos, cube_pos, joints]
  action_dim: 8
  target_pos_key: cube_pos
  success_threshold: 0.1
  max_episode_steps: 100
  ee_pos_key: ee_pos
  extra_obs_keys: [gripper]
  relational_features: [[cube_pos, ee_pos]]
"""


@pytest.mark.unit
def test_load_specs_from_dir_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    assert load_specs_from_dir(tmp_path / "does_not_exist") == []


@pytest.mark.unit
def test_load_specs_from_dir_parses_minimal_entry(tmp_path: Path) -> None:
    (tmp_path / "minimal.yaml").write_text(_MINIMAL_SPEC, encoding="utf-8")

    specs = load_specs_from_dir(tmp_path)

    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "yaml_loader_minimal"
    assert spec.n_joints == 6
    assert spec.obs_keys == ["ee_pos", "target_pos"]
    assert spec.action_dim == 6
    assert spec.target_pos_key == "target_pos"
    # Defaults applied for fields absent from the YAML.
    assert spec.success_threshold == 0.05
    assert spec.max_episode_steps == 200
    assert spec.ee_pos_key == "ee_pos"
    assert spec.extra_obs_keys == ()
    assert spec.relational_features == ()
    assert spec.lineage_name is None
    assert spec.parent_id is None


@pytest.mark.unit
def test_load_specs_from_dir_parses_full_entry_and_lineage(tmp_path: Path) -> None:
    (tmp_path / "full.yaml").write_text(_FULL_SPEC, encoding="utf-8")

    specs = load_specs_from_dir(tmp_path)

    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "yaml_loader_full"
    assert spec.lineage_name == "yaml_loader_lineage"
    assert spec.parent_id == "yaml_loader_minimal"
    assert spec.success_threshold == 0.1
    assert spec.max_episode_steps == 100
    assert spec.extra_obs_keys == ("gripper",)
    assert spec.relational_features == (("cube_pos", "ee_pos"),)


@pytest.mark.unit
def test_load_specs_from_dir_registers_specs(tmp_path: Path) -> None:
    (tmp_path / "minimal.yaml").write_text(_MINIMAL_SPEC, encoding="utf-8")

    load_specs_from_dir(tmp_path)

    assert get("yaml_loader_minimal").action_dim == 6


@pytest.mark.unit
def test_load_specs_from_dir_sorted_by_filename(tmp_path: Path) -> None:
    (tmp_path / "b_minimal.yaml").write_text(_MINIMAL_SPEC, encoding="utf-8")
    (tmp_path / "a_full.yaml").write_text(_FULL_SPEC, encoding="utf-8")

    specs = load_specs_from_dir(tmp_path)

    assert [spec.name for spec in specs] == ["yaml_loader_full", "yaml_loader_minimal"]


@pytest.mark.unit
def test_load_specs_from_dir_raises_on_non_mapping_yaml(tmp_path: Path) -> None:
    (tmp_path / "list.yaml").write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        load_specs_from_dir(tmp_path)


@pytest.mark.unit
def test_load_specs_from_dir_raises_on_missing_required_field(tmp_path: Path) -> None:
    (tmp_path / "incomplete.yaml").write_text(
        "id: incomplete\nspec:\n  n_joints: 6\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="missing required field"):
        load_specs_from_dir(tmp_path)


@pytest.mark.unit
def test_load_specs_from_dir_raises_on_invalid_target_pos_key(tmp_path: Path) -> None:
    invalid_spec = """
id: yaml_loader_invalid
spec:
  n_joints: 6
  obs_keys: [ee_pos]
  action_dim: 6
  target_pos_key: not_in_obs_keys
"""
    (tmp_path / "invalid.yaml").write_text(invalid_spec, encoding="utf-8")

    with pytest.raises(ValueError, match="must be in obs_keys"):
        load_specs_from_dir(tmp_path)
