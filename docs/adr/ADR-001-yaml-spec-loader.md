<!--
SPDX-FileCopyrightText: 2026 Arthur Mouraud
SPDX-License-Identifier: Apache-2.0
-->

# ADR-001 — Data-driven RobotSpec loading from YAML + lineage fields

- **Status**: Implemented 2026-06-10
- **Deciders**: Arthur Mouraud
- **Scope**: `ml-core` (`mlcore.robots`), consumed by `lerobot-playground-portfolio` and `orchestrator`. Cross-reference: `robotics-platform-template/docs/adr/ADR-002-env-adapter-factory-registration.md`, `lerobot-playground-portfolio/docs/adr/ADR-001-robot-specs-yaml-registry.md`.

## Context

Declaring a new robot today (`add_robot.py`) generates 6 Python/YAML files spread across `ml-core` and `lerobot-playground-portfolio` — a `RobotSpec` Python module, a registry import, Hydra env/dataset configs, and an `EnvAdapter` subclass. This is CLI-only and requires write access to source files in two repos.

The orchestrator wants to let users declare/branch robots from the frontend. Its `api` service treats sibling repos as read-only bind-mounts (subprocess-only) — it cannot write Python source. The only viable write target is a data directory.

`RobotSpec` is `frozen=True` by design; "editing" a robot must mean creating a new immutable spec linked to its predecessor, not mutating one in place.

## Decision

1. Add two optional fields to `RobotSpec`: `lineage_name: str | None = None` (UI-facing grouping label) and `parent_id: str | None = None` (link to the spec this one was derived from). Both default to `None` so every existing construction site is unaffected.
2. Close the MLC-009 gap: `__post_init__` now validates `target_pos_key in obs_keys`, raising `ValueError` otherwise. Previously a spec could declare a `target_pos_key` absent from `obs_keys`, which `validate_spec_against_env` would not catch until runtime obs-shape mismatches.
3. Add `mlcore.robots.yaml_loader.load_specs_from_dir(specs_dir: Path) -> list[RobotSpec]`: parses every `*.yaml` in `specs_dir`, builds `RobotSpec(name=entry["id"], lineage_name=entry.get("name"), parent_id=entry.get("parent_id"), **entry["spec"])`, and registers each via the existing `mlcore.robots.registry.register`. Returns `[]` if the directory does not exist (no-op). The path is supplied by the caller — `ml-core` stays agnostic of `playground` or any other consumer's directory layout.

## Alternatives considered

1. **Generate Python source from the orchestrator via subprocess** (extend `add_robot.py`, invoked through the existing CLI-subprocess pattern). Rejected: still requires the orchestrator to trigger writes into `ml-core`/`lerobot-playground-portfolio` source trees, which conflicts with the read-only sibling-repo rule even via subprocess (the subprocess itself runs with the orchestrator's mount permissions).
2. **Keep `RobotSpec` mutable / drop `frozen=True`** to support in-place edits. Rejected: frozen specs are load-bearing for hashability/registry semantics and for the "immutable parent, new child" lineage model the user explicitly requested.
3. **New separate "lineage" data model** distinct from `RobotSpec`. Rejected: the user clarified that "objectifs" (task metadata) and lineage are properties of the spec itself, not a new concept — adding two optional fields is the minimal change.

## Consequences

**Positive**:
- `RobotSpec` construction is now data-driven; new robots can be added by dropping a YAML file in a directory, no Python source changes required.
- `target_pos_key`/`obs_keys` consistency is now enforced at construction time for *all* specs (YAML and Python), closing MLC-009.
- `lineage_name`/`parent_id` give the orchestrator everything it needs to render a parent/child branch tree without a new data model.

**Negative**:
- Two registration paths now exist for `cube_reach_v1` (legacy `mlcore.robots.specs.cube_reach_v1` Python module + `robot_specs/cube_reach_v1.yaml`). The YAML version loads later and wins (last-write-wins in `mlcore.robots.registry.register`, no overwrite warning at this layer — see `lerobot-playground-portfolio/docs/adr/ADR-001-robot-specs-yaml-registry.md` for the `EnvAdapterRegistry`-level warning). Legacy files are kept for now and removed once the orchestrator-driven flow is validated in real usage (P2 cleanup).
- `add_robot.py`'s `_build_spec` had to be patched (`target_pos_key` defaulting logic) to keep producing `obs_keys`-consistent specs under the new MLC-009 validation.

**Migration plan**: P0 (this ADR) ships the loader + field additions only — no behavioural change for existing code paths beyond the MLC-009 validation. P1 (orchestrator) adds the `/api/v1/robots` endpoints that write `robot_specs/*.yaml`. P2 removes the legacy per-robot Python/YAML generation once P1 is validated in real usage.
