# Contributing — ml-core

This repo is the shared, robot-agnostic algorithm layer. Contributions must preserve that genericity and the import-direction discipline.

## Strict rules

1. **No hardware-specific code.** No robot brand names, no model numbers, no production configs in `src/mlcore/`. The only concrete spec allowed in tree is the generic example `cube_reach_v1`.
2. **No downstream imports.** `mlcore` may not import from any consumer package — that would create cycles and leak non-Apache-2.0 code into Apache-2.0 sources.
3. **No proprietary references.** No mention of `LicenseRef-Proprietary` or `All Rights Reserved`. License is Apache-2.0, period.
4. **SPDX header** on every `.py`:
   ```
   # SPDX-FileCopyrightText: 2026 Arthur Mouraud
   # SPDX-License-Identifier: Apache-2.0
   ```
5. **Coverage ≥60%** on `src/mlcore/` — enforced by `--cov-fail-under=60`. PRs that lower coverage are rejected.
6. **Namespacing.** Any new artefact path must follow `{robot_name}/{policy_type}/`. `robot_name` and `policy_type` are passed as direct parameters (not via Hydra cfg) so the library remains usable from non-Hydra callers.

## Workflow

- Conventional commits: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`
- Feature branches only — no direct commits to `master`.
- PR required (even solo). CI must be green before merge.

## Local checks

```bash
uv run pytest -m unit -v
uv run pytest --cov-fail-under=60
uv run mypy src/
uv run ruff check src/ tests/
```

Pre-commit hooks (`ruff`, `mypy`, anti-leak) run on every commit.

## Code standards

- `typing.Protocol` (`@runtime_checkable`) for public contracts; no ABCs.
- Type hints everywhere; mypy strict mode.
- Google-style docstrings on every public API.
- `from loguru import logger` — never `print()`.
- No direct `os.environ` access — Hydra cfg or explicit env-var checks.
- TDD: tests written before implementation. Mark unit tests with `@pytest.mark.unit`.

## What does NOT belong here

Anything tied to a specific robot, simulator instance, production task, or proprietary stack. Hardware adapters belong in `robotics-platform-template` (HAL Protocols) or in the consumer that owns the hardware. Task-specific pipelines and configs belong in the consumer repo that runs them.
