# CLAUDE.md — ml-core

## Identity
Shared, robot-agnostic ML library: policies, training, evaluation, data collection, robot specs.
Apache-2.0, Python 3.12+. Author: Arthur Mouraud. Package name: `mlcore`.

## Critical Rules
1. NEVER add hardware-specific code, brand names, or production configs.
2. NEVER import from downstream consumers (no `playground.*`, no proprietary modules).
3. NEVER use `LicenseRef-Proprietary` or `All Rights Reserved`.
4. SPDX header required at top of every `.py`:
   `# SPDX-FileCopyrightText: 2026 Arthur Mouraud`
   `# SPDX-License-Identifier: Apache-2.0`
5. Path namespacing `{robot_name}/{policy_type}/` is non-negotiable for checkpoints, MLflow runs, eval reports.
6. `from loguru import logger` — no `print()`. No direct `os.environ` access.
7. Coverage gate: `--cov-fail-under=60` on `src/mlcore/`.

## Code Standards
- `typing.Protocol` (`@runtime_checkable`) for public contracts; no ABCs.
- Type hints everywhere, mypy strict.
- Google-style docstrings on public API.
- TDD: tests written before implementation. Mark unit tests with `@pytest.mark.unit`.

## Documentation enfant
- [README.md](README.md) — purpose, install, quickstart, namespacing
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module map, contracts, consumers
- [docs/ROADMAP.md](docs/ROADMAP.md) — forward-looking
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — workflow, strict rules

## Workspace context (non committé)
- Cross-repo rules & memory : `../CLAUDE.md` racine workspace
- État volatile (branche active, P0) : `memory/project_state.md`
