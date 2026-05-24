# CLAUDE.md — ml-core

## Identity
Shared ML library (Apache-2.0): policies, training, evaluation, data collection, robot specs.
Importable from both public (`lerobot-playground-portfolio`) and private (`_private/my-robot-stack`) layers without IP cross-contamination. Python 3.12+.

## Critical Rules
1. NEVER import from `playground.*`, `my_robot.*`, or any downstream consumer (cycle / IP leak).
2. NEVER reference specific robots, brand names, production configs.
3. SPDX header in every `.py`:
   `# SPDX-FileCopyrightText: 2026 Arthur Mouraud`
   `# SPDX-License-Identifier: Apache-2.0`
4. `from loguru import logger` — no `print()`.
5. No direct `os.environ` access.
6. Path namespacing `{robot_name}/{policy_type}/` is non-negotiable for checkpoints, MLflow, eval reports.

## Code Standards
- mypy strict, type hints everywhere.
- TDD: tests before code. Mark `@pytest.mark.unit`.
- Coverage gate: 60% (enforced).
- Google-style docstrings on public API.

## Documentation enfant
- [README.md](README.md) — purpose, install, quickstart, namespacing
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module map, contracts, consumers
- [docs/ROADMAP.md](docs/ROADMAP.md) — forward-looking
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — workflow, import rules

## Workspace context (non committé)
- Cross-repo rules & memory : `../CLAUDE.md` racine workspace
- État volatile (branche active, P0) : `memory/project_state.md`
