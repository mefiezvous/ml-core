# Contributing — ml-core

This library is the shared algorithm layer, importable by both public and private consumers. Contributions must preserve that boundary.

## Import rules (non-negotiable)

`ml-core` may import:
- `lerobot`, `mlflow`, `hydra`, `omegaconf`
- `numpy`, `torch`, `huggingface_hub`, `loguru`
- `robotics_platform.hal.*` (types only)

`ml-core` may **NOT** import:
- `playground.*` (would create a cycle with `lerobot-playground-portfolio`)
- `my_robot.*` (would leak proprietary code into Apache-2.0)
- Anything that names a specific robot or production task

## Code standards

1. SPDX header on every `.py`:
   ```
   # SPDX-FileCopyrightText: 2026 Arthur Mouraud
   # SPDX-License-Identifier: Apache-2.0
   ```
2. `from loguru import logger` — never `print()`.
3. No direct `os.environ` access — Hydra cfg or explicit env-var checks.
4. Type hints everywhere; `mypy --strict` clean.
5. Google-style docstrings on public API.
6. TDD: tests written before implementation. Mark all unit tests with `@pytest.mark.unit`.

## Namespacing convention

Any new artefact path must follow `{robot_name}/{policy_type}/` segmentation. `robot_name` and `policy_type` are passed as direct parameters (not via Hydra cfg) so the library remains usable from non-Hydra callers.

## Workflow

- Conventional commits: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`
- Feature branches only — no direct commits to `main`.
- PR required (even solo), CI must be green.

## Local checks

```bash
uv run pytest -m unit -v
uv run pytest --cov-fail-under=60
uv run mypy src/
uv run ruff check src/ tests/
```
