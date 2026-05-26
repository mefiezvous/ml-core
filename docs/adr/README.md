<!--
SPDX-FileCopyrightText: 2026 Arthur Mouraud
SPDX-License-Identifier: Apache-2.0
-->

# Architecture Decision Records (ADR) — ml-core

Cross-module architectural decisions that affect this repo. Each ADR is a short, immutable note: once accepted, it is amended only via a successor ADR.

## When to write an ADR
- A decision changes a public Protocol, type, or contract that another repo depends on (e.g. `RobotSpec`, `Trainer`, `Evaluator`).
- A decision removes or replaces an abstraction.
- A trade-off has been made between two viable alternatives, and the reasoning should outlive the conversation.

If the decision is fully local to one module and reversible without breaking consumers, skip the ADR — a code comment is enough.

## Format

`ADR-NNN-short-kebab-title.md`:

```markdown
# ADR-NNN — Title

- **Status**: Proposed | Accepted YYYY-MM-DD | Superseded by ADR-MMM | Implemented YYYY-MM-DD
- **Deciders**: Arthur Mouraud
- **Scope**: <which repos / modules>

## Context
## Decision
## Alternatives considered
## Consequences
```

## Related ADRs in sibling repos

- [robotics-platform-template/docs/adr/ADR-001](../../../robotics-platform-template/docs/adr/ADR-001-unify-on-envadapter.md) — Unify on EnvAdapter, deprecate RobotInterface/Observation/Action. **`RobotSpec` (ml-core) is the source of truth for the observation/action schema** per this decision.

## Index

_(No local ADRs yet.)_
