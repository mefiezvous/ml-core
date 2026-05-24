# Roadmap — ml-core

Forward-looking only. For volatile state (current branch, in-progress fixes), see workspace memory.

## Current release: v0.1.0

Shared ML library covering policies, training, evaluation, data collection, observation building, and robot specs. CI green, coverage ≥60%.

## Stable surface (v0.1)

- `mlcore.policies.base.BasePolicy` — Protocol (6 methods)
- `mlcore.policies.act_wrapper.ACTWrapper`
- `mlcore.policies.diffusion_wrapper.DiffusionWrapper`
- `mlcore.training.trainer.Trainer` — Hydra cfg + MLflow + namespaced checkpoints
- `mlcore.training.checkpoint_manager.CheckpointManager` — local + HF Hub
- `mlcore.eval.evaluator.Evaluator` + `EvalResult`
- `mlcore.robots.base.RobotSpec` + registry + `GenericScriptedPolicy`
- `mlcore.collection.*` — `Collector`, `Episode`, `ScriptedCollector`, `PolicyRolloutCollector`, `TeleopCollector`, `HubSink`, curation helpers
- `mlcore.data.*` — `build_features`, filters, samplers
- `mlcore.observation.builder.ObservationBuilder`

## v0.2 candidates (not committed)

| Item | Rationale |
|---|---|
| Wandb integration (complete) | `Trainer._setup_wandb()` is partial — finish or remove |
| `action_dim` reconciliation | `RobotSpec` may declare 7 while pipelines produce 8 (gripper) — pick one source of truth |
| `delta_timestamps` Hydra schema | ACT needs temporal context; catch misconfig at instantiation |
| Vectorized eval | `Evaluator` is serial — vectorized rollout for >100 episodes |
| LoRA fine-tuning support | PEFT is available; not wired into `Trainer` |
| Reference docs (Sphinx / mkdocs) | Currently README + this docs/ folder only |

These are candidates — none planned until a consumer requires them. The library stays small on purpose.

## Maintenance discipline

- No breaking changes to public Protocols (`BasePolicy`) or to `Trainer` / `Evaluator` signatures without a major version bump.
- Coverage stays ≥60% on `src/mlcore/`. No PR may lower it.
- No hardware-specific code, no brand names, no proprietary references — checked by anti-leak pre-commit and CI.
- Path namespacing convention is non-negotiable. Any new artefact type must follow `{robot_name}/{policy_type}/`.
- No `print()`, no direct `os.environ` access — enforced.

## Downstream consumers (informational)

| Repo | Current usage |
|---|---|
| `lerobot-playground-portfolio` | Imports `mlcore.policies.*`, `mlcore.training.*`, `mlcore.eval.*`, `mlcore.collection.*`, `mlcore.robots.*` |

Changes to public APIs require coordination with these consumers.
