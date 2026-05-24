# Roadmap — ml-core

Forward-looking only. For volatile state (current branch, in-progress fixes), see workspace memory.

## Current release: v0.1.0

Shared ML library covering policies, training, evaluation, data collection, robot specs.

### Stable surface (v0.1)

- `mlcore.policies.base.BasePolicy` Protocol
- `mlcore.policies.act_wrapper.ACTWrapper`
- `mlcore.policies.diffusion_wrapper.DiffusionWrapper`
- `mlcore.training.trainer.Trainer` (with namespacing `{robot_name}/{policy_type}/`)
- `mlcore.training.checkpoint_manager.CheckpointManager` (local + HF Hub)
- `mlcore.eval.evaluator.Evaluator` + `EvalResult`
- `mlcore.robots.base.RobotSpec` + registry + `GenericScriptedPolicy`
- `mlcore.collection.*` (rollout, scripted, teleop, curation, sink)
- `mlcore.data.*` (filter, sampler, schema)
- `mlcore.observation.builder`

## v0.2 candidates (not committed)

| Item | Rationale |
|---|---|
| Wandb integration (complete) | `_setup_wandb()` currently partial; finish or remove |
| `action_dim` reconciliation | RobotSpec declares 7, pipelines produce 8 (gripper) — pick one source of truth |
| `delta_timestamps` config validation | ACT needs temporal context — surface as Hydra schema check |
| Vectorized eval | Current `Evaluator` is serial; vectorized rollout for >100 episodes |
| LoRA fine-tuning support | PEFT is in deps but not wired |

## Maintenance discipline

- API changes to `Trainer`/`Evaluator`/`BasePolicy` require coordination with all consumers.
- Coverage stays ≥60%; no PR may lower it.
- No `print()`, no direct `os.environ` access — enforced.
- Path namespacing convention is non-negotiable. Any new artefact type must follow `{robot_name}/{policy_type}/`.

## Downstream consumers

| Repo | Imports |
|---|---|
| `lerobot-playground-portfolio` | `mlcore.policies.*`, `mlcore.training.*`, `mlcore.eval.*`, `mlcore.robots.*` |
| `_private/my-robot-stack` | `mlcore.training.*`, `mlcore.eval.*` (without going through `playground`) |
