# ml-core — STATUS

## v0.1.0 — 2026-05-20

- [x] `mlcore.policies.base` — `BasePolicy` Protocol
- [x] `mlcore.policies.act_wrapper` — `ACTWrapper` (migré depuis PUBLIC)
- [x] `mlcore.policies.diffusion_wrapper` — `DiffusionWrapper` (migré depuis PUBLIC)
- [x] `mlcore.training.trainer` — `Trainer` avec namespacing `robot_name/policy_type`
- [x] `mlcore.eval.evaluator` — `Evaluator` + `EvalResult` avec namespacing

## Dépendances aval

- `lerobot-playground-portfolio` importe depuis `mlcore` (re-exports dans `playground.*`)
- `my-robot-stack` peut importer depuis `mlcore` sans accès à PUBLIC
