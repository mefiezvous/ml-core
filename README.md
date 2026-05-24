# ml-core

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

> Shared ML library for robotic policy training, evaluation, and data collection.
> Apache-2.0. Used by both public and private downstream layers without IP cross-contamination.

## What this is

`ml-core` is the algorithm layer of the workspace. It provides generic, reusable building blocks:

- **Policies**: thin wrappers around LeRobot's `ACTPolicy` and `DiffusionPolicy` (`mlcore.policies`)
- **Training**: Hydra-configurable `Trainer` with MLflow + checkpoint management (`mlcore.training`)
- **Evaluation**: `Evaluator` + `EvalResult` with success/reward metrics (`mlcore.eval`)
- **Data collection**: scripted, teleop, rollout, curation, sink (`mlcore.collection`)
- **Robot specs**: `RobotSpec` declarative descriptor + registry + generic scripted policy (`mlcore.robots`)
- **Observation**: builders for enriched obs vectors (`mlcore.observation`)
- **Data**: filters, samplers, schemas (`mlcore.data`)

It depends on `robotics-platform-template` for the HAL types. It does **not** know about specific robots or training tasks — those live in downstream consumers.

## Install

As a local path dependency:

```toml
[tool.uv.sources]
ml-core = { path = "../ml-core", editable = true }
```

Then in your code:

```python
from mlcore.policies.act_wrapper import ACTWrapper
from mlcore.training.trainer import Trainer
from mlcore.eval.evaluator import Evaluator, EvalResult
```

## Quickstart

```python
from mlcore.training.trainer import Trainer

trainer = Trainer(
    cfg=cfg,
    policy=policy,
    dataloader=dataloader,
    robot_name="cube_reach_v1",
    policy_type="act",
)
trainer.train()
# Checkpoints land in: checkpoints/cube_reach_v1/act/checkpoint_XXXXXXXX.ckpt
# MLflow runs in:      mlruns/cube_reach_v1_act/
```

`robot_name` and `policy_type` are passed directly (not via Hydra) so the lib is usable from non-Hydra callers too.

## Path namespacing convention

| Artefact | Path |
|---|---|
| Checkpoints | `checkpoints/{robot_name}/{policy_type}/checkpoint_XXXXXXXX.ckpt` |
| MLflow runs | `mlruns/{robot_name}_{policy_type}/` |
| Eval reports | `eval_reports/{robot_name}/{policy_type}/eval_report.json` |

This isolates artefacts between robots and between policy types — non-negotiable.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module map, interfaces, namespacing
- [docs/ROADMAP.md](docs/ROADMAP.md) — forward-looking
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — workflow, code standards, import rules

## License

Apache-2.0. Copyright 2026 Arthur Mouraud.
