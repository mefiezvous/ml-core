# ml-core

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

> Shared ML library for robotic policy training, evaluation, and data collection.
> Robot-agnostic building blocks importable from any downstream layer.

## What this is

A reusable algorithm layer providing:
- `BasePolicy` — Protocol shared by all policy wrappers (`select_action`, `forward`, `save`, `load_checkpoint`)
- `ACTWrapper`, `DiffusionWrapper` — thin wrappers around LeRobot policies
- `Trainer` — Hydra-configurable training loop with MLflow + checkpoint resume
- `Evaluator` + `EvalResult` — rollout-based metrics with namespaced reports
- `RobotSpec` + registry + `GenericScriptedPolicy` — declarative robot descriptors
- `ObservationBuilder` — task-agnostic obs dict → flat state vector
- `collection.*` — scripted, rollout, teleop collectors + `HubSink` for LeRobotDataset v3.0
- `data.*` — filters, samplers, feature-schema builders

Depends on `robotics-platform-template` for HAL types. Zero hardware-specific code, zero proprietary references.

## Install

As a local path dependency in your `pyproject.toml`:

```toml
[tool.uv.sources]
ml-core = { path = "../ml-core", editable = true }
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
# Checkpoints → checkpoints/cube_reach_v1/act/checkpoint_XXXXXXXX.ckpt
# MLflow run  → mlruns/cube_reach_v1_act/
```

`robot_name` and `policy_type` are passed directly (not via Hydra) so the library is usable from non-Hydra callers too.

Artefact paths are always namespaced as `{robot_name}/{policy_type}/`:

| Artefact | Path |
|---|---|
| Checkpoints | `checkpoints/{robot_name}/{policy_type}/checkpoint_XXXXXXXX.ckpt` |
| MLflow runs | `mlruns/{robot_name}_{policy_type}/` |
| Eval reports | `eval_reports/{robot_name}/{policy_type}/eval_report.json` |

## Add a robot

RobotSpecs live under [`src/mlcore/robots/specs/`](src/mlcore/robots/specs/) — one
dataclass per robot, picked up by `mlcore.robots.get(name)` and validated against
the env at pipeline startup via `mlcore.robots.validate_spec_against_env`.

The canonical end-to-end tutorial (scaffold → collect → train → eval) lives in
the template repo: [robotics-platform-template/docs/ADD_A_ROBOT.md](../robotics-platform-template/docs/ADD_A_ROBOT.md).

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module map, contracts, consumers
- [docs/ROADMAP.md](docs/ROADMAP.md) — forward-looking
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — workflow & strict rules

## License

Apache-2.0. See [LICENSE](LICENSE).
Copyright 2026 Arthur Mouraud.
