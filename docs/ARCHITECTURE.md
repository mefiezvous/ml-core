# Architecture — ml-core

Generic, robot-agnostic ML building blocks for policy training, evaluation, and data collection. Apache-2.0, importable from both public and private downstream layers without IP leakage.

## Module map

```
src/mlcore/
├── policies/
│   ├── base.py                BasePolicy Protocol — shared contract
│   ├── act_wrapper.py         ACTWrapper around lerobot.ACTPolicy
│   └── diffusion_wrapper.py   DiffusionWrapper around lerobot.DiffusionPolicy
├── training/
│   ├── trainer.py             Trainer — Hydra cfg + MLflow + checkpoint resume
│   └── checkpoint_manager.py  Local + HF Hub incremental push, pruning
├── eval/
│   └── evaluator.py           Evaluator + EvalResult
├── collection/
│   ├── interfaces.py          Protocols for data collectors
│   ├── rollout.py             Env-driven rollout collection
│   ├── scripted.py            Scripted-policy data generation
│   ├── teleop.py              Teleoperated demo collection
│   ├── curation.py            Episode filtering / quality gates
│   └── sink.py                Persist to LeRobotDataset v3.0
├── data/
│   ├── filter.py              Episode-level filters
│   ├── sampler.py             Hydra-configurable batch samplers
│   └── schema.py              LeRobotDataset v3.0 schema helpers
├── observation/
│   └── builder.py             Enriched observation vector construction
└── robots/
    ├── base.py                RobotSpec frozen dataclass
    ├── registry.py            register() / get() / list_specs()
    ├── config_builder.py      RobotSpec -> Hydra-compatible dict
    ├── generic_policy.py      GenericScriptedPolicy parameterized by RobotSpec
    └── specs/                 Concrete specs (e.g., cube_reach_v1.py)
```

## Key contracts

### `BasePolicy` (Protocol)

All policy wrappers expose the same surface:

```python
policy.reset()
action = policy.select_action(obs_dict)        # np.ndarray
loss   = policy.forward(batch)                  # torch.Tensor (scalar)
policy.save(path)
policy.load_checkpoint(path)
# Optional class method:
policy = ACTWrapper.from_pretrained("repo/model", device="cpu")
```

### `Trainer`

```python
Trainer(
    cfg=hydra_cfg,
    policy=policy,
    dataloader=loader,
    robot_name="cube_reach_v1",
    policy_type="act",
    hf_repo_id="mefiezvous/cube-reach-v1-act",  # optional, enables HF Hub uploads
    push_every=500,
    keep_last_n=3,
)
```

Behaviors:
- MLflow logging every `save_every_steps`
- Local checkpoint per `save_every_steps`, namespaced by `{robot_name}/{policy_type}/`
- Auto-resume from latest checkpoint in `checkpoint_dir`
- Optional HF Hub push (incremental, prunes old)

### `Evaluator`

```python
result: EvalResult = Evaluator(
    policy=policy,
    env_name="CubeReachV1",
    n_episodes=50,
    robot_name="cube_reach_v1",
    policy_type="act",
    mlflow_run_id="abc123",
).evaluate()
# result.success_rate, result.mean_reward, result.per_episode_rewards
```

### `RobotSpec` + registry

Declarative robot descriptor:

```python
from mlcore.robots.base import RobotSpec
from mlcore.robots.registry import register

@register
CUBE_REACH_V1 = RobotSpec(
    name="cube_reach_v1",
    n_joints=7,
    action_dim=8,                    # 7 joints + 1 gripper
    obs_keys=("ee_pos", "cube_pos", "joint_positions", "ee_to_cube"),
    success_threshold=0.05,
    ...
)
```

The `GenericScriptedPolicy` consumes a `RobotSpec` to produce a P-controller without per-robot code. `config_builder.build()` produces a Hydra-compatible dict from a spec.

## Namespacing convention (non-negotiable)

`robot_name` and `policy_type` are mandatory parameters on `Trainer` and `Evaluator`. All artefact paths are derived automatically:

| Artefact | Path |
|---|---|
| Checkpoint | `checkpoints/{robot_name}/{policy_type}/checkpoint_XXXXXXXX.ckpt` |
| MLflow run | `mlruns/{robot_name}_{policy_type}/` |
| Eval report | `eval_reports/{robot_name}/{policy_type}/eval_report.json` |

Rationale: prevents data mixing between robots, enables side-by-side comparisons, isolates failures.

## Import rules

- `ml-core` may import: `lerobot`, `mlflow`, `hydra`, `numpy`, `torch`, `loguru`, `huggingface_hub`, `robotics_platform.hal.*` (types).
- `ml-core` may **NOT** import: `playground.*`, `my_robot.*`, or anything in downstream consumers.
- Downstream consumers (`lerobot-playground-portfolio`, `_private/my-robot-stack`) import `mlcore.*` — never the inverse.

## Consumers

| Repo | Usage |
|---|---|
| `lerobot-playground-portfolio` | Re-exports as `playground.policies.*`, `playground.training.*`, etc. |
| `_private/my-robot-stack` | Imports directly without going through `playground` — no IP leak path |

## Test discipline

- All tests marked `@pytest.mark.unit` by default.
- Coverage gate: `--cov-fail-under=60`.
- TDD: tests written before implementation.
- mypy strict.
