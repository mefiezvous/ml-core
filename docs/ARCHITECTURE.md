# Architecture — ml-core

Generic, robot-agnostic ML building blocks for policy training, evaluation, and data collection. Apache-2.0, importable by any downstream layer without IP cross-contamination.

## Design principle: decouple robot × policy via Protocol + path convention

Two orthogonal axes — the **robot** (`RobotSpec`) and the **policy** (`BasePolicy` Protocol) — are kept independent. The glue is a non-negotiable path convention that namespaces every artefact by `{robot_name}/{policy_type}/`.

```python
from mlcore.policies.base import BasePolicy

class MyWrapper:                                # no inheritance
    def select_action(self, obs) -> np.ndarray: ...
    def reset(self) -> None: ...
    def forward(self, batch) -> torch.Tensor: ...
    def save(self, path: Path) -> None: ...
    def load_checkpoint(self, path: Path) -> None: ...
    def parameters(self) -> Any: ...

assert isinstance(MyWrapper(), BasePolicy)      # True via Protocol
```

## Layout

```
src/mlcore/
├── policies/                          # Policy wrappers
│   ├── base.py                BasePolicy Protocol (6 methods)
│   ├── act_wrapper.py         ACTWrapper around lerobot.ACTPolicy
│   └── diffusion_wrapper.py   DiffusionWrapper around lerobot.DiffusionPolicy
├── training/                          # Training loop
│   ├── trainer.py             Trainer — Hydra cfg + MLflow + auto-resume
│   └── checkpoint_manager.py  Local + HF Hub incremental push, pruning
├── eval/                              # Policy evaluation
│   └── evaluator.py           Evaluator + EvalResult dataclass
├── collection/                        # Data collection
│   ├── interfaces.py          Collector + Episode + EnvLike Protocols
│   ├── rollout.py             PolicyRolloutCollector
│   ├── scripted.py            ScriptedCollector + ScriptedReachPolicy
│   ├── teleop.py              TeleopCollector (stub)
│   ├── curation.py            filter_by_success, dedupe, balance_by_task
│   └── sink.py                HubSink — persist to LeRobotDataset v3.0
├── data/                              # Dataset utilities
│   ├── filter.py              Episode-level filters
│   ├── sampler.py             Hydra-configurable batch samplers
│   └── schema.py              LeRobotDataset v3.0 feature schema builders
├── observation/                       # Obs vector construction
│   └── builder.py             ObservationBuilder (frozen dataclass)
└── robots/                            # Robot descriptors
    ├── base.py                RobotSpec (frozen dataclass)
    ├── registry.py            register() / get() / list_specs()
    ├── config_builder.py      RobotSpec → Hydra-compatible dict
    ├── generic_policy.py      GenericScriptedPolicy parameterized by spec
    └── yaml_loader.py         load_specs_from_dir() — data-driven specs from YAML
```

## `policies/` — `BasePolicy` Protocol

| Method | Returns | Used for |
|---|---|---|
| `select_action(obs)` | `np.ndarray` | Inference step |
| `reset()` | `None` | Episode start |
| `forward(batch)` | `torch.Tensor` (scalar loss) | Training step |
| `save(path)` | `None` | Persist weights |
| `load_checkpoint(path)` | `None` | Resume from disk |
| `parameters()` | iterable | Optimizer construction |

`ACTWrapper.from_pretrained(repo, device=...)` is the canonical entry point for HF Hub-hosted ACT checkpoints.

## `training/` — `Trainer`

```python
Trainer(
    cfg=hydra_cfg,
    policy=policy,
    dataloader=loader,
    robot_name="cube_reach_v1",
    policy_type="act",
    hf_repo_id="user/cube-reach-v1-act",     # optional, enables HF Hub push
)
```

Behaviours:
- MLflow run started in `_setup_mlflow()`; metrics logged every `save_every_steps`.
- Local checkpoint per `save_every_steps`, namespaced by `{robot_name}/{policy_type}/`.
- Auto-resume from the latest checkpoint in `checkpoint_dir` on construction.
- Optional HF Hub push (incremental, prunes old) via `CheckpointManager`.

## `eval/` — `Evaluator` + `EvalResult`

```python
result: EvalResult = Evaluator(
    policy=policy,
    env_name="CubeReachV1",
    n_episodes=50,
    robot_name="cube_reach_v1",
    policy_type="act",
    mlflow_run_id="abc123",              # optional, logs eval/* metrics
).evaluate()
# result.success_rate, result.mean_reward, result.per_episode_rewards
```

| `EvalResult` field | Type | Meaning |
|---|---|---|
| `n_episodes` | `int` | Episodes executed |
| `success_rate` | `float` | Fraction in `[0.0, 1.0]` |
| `mean_reward` | `float` | Average cumulative reward |
| `per_episode_rewards` | `list[float]` | Per-episode totals |

The JSON report is written to `eval_reports/{robot_name}/{policy_type}/eval_report.json`.

## `collection/` — data collectors + sink

| Symbol | Role |
|---|---|
| `Episode` | Dataclass holding a single trajectory (obs, action, reward, success, task labels) |
| `Collector` | Protocol: `collect(n_episodes) -> list[Episode]` |
| `EnvLike` | Protocol: minimal env surface used by collectors |
| `ScriptedCollector` | Records rollouts of a deterministic scripted policy |
| `PolicyRolloutCollector` | Records rollouts of any `BasePolicy` |
| `TeleopCollector` | Stub for teleoperated demos |
| `HubSink` | Persists episodes to a LeRobotDataset v3.0 + optional HF Hub push |

Curation helpers (`filter_by_success`, `dedupe_by_trajectory_hash`, `balance_by_task`) are pure functions over `list[Episode]`.

## `data/` — dataset utilities

| Module | Provides |
|---|---|
| `filter.py` | Episode-level filters |
| `sampler.py` | Hydra-configurable batch samplers |
| `schema.py` | `build_features()` for LeRobotDataset v3.0 + canonical key constants |

`schema.build_features(state_dim, state_names, action_dim=8, image_keys=...)` is the single source of truth for the feature dict — both PUBLIC and any downstream consumer describe their datasets identically.

## `observation/` — `ObservationBuilder`

Frozen dataclass that concatenates raw obs values (in `keys` order) and relational deltas (`obs[b] - obs[a]`) into a flat float32 vector. `state_dim` and `state_names` are pure properties known at construction time — required by downstream feature schemas and policy input layers.

## `robots/` — `RobotSpec` + registry

Declarative robot descriptor:

```python
from mlcore.robots.base import RobotSpec
from mlcore.robots.registry import register

CUBE_REACH_V1 = RobotSpec(
    name="cube_reach_v1",
    n_joints=7,
    obs_keys=["ee_pos", "cube_pos", "joints"],
    action_dim=8,                       # 7 joints + 1 gripper
    ee_pos_key="ee_pos",
    target_pos_key="cube_pos",
    relational_features=(("cube_pos", "ee_pos"),),
)
register(CUBE_REACH_V1)
```

| Field | Type | Meaning |
|---|---|---|
| `name` | `str` | Snake-case identifier; namespace key |
| `n_joints` | `int` | DOF of the robot |
| `obs_keys` | `list[str]` | Observation keys consumed by policies |
| `action_dim` | `int` | Action vector dimension |
| `target_pos_key` | `str` | Mandatory; obs key for target XYZ |
| `ee_pos_key` | `str` | Obs key for end-effector XYZ |
| `success_threshold` | `float` | Distance (m) below which an episode succeeds |
| `max_episode_steps` | `int` | Hard step cap |
| `extra_obs_keys` | `tuple[str, ...]` | Task-specific extras (gripper, force, …) |
| `relational_features` | `tuple[tuple[str, str], ...]` | Pairs producing `obs[b] - obs[a]` deltas |

`GenericScriptedPolicy` consumes a `RobotSpec` and produces a P-controller without per-robot code. `config_builder.build()` converts a spec into a Hydra-compatible dict.

## Namespacing convention (non-negotiable)

`robot_name` and `policy_type` are mandatory parameters on `Trainer` and `Evaluator`. All artefact paths derive from them:

| Artefact | Path |
|---|---|
| Checkpoint | `checkpoints/{robot_name}/{policy_type}/checkpoint_XXXXXXXX.ckpt` |
| MLflow run | `mlruns/{robot_name}_{policy_type}/` |
| Eval report | `eval_reports/{robot_name}/{policy_type}/eval_report.json` |

Rationale: prevents cross-robot data contamination, enables side-by-side comparison, isolates failures.

## Import rules

- `mlcore` may import: `lerobot`, `mlflow`, `hydra`, `omegaconf`, `numpy`, `torch`, `loguru`, `huggingface_hub`, `robotics_platform.hal.*` (types).
- `mlcore` may **NOT** import any downstream consumer or anything naming a specific production task.
- Dependency direction: downstream consumers import `mlcore.*`; the inverse is forbidden.

## Coverage policy

`--cov-fail-under=60` enforced on `src/mlcore/`. All tests marked `@pytest.mark.unit` by default.

## Consumers

| Repo | Imports |
|---|---|
| `lerobot-playground-portfolio` | `mlcore.policies.*`, `mlcore.training.*`, `mlcore.eval.*`, `mlcore.collection.*`, `mlcore.robots.*` |
