# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Policy evaluation loop with MLflow logging and namespaced report output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
from loguru import logger


@dataclass
class EvalResult:
    """Aggregated results of a policy evaluation run.

    Attributes:
        n_episodes: Total number of episodes executed.
        success_rate: Fraction of episodes deemed successful [0.0, 1.0].
        mean_reward: Average cumulative reward per episode.
        per_episode_rewards: Cumulative reward for each individual episode.
    """

    n_episodes: int
    success_rate: float
    mean_reward: float
    per_episode_rewards: list[float] = field(default_factory=list)


class Evaluator:
    """Runs a policy in an environment for n_episodes and aggregates results.

    The evaluation report is written to:
    ``eval_reports/{robot_name}/{policy_type}/eval_report.json``

    Args:
        policy: Policy with a ``select_action(obs) → action`` interface.
        env_name: Environment identifier (used for logging).
        n_episodes: Number of evaluation episodes. Must be positive.
        robot_name: Robot identifier used to namespace the report path.
        policy_type: Policy identifier (e.g. ``"act"``, ``"diffusion"``).
        mlflow_run_id: If set, log ``eval/*`` metrics to this MLflow run.
        env: Pre-built environment instance. When ``None``, built from ``env_name``
            via mujoco_playground — pass an env directly for testing.
    """

    def __init__(
        self,
        policy: Any,
        env_name: str,
        n_episodes: int,
        *,
        robot_name: str,
        policy_type: str,
        mlflow_run_id: str | None = None,
        env: Any | None = None,
    ) -> None:
        if n_episodes <= 0:
            raise ValueError(f"n_episodes must be positive, got {n_episodes}")
        self._policy = policy
        self._env_name = env_name
        self._n_episodes = n_episodes
        self._mlflow_run_id = mlflow_run_id
        self._report_path = Path(f"eval_reports/{robot_name}/{policy_type}/eval_report.json")
        self._env: Any = env if env is not None else self._build_env()

    def _build_env(self) -> Any:
        try:
            import mujoco_playground as mp  # type: ignore[import-not-found]  # noqa: PLC0415

            return mp.make(self._env_name)
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "mujoco_playground is required to build the env automatically. "
                "Pass env=<env_instance> to the Evaluator constructor for testing."
            ) from exc

    def evaluate(self) -> EvalResult:
        """Run n_episodes rollouts and return aggregated EvalResult.

        Writes a JSON report to the namespaced ``eval_reports/`` path.
        """
        per_episode_rewards: list[float] = []
        n_successes = 0

        logger.info(f"Evaluating | env={self._env_name} n_episodes={self._n_episodes}")

        for ep in range(self._n_episodes):
            obs: Any = self._env.reset()
            done = False
            ep_reward = 0.0
            ep_success = False

            while not done:
                action: Any = self._policy.select_action(obs)
                obs, reward, done, info = self._env.step(action)
                ep_reward += float(reward)
                ep_success = bool(info.get("success", False))

            per_episode_rewards.append(ep_reward)
            if ep_success:
                n_successes += 1
            logger.debug(
                f"Episode {ep + 1}/{self._n_episodes} "
                f"reward={ep_reward:.2f} success={ep_success}"
            )

        result = EvalResult(
            n_episodes=self._n_episodes,
            success_rate=n_successes / self._n_episodes,
            mean_reward=sum(per_episode_rewards) / self._n_episodes,
            per_episode_rewards=per_episode_rewards,
        )
        logger.info(
            f"Eval done | success_rate={result.success_rate:.2%} "
            f"mean_reward={result.mean_reward:.3f}"
        )

        self._write_report(result)

        if self._mlflow_run_id is not None:
            with mlflow.start_run(run_id=self._mlflow_run_id):
                mlflow.log_metrics(
                    {
                        "eval/success_rate": result.success_rate,
                        "eval/mean_reward": result.mean_reward,
                    }
                )

        return result

    def _write_report(self, result: EvalResult) -> None:
        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "n_episodes": result.n_episodes,
            "success_rate": result.success_rate,
            "mean_reward": result.mean_reward,
            "per_episode_rewards": result.per_episode_rewards,
        }
        self._report_path.write_text(json.dumps(report, indent=2))
        logger.debug(f"Eval report written → {self._report_path}")
