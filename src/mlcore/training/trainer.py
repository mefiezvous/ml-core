# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Hydra-configurable training loop with MLflow logging and checkpoint resume."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Union

import mlflow
import torch
from loguru import logger
from omegaconf import DictConfig

from mlcore.policies.act_wrapper import ACTWrapper
from mlcore.policies.diffusion_wrapper import DiffusionWrapper
from mlcore.training.checkpoint_manager import CheckpointManager

PolicyWrapper = Union[ACTWrapper, DiffusionWrapper]


class Trainer:
    """Imitation-learning training loop over a LeRobotDataset.

    Checkpoint, MLflow, and WandB paths are namespaced automatically:
    - checkpoints/{robot_name}/{policy_type}/checkpoint_XXXXXXXX.ckpt
    - mlruns/{robot_name}_{policy_type}/

    Args:
        cfg: Full Hydra config (``training.*``, ``logging.*`` keys used).
        policy: Wrapped lerobot policy implementing the PolicyWrapper interface.
        dataloader: Iterable yielding batched tensor dicts. Cycled when exhausted.
        robot_name: Robot identifier used to namespace all output paths.
        policy_type: Policy identifier (e.g. ``"act"``, ``"diffusion"``).
    """

    def __init__(
        self,
        cfg: DictConfig,
        policy: PolicyWrapper,
        dataloader: Iterable[dict[str, torch.Tensor]],
        *,
        robot_name: str,
        policy_type: str,
        hf_repo_id: str | None = None,
    ) -> None:
        self._cfg = cfg
        self._policy = policy
        self._dataloader = dataloader
        self._step = 0
        self._robot_name = robot_name
        self._policy_type = policy_type

        self._checkpoint_dir = Path(f"checkpoints/{robot_name}/{policy_type}")
        self._mlflow_tracking_uri = f"mlruns/{robot_name}_{policy_type}"

        self.ckpt_manager = CheckpointManager(
            local_dir=self._checkpoint_dir,
            hf_repo_id=hf_repo_id,
            push_every=cfg.get("push_every", 500),
            keep_last_n=cfg.get("keep_last_n", 3),
        )

        self._setup_mlflow()
        self._resume_from_checkpoint()
        self._setup_optimizer()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_mlflow(self) -> None:
        mlflow.set_tracking_uri(self._mlflow_tracking_uri)
        mlflow.set_experiment(self._cfg.logging.experiment_name)
        mlflow.start_run(run_name=self._cfg.logging.run_name)
        dataset_cfg = self._cfg.get("dataset", {})
        mlflow.set_tags(
            {
                "dataset.repo_id": dataset_cfg.get("repo_id", "unknown"),
                "dataset.root": str(dataset_cfg.get("root", "hub")),
                "policy.type": self._policy_type,
                "robot.name": self._robot_name,
            }
        )
        logger.info(
            f"MLflow run started | experiment={self._cfg.logging.experiment_name} "
            f"run={self._cfg.logging.run_name}"
        )

    def _setup_optimizer(self) -> None:
        self._optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self._policy.parameters(),
            lr=self._cfg.training.lr,
        )

    def _setup_wandb(self) -> None:
        if not os.environ.get("WANDB_API_KEY"):
            self._wandb_enabled = False
            return
        try:
            import wandb  # noqa: PLC0415

            wandb.init(
                project=self._cfg.logging.experiment_name,
                name=self._cfg.logging.run_name,
            )
            self._wandb: Any = wandb
            self._wandb_enabled = True
            logger.info("WandB logging enabled")
        except ImportError:
            logger.warning("WANDB_API_KEY is set but wandb is not installed — skipping")
            self._wandb_enabled = False

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _detect_latest_checkpoint(self) -> Path | None:
        """Return the path of the most recent checkpoint, or ``None``."""
        checkpoints = self.ckpt_manager.list_checkpoints()
        return checkpoints[-1][1] if checkpoints else None

    def _resume_from_checkpoint(self) -> None:
        result = self.ckpt_manager.load_latest()
        if result is None:
            return
        state, step = result
        self._policy.load_state_dict(state["model"])
        self._step = step
        logger.info(f"Resumed from checkpoint at step {step}")

    def _save_checkpoint(self, step: int) -> None:
        self.ckpt_manager.save({"model": self._policy.state_dict(), "step": step}, step)
        logger.info(f"Checkpoint saved at step {step}")

    # ------------------------------------------------------------------
    # Metric logging
    # ------------------------------------------------------------------

    def _log_metrics(
        self,
        step: int,
        loss: float,
        reward: float = 0.0,
        success_rate: float = 0.0,
    ) -> None:
        metrics = {"loss": loss, "reward": reward, "success_rate": success_rate}
        mlflow.log_metrics(metrics, step=step)
        if getattr(self, "_wandb_enabled", False):
            self._wandb.log(metrics, step=step)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the training loop from the current step to ``total_steps``."""
        self._setup_wandb()
        total = self._cfg.training.total_steps
        save_every = self._cfg.training.save_every_steps

        dataloader_iter: Iterator[dict[str, torch.Tensor]] = iter(self._dataloader)

        try:
            for step in range(self._step, total):
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(self._dataloader)
                    batch = next(dataloader_iter)

                self._optimizer.zero_grad()
                loss_tensor = self._policy.forward(batch)
                loss_tensor.backward()  # type: ignore[no-untyped-call]
                self._optimizer.step()

                current_step = step + 1
                if current_step % save_every == 0:
                    loss_val = float(loss_tensor.detach())
                    self._save_checkpoint(current_step)
                    self._log_metrics(current_step, loss_val)
                    logger.info(f"Step {current_step}/{total} | loss={loss_val:.4f}")
        finally:
            mlflow.end_run()

        logger.info("Training complete.")
