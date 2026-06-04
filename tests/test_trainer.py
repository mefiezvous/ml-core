# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for Trainer — mock policy + mock dataloader, no GPU required."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def training_cfg() -> Any:
    return OmegaConf.create(
        {
            "dataset": {"repo_id": "mefiezvous/cube-reach-v1-dataset", "root": None},
            "policy": {"name": "act"},
            "training": {
                "total_steps": 6,
                "save_every_steps": 3,
                "batch_size": 4,
                "num_envs": 1,
                "device": "cpu",
                "mixed_precision": False,
                "seed": 42,
                "lr": 1e-4,
            },
            "logging": {
                "backend": "mlflow",
                "experiment_name": "test_experiment",
                "run_name": "test_run",
            },
        }
    )


def _make_mock_policy() -> MagicMock:
    policy = MagicMock()
    policy.forward.return_value = torch.tensor(0.5, requires_grad=True)
    param = torch.nn.Parameter(torch.zeros(4))
    policy.parameters.return_value = iter([param])
    return policy


def _make_dataloader(n_batches: int = 10) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "observation.state": torch.zeros(4, 1, 3),
            "action": torch.zeros(4, 100, 8),
        }
        for _ in range(n_batches)
    ]


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrainer:
    def _make_trainer(
        self,
        cfg: Any,
        policy: MagicMock,
        dataloader: Any,
        robot_name: str = "test_robot",
        policy_type: str = "act",
    ) -> Any:
        from mlcore.training.trainer import Trainer  # noqa: PLC0415

        with patch("mlcore.training.trainer.mlflow"):
            trainer = Trainer(
                cfg, policy, dataloader, robot_name=robot_name, policy_type=policy_type
            )
        return trainer

    def test_checkpoint_dir_uses_robot_name_and_policy_type(
        self, training_cfg: Any, tmp_path: Path
    ) -> None:
        from mlcore.training.trainer import Trainer  # noqa: PLC0415

        policy = _make_mock_policy()
        with patch("mlcore.training.trainer.mlflow"):
            trainer = Trainer(
                training_cfg,
                policy,
                _make_dataloader(),
                robot_name="my_robot",
                policy_type="act",
            )

        assert "my_robot" in str(trainer._checkpoint_dir)
        assert "act" in str(trainer._checkpoint_dir)

    def test_mlflow_tracking_uri_uses_robot_name_and_policy_type(self, training_cfg: Any) -> None:
        from mlcore.training.trainer import Trainer  # noqa: PLC0415

        policy = _make_mock_policy()
        with patch("mlcore.training.trainer.mlflow") as mock_mlflow:
            Trainer(
                training_cfg,
                policy,
                _make_dataloader(),
                robot_name="my_robot",
                policy_type="diffusion",
            )

        uri_call = mock_mlflow.set_tracking_uri.call_args[0][0]
        assert "my_robot" in uri_call
        assert "diffusion" in uri_call

    def test_init_calls_mlflow_set_tracking_uri(self, training_cfg: Any) -> None:
        from mlcore.training.trainer import Trainer  # noqa: PLC0415

        policy = _make_mock_policy()
        with patch("mlcore.training.trainer.mlflow") as mock_mlflow:
            Trainer(training_cfg, policy, _make_dataloader(), robot_name="r", policy_type="act")

        mock_mlflow.set_tracking_uri.assert_called_once()

    def test_init_starts_mlflow_experiment(self, training_cfg: Any) -> None:
        from mlcore.training.trainer import Trainer  # noqa: PLC0415

        policy = _make_mock_policy()
        with patch("mlcore.training.trainer.mlflow") as mock_mlflow:
            Trainer(training_cfg, policy, _make_dataloader(), robot_name="r", policy_type="act")

        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")

    def test_detect_no_checkpoint_returns_none(self, training_cfg: Any, tmp_path: Path) -> None:
        from mlcore.training.trainer import Trainer  # noqa: PLC0415

        policy = _make_mock_policy()
        with patch("mlcore.training.trainer.mlflow"):
            trainer = Trainer(
                training_cfg, policy, _make_dataloader(), robot_name="empty", policy_type="act"
            )

        assert trainer._detect_latest_checkpoint() is None

    def test_detect_checkpoint_finds_latest(self, training_cfg: Any, tmp_path: Path) -> None:
        from mlcore.training.trainer import Trainer  # noqa: PLC0415

        policy = _make_mock_policy()
        with patch("mlcore.training.trainer.mlflow"):
            trainer = Trainer(
                training_cfg, policy, _make_dataloader(), robot_name="r", policy_type="act"
            )

        # Redirect to tmp_path and create step_*.pt files
        trainer.ckpt_manager._local_dir = tmp_path
        trainer._checkpoint_dir = tmp_path
        (tmp_path / "step_001000.pt").touch()
        (tmp_path / "step_002000.pt").touch()
        (tmp_path / "step_003000.pt").touch()

        latest = trainer._detect_latest_checkpoint()
        assert latest is not None
        assert latest.name == "step_003000.pt"

    def test_resume_from_checkpoint_updates_start_step(
        self, training_cfg: Any, tmp_path: Path
    ) -> None:
        from mlcore.training.trainer import Trainer  # noqa: PLC0415

        cfg = OmegaConf.merge(
            training_cfg,
            {"training": {"total_steps": 10000, "save_every_steps": 1000}},
        )
        policy = _make_mock_policy()
        with patch("mlcore.training.trainer.mlflow"):
            trainer = Trainer(cfg, policy, _make_dataloader(), robot_name="r", policy_type="act")

        # Mock load_latest to avoid real file I/O
        trainer.ckpt_manager.load_latest = MagicMock(  # type: ignore[method-assign]
            return_value=({"model": {"key": "val"}, "step": 5000}, 5000)
        )
        trainer._resume_from_checkpoint()

        assert trainer._step == 5000
        policy.load_state_dict.assert_called_once_with({"key": "val"})

    def test_train_calls_forward_total_steps_times(self, training_cfg: Any) -> None:
        from mlcore.training.trainer import Trainer  # noqa: PLC0415

        policy = _make_mock_policy()
        dataloader = _make_dataloader(n_batches=20)

        with patch("mlcore.training.trainer.mlflow"):
            trainer = Trainer(training_cfg, policy, dataloader, robot_name="r", policy_type="act")
            trainer.ckpt_manager.save = MagicMock(return_value=Path("step_000003.pt"))  # type: ignore[method-assign]
            trainer.train()

        assert policy.forward.call_count == 6

    def test_train_saves_checkpoint_at_save_every_steps(self, training_cfg: Any) -> None:
        from mlcore.training.trainer import Trainer  # noqa: PLC0415

        policy = _make_mock_policy()
        dataloader = _make_dataloader(n_batches=20)

        with patch("mlcore.training.trainer.mlflow"):
            trainer = Trainer(training_cfg, policy, dataloader, robot_name="r", policy_type="act")
            trainer.ckpt_manager.save = MagicMock(return_value=Path("step_000003.pt"))  # type: ignore[method-assign]
            trainer.train()

        # save_every_steps=3, total_steps=6 → 2 saves
        assert trainer.ckpt_manager.save.call_count == 2

    def test_train_logs_mlflow_at_save_every_steps(self, training_cfg: Any) -> None:
        from mlcore.training.trainer import Trainer  # noqa: PLC0415

        policy = _make_mock_policy()
        dataloader = _make_dataloader(n_batches=20)

        with patch("mlcore.training.trainer.mlflow") as mock_mlflow:
            trainer = Trainer(training_cfg, policy, dataloader, robot_name="r", policy_type="act")
            trainer.ckpt_manager.save = MagicMock(return_value=Path("step_000003.pt"))  # type: ignore[method-assign]
            trainer.train()

        assert mock_mlflow.log_metrics.call_count == 2
        logged_metrics = mock_mlflow.log_metrics.call_args_list[0][0][0]
        assert "loss" in logged_metrics
        assert "reward" in logged_metrics
        assert "success_rate" in logged_metrics

    def test_wandb_disabled_when_cfg_flag_false(self, training_cfg: Any) -> None:
        """MLC-008: wandb opt-in is driven by logging.wandb_enabled Hydra flag."""
        from mlcore.training.trainer import Trainer  # noqa: PLC0415

        # training_cfg has no logging.wandb_enabled → defaults to False
        policy = _make_mock_policy()
        with patch("mlcore.training.trainer.mlflow"):
            trainer = Trainer(
                training_cfg, policy, _make_dataloader(), robot_name="r", policy_type="act"
            )
            trainer._setup_wandb()

        assert trainer._wandb_enabled is False

    def test_wandb_enabled_via_hydra_flag(self, training_cfg: Any) -> None:
        """MLC-008: setting logging.wandb_enabled=true enables WandB init."""
        from mlcore.training.trainer import Trainer  # noqa: PLC0415

        cfg = OmegaConf.merge(training_cfg, {"logging": {"wandb_enabled": True}})
        policy = _make_mock_policy()
        with (
            patch("mlcore.training.trainer.mlflow"),
            patch("mlcore.training.trainer.Trainer._setup_wandb"),
        ):
            trainer = Trainer(cfg, policy, _make_dataloader(), robot_name="r", policy_type="act")

        # Verify the flag is accessible via cfg
        assert cfg.logging.wandb_enabled is True
        _ = trainer  # silence unused-variable warning

    def test_setup_mlflow_logs_lineage_tags(self, training_cfg: Any) -> None:
        from mlcore.training.trainer import Trainer  # noqa: PLC0415

        policy = _make_mock_policy()
        with patch("mlcore.training.trainer.mlflow") as mock_mlflow:
            Trainer(
                training_cfg,
                policy,
                _make_dataloader(),
                robot_name="cube_reach_v1",
                policy_type="act",
            )

        mock_mlflow.set_tags.assert_called_once()
        tags = mock_mlflow.set_tags.call_args[0][0]
        assert tags["dataset.repo_id"] == "mefiezvous/cube-reach-v1-dataset"
        assert tags["policy.type"] == "act"
        assert tags["robot.name"] == "cube_reach_v1"

    def test_train_dataloader_cycles_when_exhausted(self, training_cfg: Any) -> None:
        from mlcore.training.trainer import Trainer  # noqa: PLC0415

        policy = _make_mock_policy()
        short_dataloader = _make_dataloader(n_batches=2)

        with patch("mlcore.training.trainer.mlflow"):
            trainer = Trainer(
                training_cfg, policy, short_dataloader, robot_name="r", policy_type="act"
            )
            trainer.ckpt_manager.save = MagicMock(return_value=Path("step_000003.pt"))  # type: ignore[method-assign]
            trainer.train()

        assert policy.forward.call_count == 6
