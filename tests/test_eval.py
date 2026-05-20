# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for Evaluator and EvalResult."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_mock_policy() -> MagicMock:
    policy = MagicMock()
    policy.select_action.return_value = np.zeros(8, dtype=np.float32)
    return policy


def _make_mock_env(*, success: bool = False) -> MagicMock:
    env = MagicMock()
    env.reset.return_value = np.zeros(3, dtype=np.float32)
    env.step.return_value = (np.zeros(3), 1.0, True, {"success": success})
    return env


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEvaluator:
    def test_evaluator_runs_n_episodes(self) -> None:
        from mlcore.eval.evaluator import Evaluator  # noqa: PLC0415

        policy = _make_mock_policy()
        env = _make_mock_env()
        ev = Evaluator(
            policy, "CubeReachV1", n_episodes=5, env=env,
            robot_name="test_robot", policy_type="act",
        )
        result = ev.evaluate()

        assert result.n_episodes == 5
        assert env.reset.call_count == 5
        assert len(result.per_episode_rewards) == 5

    def test_evaluator_success_rate(self) -> None:
        from mlcore.eval.evaluator import EvalResult, Evaluator  # noqa: PLC0415

        policy = _make_mock_policy()
        env: MagicMock = MagicMock()
        env.reset.return_value = np.zeros(3)
        env.step.side_effect = [
            (np.zeros(3), 5.0, True, {"success": True}),
            (np.zeros(3), 5.0, True, {"success": True}),
            (np.zeros(3), 0.5, True, {"success": False}),
        ]

        ev = Evaluator(
            policy, "CubeReachV1", n_episodes=3, env=env,
            robot_name="test_robot", policy_type="act",
        )
        result = ev.evaluate()

        assert isinstance(result, EvalResult)
        assert result.success_rate == pytest.approx(2 / 3)

    def test_evaluator_logs_to_mlflow(self) -> None:
        from mlcore.eval.evaluator import Evaluator  # noqa: PLC0415

        policy = _make_mock_policy()
        env = _make_mock_env(success=True)

        with patch("mlcore.eval.evaluator.mlflow") as mock_mlflow:
            ev = Evaluator(
                policy, "CubeReachV1", n_episodes=2, mlflow_run_id="run_abc", env=env,
                robot_name="test_robot", policy_type="act",
            )
            ev.evaluate()

        mock_mlflow.start_run.assert_called_once_with(run_id="run_abc")
        mock_mlflow.log_metrics.assert_called_once()
        logged: Any = mock_mlflow.log_metrics.call_args[0][0]
        assert "eval/success_rate" in logged
        assert "eval/mean_reward" in logged

    def test_eval_report_written_to_namespaced_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from mlcore.eval.evaluator import Evaluator  # noqa: PLC0415

        monkeypatch.chdir(tmp_path)
        policy = _make_mock_policy()
        env = _make_mock_env(success=True)

        ev = Evaluator(
            policy, "CubeReachV1", n_episodes=2, env=env,
            robot_name="my_robot", policy_type="diffusion",
        )
        ev.evaluate()

        report_path = tmp_path / "eval_reports" / "my_robot" / "diffusion" / "eval_report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert "success_rate" in data
        assert "mean_reward" in data

    def test_n_episodes_must_be_positive(self) -> None:
        from mlcore.eval.evaluator import Evaluator  # noqa: PLC0415

        with pytest.raises(ValueError, match="n_episodes"):
            Evaluator(
                _make_mock_policy(), "CubeReachV1", n_episodes=0, env=_make_mock_env(),
                robot_name="r", policy_type="act",
            )

    def test_report_path_uses_robot_name_and_policy_type(self) -> None:
        from mlcore.eval.evaluator import Evaluator  # noqa: PLC0415

        ev = Evaluator(
            _make_mock_policy(), "CubeReachV1", n_episodes=1, env=_make_mock_env(),
            robot_name="my_robot", policy_type="act",
        )
        assert "my_robot" in str(ev._report_path)
        assert "act" in str(ev._report_path)
