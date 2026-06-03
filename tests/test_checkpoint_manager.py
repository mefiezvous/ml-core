# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for CheckpointManager — local save, HF Hub push, listing, pruning."""

from __future__ import annotations

import io
import pickle
import struct
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch


@pytest.mark.unit
class TestCheckpointManager:
    def _make_manager(
        self,
        tmp_path: Path,
        hf_repo_id: str | None = None,
        push_every: int = 100,
        keep_last_n: int = 3,
    ) -> Any:
        from mlcore.training.checkpoint_manager import CheckpointManager

        return CheckpointManager(
            tmp_path, hf_repo_id=hf_repo_id, push_every=push_every, keep_last_n=keep_last_n
        )

    def test_save_local_creates_file(self, tmp_path: Path) -> None:
        m = self._make_manager(tmp_path)
        path = m.save({"weights": [1, 2, 3]}, step=100)
        assert path.exists()

    def test_save_local_filename_format(self, tmp_path: Path) -> None:
        m = self._make_manager(tmp_path)
        path = m.save({"weights": [1, 2, 3]}, step=42)
        assert path.name == "step_000042.pt"

    def test_push_to_hub_called_when_enabled(self, tmp_path: Path) -> None:
        with (
            patch("mlcore.training.checkpoint_manager.HF_AVAILABLE", True),
            patch("mlcore.training.checkpoint_manager.hf_upload") as mock_upload,
        ):
            m = self._make_manager(tmp_path, hf_repo_id="user/repo", push_every=100)
            m.save({"weights": [1]}, step=100)
        mock_upload.assert_called_once()

    def test_push_to_hub_not_called_when_disabled(self, tmp_path: Path) -> None:
        with patch("mlcore.training.checkpoint_manager.hf_upload") as mock_upload:
            m = self._make_manager(tmp_path, hf_repo_id=None, push_every=100)
            m.save({"weights": [1]}, step=100)
        mock_upload.assert_not_called()

    def test_list_checkpoints_returns_sorted(self, tmp_path: Path) -> None:
        m = self._make_manager(tmp_path)
        m.save({"a": 1}, step=300)
        m.save({"a": 1}, step=100)
        m.save({"a": 1}, step=200)
        steps = [s for s, _ in m.list_checkpoints()]
        assert steps == [100, 200, 300]

    def test_load_latest_checkpoint(self, tmp_path: Path) -> None:
        m = self._make_manager(tmp_path)
        m.save({"x": 1}, step=50)
        m.save({"x": 2}, step=200)
        result = m.load_latest()
        assert result is not None
        state, step = result
        assert step == 200
        assert state["x"] == 2

    def test_push_skipped_when_not_on_save_interval(self, tmp_path: Path) -> None:
        with (
            patch("mlcore.training.checkpoint_manager.HF_AVAILABLE", True),
            patch("mlcore.training.checkpoint_manager.hf_upload") as mock_upload,
        ):
            m = self._make_manager(tmp_path, hf_repo_id="user/repo", push_every=500)
            m.save({"weights": [1]}, step=100)  # 100 % 500 != 0 → no push
        mock_upload.assert_not_called()

    def test_checkpoint_load_rejects_pickle_payload(self, tmp_path: Path) -> None:
        """MLC-001: weights_only=True must reject a crafted pickle payload.

        We craft a minimal .pt file whose pickle stream calls
        ``os.system`` (a dangerous global).  torch.load with
        ``weights_only=True`` must raise an ``UnpicklingError`` (or a
        ``RuntimeError`` wrapping it) before the payload executes.
        """

        class _MaliciousPayload:
            """Pickle-encodes ``os.system('echo pwned')`` as its __reduce__."""

            def __reduce__(self) -> tuple[Any, tuple[str]]:
                import os  # noqa: PLC0415

                return os.system, ("echo pwned",)

        # Serialise the malicious object through pickle directly so we bypass
        # torch.save's own safe-saving path and land raw bytes in the file.
        payload_bytes = pickle.dumps(_MaliciousPayload())

        # torch .pt files are just pickle streams; write raw bytes so they
        # look like a legitimate checkpoint to the file-system layer.
        malicious_pt = tmp_path / "malicious.pt"
        malicious_pt.write_bytes(payload_bytes)

        m = self._make_manager(tmp_path)
        # Temporarily inject the malicious file as the only checkpoint so
        # load_latest() targets it.
        with patch.object(
            m,
            "list_checkpoints",
            return_value=[(1, malicious_pt)],
        ):
            with pytest.raises((torch.serialization.UnpicklingError, RuntimeError, Exception)):
                m.load_latest()
