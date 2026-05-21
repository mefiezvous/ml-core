# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint manager with local save and optional HuggingFace Hub push."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from loguru import logger

try:
    from huggingface_hub import upload_file as hf_upload

    HF_AVAILABLE = True
except ImportError:
    hf_upload = None  # type: ignore[assignment]
    HF_AVAILABLE = False


class CheckpointManager:
    """Saves checkpoints locally and optionally pushes to HuggingFace Hub.

    Args:
        local_dir: Directory where checkpoints are saved.
        hf_repo_id: HuggingFace Hub repo ID (e.g. "user/repo"). None disables push.
        push_every: Number of steps between Hub pushes.
        keep_last_n: Number of recent checkpoints to keep locally.
    """

    def __init__(
        self,
        local_dir: Path,
        hf_repo_id: str | None = None,
        push_every: int = 500,
        keep_last_n: int = 3,
    ) -> None:
        self._local_dir = local_dir
        self.hf_repo_id = hf_repo_id
        self._push_every = push_every
        self._keep_last_n = keep_last_n

    def save(self, state: dict[str, Any], step: int) -> Path:
        """Save checkpoint locally and push to Hub if on interval."""
        self._local_dir.mkdir(parents=True, exist_ok=True)
        path = self._local_dir / f"step_{step:06d}.pt"
        torch.save(state, path)
        logger.debug(f"Checkpoint saved: {path.name}")
        self._prune_old()
        if step % self._push_every == 0:
            self._push_to_hub(path, step)
        return path

    def load_latest(self) -> tuple[dict[str, Any], int] | None:
        """Load the most recent checkpoint. Returns (state, step) or None."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        step, path = checkpoints[-1]
        state: dict[str, Any] = torch.load(path, weights_only=False)
        return state, step

    def list_checkpoints(self) -> list[tuple[int, Path]]:
        """Return (step, path) pairs sorted by step ascending."""
        if not self._local_dir.exists():
            return []
        result: list[tuple[int, Path]] = []
        for f in self._local_dir.glob("step_*.pt"):
            try:
                step = int(f.stem.split("_", 1)[1])
                result.append((step, f))
            except (ValueError, IndexError):
                continue
        return sorted(result, key=lambda x: x[0])

    def _push_to_hub(self, path: Path, step: int) -> None:
        if not HF_AVAILABLE or self.hf_repo_id is None:
            return
        logger.info(f"Pushing checkpoint step {step} to {self.hf_repo_id}")
        hf_upload(
            path_or_fileobj=str(path),
            path_in_repo=f"checkpoints/step_{step:06d}.pt",
            repo_id=self.hf_repo_id,
        )

    def _prune_old(self) -> None:
        """Delete oldest checkpoints, keeping only the most recent keep_last_n."""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= self._keep_last_n:
            return
        for _, path in checkpoints[: -self._keep_last_n]:
            path.unlink()
            logger.debug(f"Pruned checkpoint: {path.name}")
