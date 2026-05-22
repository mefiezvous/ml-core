# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper around lerobot ACTPolicy for use in training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

try:
    from lerobot.common.policies.act.configuration_act import (
        ACTConfig,
    )
    from lerobot.common.policies.act.modeling_act import (
        ACTPolicy,
    )

    _LEROBOT_AVAILABLE = True
except ImportError:
    ACTPolicy = None
    ACTConfig = None
    _LEROBOT_AVAILABLE = False


class ACTWrapper:
    """Thin wrapper around :class:`lerobot.common.policies.act.modeling_act.ACTPolicy`.

    Converts between numpy observations (env API) and the tensor batches
    expected by the lerobot policy. Also provides save / resume helpers used
    by the :class:`~mlcore.training.trainer.Trainer`.

    Args:
        cfg: Hydra policy config (``configs/policy/act.yaml``).
        device: PyTorch device string (e.g. ``"cpu"``, ``"cuda"``).

    Raises:
        ImportError: If lerobot is not installed.
    """

    def __init__(self, cfg: DictConfig, device: str = "cpu") -> None:
        if not _LEROBOT_AVAILABLE:
            raise ImportError("lerobot is required. Install with: uv sync")

        self._cfg = cfg
        self._device = torch.device(device)

        if cfg.get("pretrained_repo"):
            logger.info(f"Loading ACTPolicy from pretrained: {cfg.pretrained_repo}")
            self._policy = ACTPolicy.from_pretrained(cfg.pretrained_repo)
        else:
            input_shapes: dict[str, list[int]] = OmegaConf.to_container(  # type: ignore[assignment]
                cfg.input_shapes, resolve=True
            )
            output_shapes: dict[str, list[int]] = OmegaConf.to_container(  # type: ignore[assignment]
                cfg.output_shapes, resolve=True
            )
            act_config = ACTConfig(
                n_obs_steps=cfg.n_obs_steps,
                n_action_steps=cfg.n_action_steps,
                chunk_size=cfg.chunk_size,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
            )
            self._policy = ACTPolicy(config=act_config)
            logger.info("ACTPolicy initialised from config (random weights)")

        self._policy.to(self._device)

    def select_action(
        self, obs: dict[str, np.ndarray[Any, np.dtype[Any]]]
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Convert an observation dict to a flat action array.

        Args:
            obs: Unbatched observation dict (keys match ``input_shapes``).

        Returns:
            Action array of shape ``(action_dim,)`` and dtype ``float32``.
        """
        batch: dict[str, torch.Tensor] = {
            k: torch.tensor(v, dtype=torch.float32, device=self._device).unsqueeze(0)
            for k, v in obs.items()
        }
        with torch.no_grad():
            action: torch.Tensor = self._policy.select_action(batch)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run a forward pass and return the scalar training loss."""
        output: dict[str, torch.Tensor] = self._policy.forward(batch)
        return output["loss"]

    def reset(self) -> None:
        """Clear the ACT action-chunking queue between episodes."""
        self._policy.reset()

    def save(self, path: Path) -> None:
        """Save model weights to a checkpoint file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self._policy.state_dict()}, path)
        logger.debug(f"ACTWrapper checkpoint saved → {path}")

    def load_checkpoint(self, path: Path) -> None:
        """Restore model weights from a checkpoint file."""
        checkpoint: dict[str, Any] = torch.load(path, map_location=self._device)
        self._policy.load_state_dict(checkpoint["model"])
        logger.debug(f"ACTWrapper checkpoint loaded ← {path}")

    def state_dict(self) -> dict[str, Any]:
        """Return underlying policy state dict."""
        return dict(self._policy.state_dict())

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load weights from a state dict."""
        self._policy.load_state_dict(state_dict)

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Yield underlying policy parameters for the optimiser."""
        return self._policy.parameters()

    @property
    def lerobot_policy(self) -> Any:
        """Direct access to the wrapped lerobot policy (advanced use)."""
        return self._policy
