# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""BasePolicy Protocol — common interface for ACT, Diffusion, and future policy wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BasePolicy(Protocol):
    def select_action(self, obs: dict[str, np.ndarray]) -> np.ndarray: ...
    def reset(self) -> None: ...
    def forward(self, batch: dict[str, Any]) -> Any: ...
    def save(self, path: Path) -> None: ...
    def load_checkpoint(self, path: Path) -> None: ...
    def parameters(self) -> Any: ...
