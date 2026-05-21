# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for ml-core tests."""

from __future__ import annotations

import shutil
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def clean_checkpoints() -> Generator[None, None, None]:
    """Remove any leftover checkpoint directories before and after each test."""
    ckpt_root = Path("checkpoints")
    if ckpt_root.exists():
        shutil.rmtree(ckpt_root)
    yield
    if ckpt_root.exists():
        shutil.rmtree(ckpt_root)
