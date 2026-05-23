# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``mlcore.collection.teleop``."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from mlcore.collection import TeleopCollector


@pytest.mark.unit
def test_teleop_collector_accepts_arbitrary_constructor_args() -> None:
    """The stub constructor swallows any positional/keyword args."""
    coll: Any = TeleopCollector("anything", config={"foo": "bar"})
    assert coll is not None


@pytest.mark.unit
def test_teleop_collector_collect_raises_not_implemented() -> None:
    coll = TeleopCollector()
    with pytest.raises(NotImplementedError, match="stub"):
        coll.collect(MagicMock(), n_episodes=1)
