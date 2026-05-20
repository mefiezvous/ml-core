# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
"""Plug-and-play robot specification system."""

from mlcore.robots.base import RobotSpec as RobotSpec
from mlcore.robots.config_builder import build_train_config as build_train_config
from mlcore.robots.generic_policy import GenericScriptedPolicy as GenericScriptedPolicy
from mlcore.robots.registry import get as get
from mlcore.robots.registry import list_specs as list_specs
from mlcore.robots.registry import register as register
from mlcore.robots.specs import cube_reach_v1 as _cube_reach_v1  # noqa: F401 — triggers registration
