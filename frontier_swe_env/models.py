# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Frontier Swe Env Environment.

The frontier_swe_env environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class FrontierSweAction(Action):
    """Action for the Frontier Swe Env environment - just a message to echo."""

    message: str = Field(..., description="Message to echo back")


class FrontierSweObservation(Observation):
    """Observation from the Frontier Swe Env environment - the echoed message."""

    echoed_message: str = Field(default="", description="The echoed message")
    message_length: int = Field(default=0, description="Length of the echoed message")
