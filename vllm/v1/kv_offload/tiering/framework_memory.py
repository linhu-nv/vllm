# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TransportEndpoint:
    """In-process transport endpoint owned by the primary tier."""

    name: str
    end_point: Any
    info: str = ""


@dataclass(frozen=True)
class MemDescriptor:
    """Transport-addressable memory span for a pinned KV block."""

    end_point_name: str
    mem_type: str
    addr: int
    size: int
    device_Id: int
    info: str
