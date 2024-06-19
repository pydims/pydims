# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Pydray contributors (https://github.com/pydray)
from dataclasses import dataclass
from typing import Protocol

from collections.abc import Hashable


class Values(Protocol):
    """Array of values following the Python array API standard."""

    pass


class Unit(Protocol):
    pass


@dataclass
class Dray:
    values: Values
    dims: tuple[Hashable, ...]
    unit: Unit | None
