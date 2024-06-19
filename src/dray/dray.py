# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Pydray contributors (https://github.com/pydray)
from dataclasses import dataclass
from typing import Protocol

from collections.abc import Hashable


class ArrayImplementation(Protocol):
    """Array of values following the Python array API standard."""


class UnitImplementation(Protocol):
    pass


@dataclass
class Dray:
    """
    General idea:
    - __getitem__ accepts dict with dims labels. Only 1-D allows for omitting index.
    - Probably we need to support duplicate dims
    """

    values: ArrayImplementation
    dims: tuple[Hashable, ...]
    unit: UnitImplementation | None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    # TODO
    # - not mutable dict
    # - not dict, since duplicates need to be supported
    @property
    def sizes(self) -> dict[Hashable, int]:
        return dict(zip(self.dims, self.shape))
