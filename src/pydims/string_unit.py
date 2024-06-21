# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StringUnit:
    """Very simple unit class that stores a unit as a string."""

    value: str = ''

    def __mul__(self, other: StringUnit) -> StringUnit:
        return StringUnit(f'({self.value})*({other.value})')
