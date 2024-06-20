# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Dims contributors (https://github.com/pydray)
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StringUnit:
    """Very simple unit class that stores a unit as a string."""

    value: str = ''

    @staticmethod
    def dimensionless() -> StringUnit:
        return StringUnit()
