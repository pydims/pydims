# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
from __future__ import annotations

from pint import UnitRegistry


class PintsUnitsNamespace:
    def __init__(self, ureg: UnitRegistry):
        self._ureg = ureg

    def __getattr__(self, item):
        return getattr(self._ureg, item)

    def get_scale(self, *, src, dst) -> float:
        q_src = self._ureg.Quantity(1, src)
        q_dst = q_src.to(dst)
        return q_dst.magnitude
