# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)

import scipp as _scipp

dimensionless = _scipp.units.dimensionless
Unit = _scipp.units.Unit


def get_scale(*, src, dst) -> float:
    q_src = _scipp.scalar(1, unit=src)
    q_dst = _scipp.to_unit(q_src, unit=dst)
    return float(q_dst.value)
