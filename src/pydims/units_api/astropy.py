# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
from __future__ import annotations

import astropy.units as _units

dimensionless = _units.dimensionless_unscaled

Unit = _units.Unit


def get_scale(*, src, dst) -> float:
    return src.to(dst)
