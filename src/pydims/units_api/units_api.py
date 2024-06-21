# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
from __future__ import annotations

import sys
from typing import Any


def _is_astropy_unit(unit: Any) -> bool:
    if 'astropy.units' not in sys.modules:
        return False
    import astropy.units

    return issubclass(unit.__class__, astropy.units.UnitBase)


def _is_pint_unit(unit: Any) -> bool:
    if 'pint' not in sys.modules:
        return False
    import pint

    return issubclass(unit.__class__, pint.Unit)


def is_scipp_unit(unit: Any) -> bool:
    if 'scipp' not in sys.modules:
        return False
    import scipp

    return isinstance(unit, scipp.units.Unit)


def _is_string_unit(unit: Any) -> bool:
    from pydims.string_unit import StringUnit

    return issubclass(unit.__class__, StringUnit)


def units_namespace(unit: Any) -> Any:
    if _is_astropy_unit(unit):
        from . import astropy

        return astropy
    elif _is_pint_unit(unit):
        from .pint import PintsUnitsNamespace

        return PintsUnitsNamespace(unit._REGISTRY)
    elif is_scipp_unit(unit):
        from . import scipp

        return scipp
    elif _is_string_unit(unit):
        from . import string_unit

        return string_unit


__all__ = ['units_namespace']
