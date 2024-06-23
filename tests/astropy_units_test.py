# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
import array_api_strict as array
from astropy import units

import pydims as dms
from pydims.testing import assert_identical

make = dms.CreationFunctions(array=array, units=units)


def test_from_astropy_units():
    x = make.linspace('x', 0, 1, 3, unit='m')
    assert_identical(
        x,
        dms.DimensionedArray(
            values=array.linspace(0, 1, 3), dims=('x',), unit=units.meter
        ),
    )
    assert (x + x).unit == units.meter
    assert (x * x).unit == units.meter**2


def test_unit_conversion():
    m = make.linspace('x', 0, 1, 3, unit='m')
    km = m.to(unit='km')
    assert km.unit == units.km
    assert all(km.values == m.values / 1000)


def test_units_namespace_detects_astropy():
    from pydims.units_api_compat import units_namespace

    units_api = units_namespace(units.Unit(''))
    assert units_api is not None
    assert units_api.Unit('m') == units.meter
