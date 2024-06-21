# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
import array_api_strict as array
import astropy.units as u

import dims as dms
from dims.testing import assert_identical


def test_from_astropy_units():
    make = dms.CreationFunctions(array, u.Unit)
    x = make.linspace('x', 0, 1, 3, unit='m')
    assert_identical(
        x,
        dms.DimensionedArray(values=array.linspace(0, 1, 3), dims=('x',), unit=u.meter),
    )
    assert (x + x).unit == u.meter
    assert (x * x).unit == u.meter**2
