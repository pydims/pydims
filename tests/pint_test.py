# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
import array_api_strict as array
from pint import UnitRegistry as ureg

import pydims as dms
from pydims.testing import assert_identical

make = dms.CreationFunctions(array, ureg.Unit)


def test_create_with_pint_unit():
    x = make.linspace('x', 0, 1, 3, unit='m')
    assert x.unit == ureg.Unit('m')


def test_to_unit():
    x = make.linspace('x', 0, 1, 3, unit='m')
    y = x.to(unit='cm')
    assert y.unit == ureg.Unit('cm')
    assert_identical(y, make.linspace('x', 0, 100, 3, unit='cm'))
