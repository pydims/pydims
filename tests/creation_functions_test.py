# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
import numpy as np

import pydims as dms
from pydims.string_unit import StringUnit
from pydims.testing import assert_identical


def test_create_creation_functions_from_numpy():
    make = dms.CreationFunctions(np, StringUnit)
    x = make.linspace('x', 0, 1, 3, unit='m')
    assert_identical(
        x,
        dms.DimensionedArray(
            values=np.linspace(0, 1, 3), dims=('x',), unit=StringUnit('m')
        ),
    )
