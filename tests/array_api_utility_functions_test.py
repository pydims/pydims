# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
import numpy as np

import pydims as dms
from pydims.testing import assert_identical


def test_all():
    da = dms.DimensionedArray(
        values=np.array([[True, True, False], [True, True, True]]),
        dims=('x', 'y'),
        unit=None,
    )
    assert_identical(
        dms.all(da, dim='x'),
        dms.DimensionedArray(
            values=np.array([True, True, False]), dims=('y',), unit=None
        ),
    )
    assert_identical(
        dms.all(da, dim='y'),
        dms.DimensionedArray(values=np.array([False, True]), dims=('x',), unit=None),
    )


def test_any():
    da = dms.DimensionedArray(
        values=np.array([[False, True, False], [False, False, False]]),
        dims=('x', 'y'),
        unit=None,
    )
    assert_identical(
        dms.any(da, dim='x'),
        dms.DimensionedArray(
            values=np.array([False, True, False]), dims=('y',), unit=None
        ),
    )
    assert_identical(
        dms.any(da, dim='y'),
        dms.DimensionedArray(values=np.array([True, False]), dims=('x',), unit=None),
    )
