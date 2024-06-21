# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
import dask.array as da
import numpy as np

import pydims as dms
from pydims.common import unary
from pydims.testing import assert_identical


def compute(x):
    return x.compute()


def test_create_from_chunked_dask_array():
    values = da.ones((10, 10), chunks=(5, 5))
    dims = ('x', 'y')
    a = dms.DimensionedArray(values=values, dims=dims, unit=None)
    b = a + a
    # Binary op should not trigger computation
    assert not isinstance(b.values, np.ndarray)
    result = unary(b, values_op=lambda x: x.compute(), unit_op=None)
    assert isinstance(result.values, np.ndarray)
    assert_identical(
        result, dms.DimensionedArray(values=2 * np.ones((10, 10)), dims=dims, unit=None)
    )


def test_slice_chunked_dask_array():
    values = da.ones((10, 10), chunks=(5, 5))
    dims = ('x', 'y')
    a = dms.DimensionedArray(values=values, dims=dims, unit=None)
    b = a[{'x': slice(4, 8), 'y': slice(6, 8)}]
    # Slicing should not compute
    assert not isinstance(b.values, np.ndarray)
    result = unary(b, values_op=lambda x: x.compute(), unit_op=None)
    assert isinstance(result.values, np.ndarray)
    assert_identical(
        result, dms.DimensionedArray(values=np.ones((4, 2)), dims=dims, unit=None)
    )


def test_can_create_chunked_using_CreationFunctions():
    make = dms.CreationFunctions(da, None)
    x = make.linspace('x', 0, 1, 4, unit=None, chunks=(2,))
    assert x.values.chunks == ((2, 2),)
