# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
import numpy as np
import pytest

import pydims as dms
from pydims.testing import assert_identical

make = dms.CreationFunctions(array=np, units=dms.string_units)


@pytest.mark.parametrize(
    'indices',
    [
        make.ones(dims=(), shape=(), unit=None),
        make.ones(dims=('x', 'y'), shape=(1, 1), unit=None),
    ],
)
def test_take_raises_if_indices_are_not_1d(indices: dms.DimensionedArray):
    arr = make.asarray(dims=('x', 'y'), values=[[1, 2, 3], [4, 5, 6]], unit=None)
    with pytest.raises(dms.DimensionError, match="Number of dimensions must be 1"):
        dms.take(arr, indices)


def test_take_raises_in_dim_of_indices_not_in_data_dims():
    arr = make.asarray(dims=('x', 'y'), values=[[1, 2, 3], [4, 5, 6]], unit=None)
    indices = make.asarray(dims=('z',), values=[0], unit=None)
    with pytest.raises(dms.DimensionError, match="not in data dimensions"):
        dms.take(arr, indices)


def test_take_extract_correct_values_and_preserves_unit():
    arr = make.asarray(dims=('x', 'y'), values=[[1, 2, 3], [4, 5, 6]], unit='m')

    indices_x = make.asarray(dims=('x',), values=[1, 0], unit=None)
    result = dms.take(arr, indices_x)
    expected = make.asarray(dims=('x', 'y'), values=[[4, 5, 6], [1, 2, 3]], unit='m')
    assert_identical(result, expected)

    indices_y = make.asarray(dims=('y',), values=[0, 2, 0, 1], unit=None)
    result = dms.take(arr, indices_y)
    expected = make.asarray(
        dims=('x', 'y'), values=[[1, 3, 1, 2], [4, 6, 4, 5]], unit='m'
    )
    assert_identical(result, expected)
