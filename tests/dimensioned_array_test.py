# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
import array_api_strict as array
import pytest

import pydims as dms
from pydims.string_unit import StringUnit
from pydims.testing import assert_identical


@pytest.mark.parametrize('dims', [(), ('x',), ('x', 'y', 'z')])
def test_init_raises_if_dims_has_wrong_length(dims: tuple[str, ...]):
    with pytest.raises(ValueError, match="Number of dimensions"):
        dms.DimensionedArray(values=array.ones((2, 3)), dims=dims, unit=None)


def test_sizes():
    da = dms.DimensionedArray(values=array.ones((2, 3)), dims=('x', 'y'), unit=None)
    assert da.sizes == {'x': 2, 'y': 3}


def test_getitem_no_dim_raises_if_not_1d():
    da = dms.DimensionedArray(values=array.ones((2, 3)), dims=('x', 'y'), unit=None)
    with pytest.raises(
        dms.DimensionError, match="Only 1-D arrays can be indexed without dims"
    ):
        _ = da[0]


def test_getitem_1d_no_dim():
    da = dms.DimensionedArray(dims=('x',), values=array.arange(2.0), unit=None)
    assert_identical(
        da[0], dms.DimensionedArray(dims=(), values=array.asarray(0.0), unit=None)
    )
    assert_identical(
        da[1], dms.DimensionedArray(dims=(), values=array.asarray(1.0), unit=None)
    )


def test_getitem_2d_with_dims():
    da = dms.DimensionedArray(
        values=array.reshape(array.arange(6), (2, 3)), dims=('x', 'y'), unit=None
    )
    assert_identical(
        da[{'x': 0}],
        dms.DimensionedArray(dims=('y',), values=array.asarray([0, 1, 2]), unit=None),
    )
    assert_identical(
        da[{'y': 0}],
        dms.DimensionedArray(dims=('x',), values=array.asarray([0, 3]), unit=None),
    )
    assert_identical(
        da[{'x': 1, 'y': 1}],
        dms.DimensionedArray(dims=(), values=array.asarray(4), unit=None),
    )


@pytest.mark.parametrize('unit', [None, StringUnit(), StringUnit('m')])
def test_neg(unit: StringUnit | None):
    da = dms.DimensionedArray(values=array.ones((2, 3)), dims=('x', 'y'), unit=unit)
    result = -da
    assert_identical(
        result,
        dms.DimensionedArray(values=-array.ones((2, 3)), dims=('x', 'y'), unit=unit),
    )


def test_exp_raises_if_unit_is_not_dimensionless():
    da = dms.DimensionedArray(
        values=array.ones((2, 3)), dims=('x', 'y'), unit=StringUnit('m')
    )
    with pytest.raises(ValueError, match="Unit must be dimensionless"):
        dms.exp(da)


def test_add_raises_if_units_differ():
    da1 = dms.DimensionedArray(
        values=array.ones((2, 3)), dims=('x', 'y'), unit=StringUnit('m')
    )
    da2 = dms.DimensionedArray(
        values=array.ones((2, 3)), dims=('x', 'y'), unit=StringUnit('s')
    )
    with pytest.raises(ValueError, match="Units must be identical"):
        da1 + da2


def test_elemwise_binary_broadcasts_dims():
    xy = dms.DimensionedArray(values=array.ones((2, 3)), dims=('x', 'y'), unit=None)
    yz = dms.DimensionedArray(values=array.ones((3, 4)), dims=('y', 'z'), unit=None)
    result = dms.common.elemwise_binary(
        xy, yz, values_op=lambda a, b: a + b, unit_op=lambda a, b: a
    )
    assert result.dims == ('x', 'y', 'z')


def test_elemwise_binary_transposes_dims():
    xy = dms.DimensionedArray(values=array.ones((2, 3)), dims=('x', 'y'), unit=None)
    yx = dms.DimensionedArray(values=array.ones((3, 2)), dims=('y', 'x'), unit=None)
    result = dms.common.elemwise_binary(
        xy, yx, values_op=lambda a, b: a + b, unit_op=lambda a, b: a
    )
    assert result.dims == ('x', 'y')


def test_elemwise_binary_broadcasts_and_transposes_dims():
    xy = dms.DimensionedArray(values=array.ones((2, 3)), dims=('x', 'y'), unit=None)
    yxz = dms.DimensionedArray(
        values=array.ones((3, 2, 4)), dims=('y', 'x', 'z'), unit=None
    )
    result = dms.common.elemwise_binary(
        xy, yxz, values_op=lambda a, b: a + b, unit_op=lambda a, b: a
    )
    assert result.sizes == {'x': 2, 'y': 3, 'z': 4}
