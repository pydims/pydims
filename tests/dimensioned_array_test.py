# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
import array_api_strict as array
import pytest

import pydims as dms
from pydims.string_units import Unit
from pydims.testing import assert_identical


@pytest.mark.parametrize('dims', [(), ('x',), ('x', 'y', 'z')])
def test_init_raises_if_dims_has_wrong_length(dims: tuple[str, ...]):
    with pytest.raises(ValueError, match="Number of dimensions"):
        dms.DimensionedArray(values=array.ones((2, 3)), dims=dims, unit=None)


def test_init_raises_if_dims_not_unique():
    with pytest.raises(ValueError, match="Dimensions must be unique"):
        dms.DimensionedArray(values=array.ones((2, 3)), dims=('x', 'x'), unit=None)


def test_sizes():
    da = dms.DimensionedArray(values=array.ones((2, 3)), dims=('x', 'y'), unit=None)
    assert da.sizes == {'x': 2, 'y': 3}


def test_getitem_ellipsis_returns_everything():
    da = dms.DimensionedArray(
        values=array.ones((2, 3, 4)), dims=('x', 'y', 'z'), unit=None
    )
    assert_identical(da[...], da)


def test_getitem_no_dim_raises_if_not_1d():
    da = dms.DimensionedArray(values=array.ones((2, 3)), dims=('x', 'y'), unit=None)
    with pytest.raises(
        dms.DimensionError, match="Only 1-D arrays can be indexed without dims"
    ):
        _ = da[0]


def test_getitem_raises_if_dim_not_in_dims():
    da = dms.DimensionedArray(values=array.ones((2, 3)), dims=('x', 'y'), unit=None)
    with pytest.raises(dms.DimensionError, match=r"Unknown dimensions: \('z',\)"):
        _ = da[{'z': 0}]


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


def test_getitem_order_in_dict_does_not_matter():
    da = dms.DimensionedArray(
        values=array.reshape(array.arange(6), (2, 3)), dims=('x', 'y'), unit=None
    )
    expected = dms.DimensionedArray(dims=(), values=array.asarray(3), unit=None)
    assert_identical(da[{'y': 0, 'x': 1}], expected)
    assert_identical(da[{'x': 1, 'y': 0}], expected)


def test_getitem_preserves_unit():
    da = dms.DimensionedArray(
        values=array.ones((2, 3)), dims=('x', 'y'), unit=Unit('m')
    )
    assert da[{'x': 0}].unit == Unit('m')


def test_setitem_raises_if_value_has_extra_dims():
    da = dms.DimensionedArray(values=array.ones((2, 3)), dims=('x', 'y'), unit=None)
    with pytest.raises(dms.DimensionError, match="Value has extra dimensions"):
        da[{'x': 0}] = dms.DimensionedArray(
            values=array.zeros((2, 3, 4)), dims=('x', 'y', 'z'), unit=None
        )


def test_setitem_transposes_automatically():
    da = dms.DimensionedArray(values=array.ones((2, 3)), dims=('x', 'y'), unit=None)
    da[{'x': slice(None)}] = dms.DimensionedArray(
        values=array.reshape(array.arange(6), (3, 2)), dims=('y', 'x'), unit=None
    )
    assert_identical(
        da,
        dms.DimensionedArray(
            values=array.asarray([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]]),
            dims=('x', 'y'),
            unit=None,
        ),
    )


def test_setitem_raises_if_sizes_do_not_match():
    da = dms.DimensionedArray(values=array.ones((2, 3)), dims=('x', 'y'), unit=None)
    with pytest.raises(
        dms.DimensionError,
        match="Sizes of dimension 'x' do not match: 2 != 3",
    ):
        da[{'x': slice(None)}] = dms.DimensionedArray(
            values=array.ones((3, 3)), dims=('x', 'y'), unit=None
        )


def test_setitem_raises_if_units_differ():
    da1 = dms.DimensionedArray(
        values=array.ones((2, 3)), dims=('x', 'y'), unit=Unit('m')
    )
    da2 = dms.DimensionedArray(
        values=array.ones((2, 3)), dims=('x', 'y'), unit=Unit('s')
    )
    with pytest.raises(dms.UnitsError, match="Units must be identical"):
        da1[{'x': slice(None)}] = da2


def test_setitem_can_broadcast():
    da = dms.DimensionedArray(values=array.ones((2, 3)), dims=('x', 'y'), unit=None)
    da[{'x': 0}] = dms.DimensionedArray(values=array.zeros(()), dims=(), unit=None)
    assert_identical(
        da,
        dms.DimensionedArray(
            values=array.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            dims=('x', 'y'),
            unit=None,
        ),
    )


def test_setitem_ellipsis_broadcasts_to_everything():
    da = dms.DimensionedArray(values=array.zeros((2, 3)), dims=('x', 'y'), unit=None)
    da[...] = dms.DimensionedArray(values=array.ones(()), dims=(), unit=None)
    ones = dms.DimensionedArray(values=array.ones((2, 3)), dims=('x', 'y'), unit=None)
    assert_identical(da, ones)


def test_setitem_broadcasts_and_transposed_simultaneously():
    da = dms.DimensionedArray(
        values=array.ones((2, 3, 4)), dims=('x', 'y', 'z'), unit=None
    )
    da[{'x': slice(None)}] = dms.DimensionedArray(
        values=array.reshape(array.arange(8), (4, 2)), dims=('z', 'x'), unit=None
    )
    assert_identical(
        da,
        dms.DimensionedArray(
            dims=('x', 'y', 'z'),
            values=array.asarray(
                [
                    [[0.0, 2.0, 4.0, 6.0], [0.0, 2.0, 4.0, 6.0], [0.0, 2.0, 4.0, 6.0]],
                    [[1.0, 3.0, 5.0, 7.0], [1.0, 3.0, 5.0, 7.0], [1.0, 3.0, 5.0, 7.0]],
                ]
            ),
            unit=None,
        ),
    )


def test_setitem_does_not_broadcast_dims_of_size_1():
    da = dms.DimensionedArray(values=array.ones((2, 3)), dims=('x', 'y'), unit=None)
    with pytest.raises(
        dms.DimensionError,
        match="Sizes of dimension 'y' do not match: 3 != 1",
    ):
        da[{'x': 0}] = dms.DimensionedArray(
            values=array.ones((1,)), dims=('y'), unit=None
        )


@pytest.mark.parametrize('unit', [None, Unit(), Unit('m')])
def test_neg(unit: Unit | None):
    da = dms.DimensionedArray(values=array.ones((2, 3)), dims=('x', 'y'), unit=unit)
    result = -da
    assert_identical(
        result,
        dms.DimensionedArray(values=-array.ones((2, 3)), dims=('x', 'y'), unit=unit),
    )


def test_exp_raises_if_unit_is_not_dimensionless():
    da = dms.DimensionedArray(
        values=array.ones((2, 3)), dims=('x', 'y'), unit=Unit('m')
    )
    with pytest.raises(ValueError, match="Unit must be dimensionless"):
        dms.exp(da)


def test_add_raises_if_units_differ():
    da1 = dms.DimensionedArray(
        values=array.ones((2, 3)), dims=('x', 'y'), unit=Unit('m')
    )
    da2 = dms.DimensionedArray(
        values=array.ones((2, 3)), dims=('x', 'y'), unit=Unit('s')
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


def test_elemwise_binary_xyz_zx():
    xyz = dms.DimensionedArray(
        values=array.ones((2, 3, 4)), dims=('x', 'y', 'z'), unit=None
    )
    zx = dms.DimensionedArray(values=array.ones((4, 2)), dims=('z', 'x'), unit=None)
    result = dms.common.elemwise_binary(
        xyz, zx, values_op=lambda a, b: a + b, unit_op=lambda a, b: a
    )
    assert result.sizes == {'x': 2, 'y': 3, 'z': 4}


def test_elemwise_binary_raises_if_dim_of_size_1_would_need_broadcasting():
    x = dms.DimensionedArray(values=array.ones((2, 1)), dims=('x', 'y'), unit=None)
    y = dms.DimensionedArray(values=array.ones((2, 3)), dims=('x', 'y'), unit=None)
    with pytest.raises(
        dms.DimensionError,
        match="Sizes of dimension 'y' do not match: 1 != 3",
    ):
        dms.common.elemwise_binary(
            x, y, values_op=lambda a, b: a + b, unit_op=lambda a, b: a
        )
