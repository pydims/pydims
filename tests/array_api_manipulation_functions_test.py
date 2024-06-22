# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
import numpy as np
import pytest

import pydims as dms
from pydims.testing import assert_identical


def test_concat():
    da = dms.DimensionedArray(values=np.ones((2, 3)), dims=('x', 'y'), unit=None)
    assert_identical(
        dms.concat((da, da), dim='x'),
        dms.DimensionedArray(values=np.ones((4, 3)), dims=('x', 'y'), unit=None),
    )
    assert_identical(
        dms.concat((da, da), dim='y'),
        dms.DimensionedArray(values=np.ones((2, 6)), dims=('x', 'y'), unit=None),
    )


@pytest.mark.parametrize(
    'dims',
    [
        ('y', 'x'),
        ('x', 'z'),
        ('z', 'x'),
        ('y', 'z', 'x'),
        ('z', 'y', 'x'),
        ('z', 'x', 'y'),
        ('x', 'z', 'y'),
    ],
)
def test_flatten_raises_if_dims_not_contiguous_and_ordered(dims: tuple[str, ...]):
    da = dms.DimensionedArray(
        values=np.arange(6).reshape((2, 3, 1)), dims=('x', 'y', 'z'), unit=None
    )
    with pytest.raises(ValueError, match="Dimensions must be contiguous and ordered"):
        dms.flatten(da, dims=dims)


def test_flatten_raises_if_dims_not_in_dims():
    da = dms.DimensionedArray(
        values=np.arange(6).reshape((2, 3)), dims=('x', 'y'), unit=None
    )
    with pytest.raises(ValueError, match="All dims must be in the array"):
        dms.flatten(da, dims=('x', 'z'))


def test_flatten_raises_if_output_dim_in_preserved_dims():
    da = dms.DimensionedArray(
        values=np.arange(6).reshape((2, 3)), dims=('x', 'y'), unit=None
    )
    with pytest.raises(ValueError, match="Output dim must not be in preserved dims"):
        dms.flatten(da, dims=('x',), dim='y')


def test_flatten_does_not_raise_if_output_dim_in_flattened_dims():
    da = dms.DimensionedArray(
        values=np.arange(6).reshape((2, 3)), dims=('x', 'y'), unit=None
    )
    dms.flatten(da, dims=('x',), dim='x')
    dms.flatten(da, dims=('x', 'y'), dim='x')


def test_flatten_raises_if_automatic_output_dims_clashes_with_preserved_dim():
    da = dms.DimensionedArray(
        values=np.arange(6).reshape((2, 3, 1)), dims=('x', 'y', 'x_y'), unit=None
    )
    with pytest.raises(ValueError, match="Output dim must not be in preserved dims"):
        dms.flatten(da, dims=('x', 'y'), dim='x_y')


def test_flatten_all():
    da = dms.DimensionedArray(
        values=np.arange(6).reshape((2, 3)), dims=('x', 'y'), unit=None
    )
    assert_identical(
        dms.flatten(da),
        dms.DimensionedArray(values=np.arange(6), dims=('x_y',), unit=None),
    )


def test_flatten_all_with_dim():
    da = dms.DimensionedArray(
        values=np.arange(6).reshape((2, 3)), dims=('x', 'y'), unit=None
    )
    assert_identical(
        dms.flatten(da, dim='z'),
        dms.DimensionedArray(values=np.arange(6), dims=('z',), unit=None),
    )


def test_flatten_first_renames():
    da = dms.DimensionedArray(
        values=np.arange(6).reshape((2, 3)), dims=('x', 'y'), unit=None
    )
    assert_identical(
        dms.flatten(da, dims=('x',), dim='z'),
        dms.DimensionedArray(
            values=np.arange(6).reshape((2, 3)), dims=('z', 'y'), unit=None
        ),
    )


def test_flatten_first_two():
    da = dms.DimensionedArray(
        values=np.arange(24).reshape((2, 3, 4)), dims=('x', 'y', 'z'), unit=None
    )
    assert_identical(
        dms.flatten(da, dims=('x', 'y'), dim='xy'),
        dms.DimensionedArray(
            values=np.arange(24).reshape((6, 4)), dims=('xy', 'z'), unit=None
        ),
    )


def test_flatten_4d_middle_two():
    da = dms.DimensionedArray(
        values=np.arange(24).reshape((2, 3, 2, 2)), dims=('x', 'y', 'z', 'w'), unit=None
    )
    assert_identical(
        dms.flatten(da, dims=('y', 'z'), dim='yz'),
        dms.DimensionedArray(
            values=np.arange(24).reshape((2, 6, 2)), dims=('x', 'yz', 'w'), unit=None
        ),
    )


def test_flatten_last_two():
    da = dms.DimensionedArray(
        values=np.arange(24).reshape((2, 3, 4)), dims=('x', 'y', 'z'), unit=None
    )
    assert_identical(
        dms.flatten(da, dims=('y', 'z'), dim='yz'),
        dms.DimensionedArray(
            values=np.arange(24).reshape((2, 12)), dims=('x', 'yz'), unit=None
        ),
    )


def test_fold_raises_if_dim_not_in_dims():
    da = dms.DimensionedArray(values=np.ones((2, 3)), dims=('x', 'y'), unit=None)
    with pytest.raises(ValueError, match="Dimension not found"):
        dms.fold(da, dim='z', sizes={'z1': 2, 'z2': 3})


def test_fold_raises_if_output_shape_is_not_consistent_with_dim_size():
    da = dms.DimensionedArray(values=np.ones((2, 3)), dims=('x', 'y'), unit=None)
    with pytest.raises(ValueError, match="cannot reshape array of size 6 into shape"):
        dms.fold(da, dim='x', sizes={'x1': 2, 'x2': 4})


def test_fold_raises_if_output_dim_in_preserved_dims():
    da = dms.DimensionedArray(values=np.ones((4, 3)), dims=('x', 'y'), unit=None)
    with pytest.raises(ValueError, match="Duplicate dimensions"):
        dms.fold(da, dim='x', sizes={'x': 2, 'y': 2})


def test_fold_first():
    da = dms.DimensionedArray(values=np.ones((6, 3)), dims=('x', 'y'), unit=None)
    assert_identical(
        dms.fold(da, dim='x', sizes={'x1': 2, 'x2': 3}),
        dms.DimensionedArray(
            values=np.ones((2, 3, 3)), dims=('x1', 'x2', 'y'), unit=None
        ),
    )


def test_fold_middle():
    da = dms.DimensionedArray(
        values=np.ones((2, 6, 3)), dims=('x', 'y', 'z'), unit=None
    )
    assert_identical(
        dms.fold(da, dim='y', sizes={'y1': 2, 'y2': 3}),
        dms.DimensionedArray(
            values=np.ones((2, 2, 3, 3)), dims=('x', 'y1', 'y2', 'z'), unit=None
        ),
    )


def test_fold_last():
    da = dms.DimensionedArray(
        values=np.ones((2, 3, 6)), dims=('x', 'y', 'z'), unit=None
    )
    assert_identical(
        dms.fold(da, dim='z', sizes={'z1': 2, 'z2': 3}),
        dms.DimensionedArray(
            values=np.ones((2, 3, 2, 3)), dims=('x', 'y', 'z1', 'z2'), unit=None
        ),
    )


def test_reshape_raises_NotImplementedError():
    with pytest.raises(NotImplementedError, match="`reshape` is not supported"):
        dms.reshape()


def test_stack():
    da = dms.DimensionedArray(values=np.ones((4, 3)), dims=('x', 'y'), unit=None)
    assert_identical(
        dms.stack((da, da), dim='z'),
        dms.DimensionedArray(
            values=np.ones((2, 4, 3)), dims=('z', 'x', 'y'), unit=None
        ),
    )
    assert_identical(
        dms.stack((da, da), dim='z', axis=0),
        dms.DimensionedArray(
            values=np.ones((2, 4, 3)), dims=('z', 'x', 'y'), unit=None
        ),
    )
    assert_identical(
        dms.stack((da, da), dim='z', axis=1),
        dms.DimensionedArray(
            values=np.ones((4, 2, 3)), dims=('x', 'z', 'y'), unit=None
        ),
    )
    assert_identical(
        dms.stack((da, da), dim='z', axis=2),
        dms.DimensionedArray(
            values=np.ones((4, 3, 2)), dims=('x', 'y', 'z'), unit=None
        ),
    )
    assert_identical(
        dms.stack((da, da), dim='z', axis=-1),
        dms.DimensionedArray(
            values=np.ones((4, 3, 2)), dims=('x', 'y', 'z'), unit=None
        ),
    )
    assert_identical(
        dms.stack((da, da), dim='z', axis=-2),
        dms.DimensionedArray(
            values=np.ones((4, 2, 3)), dims=('x', 'z', 'y'), unit=None
        ),
    )
    assert_identical(
        dms.stack((da, da), dim='z', axis=-3),
        dms.DimensionedArray(
            values=np.ones((2, 4, 3)), dims=('z', 'x', 'y'), unit=None
        ),
    )
