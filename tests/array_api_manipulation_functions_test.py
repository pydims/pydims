# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
import numpy as np
import pytest

import dims as dms
from dims.testing import assert_identical


@pytest.mark.parametrize('dims', [(), ('x',), ('x', 'y', 'z')])
def test_init_raises_if_dims_has_wrong_length(dims: tuple[str, ...]):
    with pytest.raises(ValueError, match="Number of dimensions"):
        dms.DimensionedArray(values=np.ones((2, 3)), dims=dims, unit=None)


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
