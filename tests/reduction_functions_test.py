# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
import numpy as np
import pytest

import pydims as dms
from pydims.testing import assert_identical

make = dms.CreationFunctions(array=np, units=dms.string_units)


@pytest.mark.parametrize(
    'func',
    [dms.all, dms.any, dms.max, dms.min, dms.sum, dms.mean, dms.std, dms.var, dms.prod],
)
def reduction_raises_if_given_keepdims_argument(func):
    da = make.asarray(dims=('x',), values=[1, 2, 3], unit=None)
    with pytest.raises(ValueError, match="keepdims is not supported"):
        _ = func(da, dim='x', keepdims=True)


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


def test_all_raises_if_unit_is_not_none():
    da = make.asarray(dims=('x',), values=[True, True, False], unit='m')
    with pytest.raises(ValueError, match="Unit is not supported for logical operation"):
        _ = dms.all(da, dim='x')


def test_any_raises_if_unit_is_not_none():
    da = make.asarray(dims=('x',), values=[True, True, False], unit='m')
    with pytest.raises(ValueError, match="Unit is not supported for logical operation"):
        _ = dms.any(da, dim='x')


@pytest.mark.parametrize('unit', [None, 'm'])
def test_max_returns_input_unit(unit: None | str):
    da = make.asarray(dims=('x',), values=[1, 2, 3], unit=unit)
    assert dms.max(da).unit == da.unit


@pytest.mark.parametrize('unit', [None, 'm'])
def test_min_returns_input_unit(unit: None | str):
    da = make.asarray(dims=('x',), values=[1, 2, 3], unit=unit)
    assert dms.min(da).unit == da.unit


@pytest.mark.parametrize('unit', [None, 'm'])
def test_sum_returns_input_unit(unit: None | str):
    da = make.asarray(dims=('x',), values=[1, 2, 3], unit=unit)
    assert dms.sum(da).unit == da.unit


@pytest.mark.parametrize('unit', [None, 'm'])
def test_mean_returns_input_unit(unit: None | str):
    da = make.asarray(dims=('x',), values=[1, 2, 3], unit=unit)
    assert dms.mean(da).unit == da.unit


@pytest.mark.parametrize('unit', [None, 'm'])
def test_std_returns_input_unit(unit: None | str):
    da = make.asarray(dims=('x',), values=[1, 2, 3], unit=unit)
    assert dms.std(da).unit == da.unit


@pytest.mark.parametrize('unit', [None, ''])
def test_prod_works_with_none_or_dimensionless_unit(unit):
    da = make.asarray(dims=('x',), values=[1, 2, 3], unit=unit)
    assert dms.prod(da).unit == da.unit


def test_var_works_with_no_unit():
    da = make.asarray(dims=('x',), values=[1, 2, 3], unit=None)
    assert dms.var(da).unit is None


def test_var_squares_unit():
    da = make.asarray(dims=('x',), values=[1, 2, 3], unit='m')
    assert dms.var(da).unit == da.unit * da.unit
