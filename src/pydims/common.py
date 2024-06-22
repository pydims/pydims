# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
from __future__ import annotations

from collections.abc import Callable

from .dimensioned_array import (
    ArrayImplementation,
    DimArr,
    Dims,
    UnitImplementation,
)


def _merge_dims(a: Dims, b: Dims) -> Dims:
    """Favor order in a."""
    # TODO Avoid transpose of b if possible
    return a + tuple(dim for dim in b if dim not in a)


def unary(
    x: DimArr,
    values_op: Callable[[ArrayImplementation], ArrayImplementation],
    unit_op: Callable[[UnitImplementation], UnitImplementation],
) -> DimArr:
    return x.__class__(
        values=values_op(x.values),
        dims=x.dims,
        unit=None if x.unit is None else unit_op(x.unit),
    )


def elemwise_binary(
    x: DimArr,
    /,
    y: DimArr,
    *,
    values_op: Callable[
        [ArrayImplementation, ArrayImplementation], ArrayImplementation
    ],
    unit_op: Callable[[UnitImplementation, UnitImplementation], UnitImplementation],
) -> DimArr:
    dims = _merge_dims(x.dims, y.dims)
    a = x.values
    b = y.values
    for dim in dims:
        if dim not in x.dims:
            a = x.array_api.expand_dims(a, axis=dims.index(dim))
        if dim not in y.dims:
            b = y.array_api.expand_dims(b, axis=0)
    b_dims = (*(set(dims) - set(y.dims)), *y.dims)
    b = y.array_api.permute_dims(b, axes=tuple(b_dims.index(dim) for dim in dims))
    # TODO What if y.__class__ != x.__class__?
    return x.__class__(
        values=values_op(a, b),
        dims=dims,
        unit=(
            None
            # TODO do not mix unit with None
            if x.unit is None and y.unit is None
            else unit_op(x.unit, y.unit)
        ),
    )
