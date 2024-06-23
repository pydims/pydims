# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
from __future__ import annotations

from collections.abc import Callable

from .dimensioned_array import (
    ArrayImplementation,
    DimArr,
    DimensionedArray,
    DimensionError,
    Dims,
    UnitImplementation,
)

_pretty_project = "PyDims"


def _merge_dims(a: Dims, b: Dims) -> Dims:
    """Favor order in a."""
    # TODO Avoid transpose of b if possible
    return a + tuple(dim for dim in b if dim not in a)


def broadcast_and_transpose_values(
    *, array: DimensionedArray, dims: Dims
) -> ArrayImplementation:
    values = array.values
    for dim in dims:
        if dim not in array.dims:
            values = array.array_namespace.expand_dims(values, axis=0)
    new_dims = (*(set(dims) - set(array.dims)), *array.dims)
    axes = tuple(new_dims.index(dim) for dim in dims)
    return array.array_namespace.permute_dims(values, axes=axes)


def check_compatible_dims_and_shape(x: DimensionedArray, y: DimensionedArray) -> None:
    shared = set(x.dims) & set(y.dims)
    for dim in shared:
        if x.sizes[dim] != y.sizes[dim]:
            msg = (
                f"Sizes of dimension '{dim}' do not match: "
                f"{x.sizes[dim]} != {y.sizes[dim]}."
            )
            if x.sizes[dim] == 1 or y.sizes[dim] == 1:
                msg += f" Note: {_pretty_project} never broadcasts dims of size 1."
            raise DimensionError(msg)


# TODO make this a method, DimensionedArray.transform?
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
    check_compatible_dims_and_shape(x, y)
    dims = _merge_dims(x.dims, y.dims)
    # TODO What if y.__class__ != x.__class__?
    return x.__class__(
        values=values_op(
            broadcast_and_transpose_values(array=x, dims=dims),
            broadcast_and_transpose_values(array=y, dims=dims),
        ),
        dims=dims,
        unit=(
            None
            # TODO do not mix unit with None
            if x.unit is None and y.unit is None
            else unit_op(x.unit, y.unit)
        ),
    )
