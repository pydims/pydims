# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
"""
Reduction functions.

Includes functions from the "Utility Functions" and "Statistical Functions" section
of the Python Array API standard.
"""

from collections.abc import Callable
from typing import Any

from .dimensioned_array import (
    ArrayImplementation,
    Dim,
    DimArr,
    Dims,
    UnitImplementation,
)


def _reduce(
    x: DimArr,
    /,
    *,
    dim: Dim | Dims | None = None,
    values_op: Callable[[ArrayImplementation], ArrayImplementation],
    unit_op: Callable[[UnitImplementation | None], UnitImplementation | None],
    **kwargs: Any,
) -> DimArr:
    if 'keepdims' in kwargs:
        raise ValueError("keepdims is not supported")
    axis, dims = _axis_dims_for_reduce(x, dim)
    return x.__class__(
        values=values_op(x.values, axis=axis, **kwargs), dims=dims, unit=unit_op(x.unit)
    )


def _axis_dims_for_reduce(x, dim):
    dim = dim or x.dims
    if not isinstance(dim, tuple):
        dim = (dim,)

    axis = tuple(x.dims.index(d) for d in dim)
    dims = tuple(d for d in x.dims if d not in dim)
    return axis, dims


def _reduce_docstring(short: str, op: str) -> str:
    return f"""
    {short} along a dimension.

    Parameters
    ----------
    x:
        Input array
    dim:
        Dimension or dimensions along which to perform {op}.

    Returns
    -------
    :
        Result array
    """


def _unit_must_be_none(unit: UnitImplementation) -> None:
    if unit is not None:
        # TODO Is this what we want? Is there any harm in allowing a unit?
        raise ValueError("Unit is not supported for logical operation")


def all(x: DimArr, /, *, dim: Dim | Dims | None = None) -> DimArr:
    return _reduce(
        x,
        dim=dim,
        values_op=x.array_namespace.all,
        unit_op=_unit_must_be_none,
    )


def any(x: DimArr, /, *, dim: Dim | Dims | None = None) -> DimArr:
    return _reduce(
        x,
        dim=dim,
        values_op=x.array_namespace.any,
        unit_op=_unit_must_be_none,
    )


all.__doc__ = _reduce_docstring(
    short="Test whether all elements are true", op="a logical AND reduction"
)
any.__doc__ = _reduce_docstring(
    short="Test whether any elements are true", op="a logical OR reduction"
)


__all__ = ['all', 'any']
