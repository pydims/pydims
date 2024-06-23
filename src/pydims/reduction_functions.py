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
    DType,
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
    {short} along one or multiple dimensions.

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


def _unit_must_be_none(unit: UnitImplementation | None) -> None:
    if unit is not None:
        # TODO Is this what we want? Is there any harm in allowing a unit?
        raise ValueError("Unit is not supported for logical operation")


def _keep_unit(unit: UnitImplementation | None) -> UnitImplementation | None:
    return unit


def _unit_must_be_idempotent(
    unit: UnitImplementation | None,
) -> UnitImplementation | None:
    if unit is None:
        return None
    if unit * unit != unit:
        raise ValueError("Unit must be idempotent")
    return unit


def all(x: DimArr, /, *, dim: Dim | Dims | None = None, **kwargs: Any) -> DimArr:
    return _reduce(
        x,
        dim=dim,
        values_op=x.array_namespace.all,
        unit_op=_unit_must_be_none,
        **kwargs,
    )


def any(x: DimArr, /, *, dim: Dim | Dims | None = None, **kwargs: Any) -> DimArr:
    return _reduce(
        x,
        dim=dim,
        values_op=x.array_namespace.any,
        unit_op=_unit_must_be_none,
        **kwargs,
    )


def max(x: DimArr, /, *, dim: Dim | Dims | None = None, **kwargs: Any) -> DimArr:
    return _reduce(
        x,
        dim=dim,
        values_op=x.array_namespace.max,
        unit_op=_keep_unit,
        **kwargs,
    )


def min(x: DimArr, /, *, dim: Dim | Dims | None = None, **kwargs: Any) -> DimArr:
    return _reduce(
        x,
        dim=dim,
        values_op=x.array_namespace.min,
        unit_op=_keep_unit,
        **kwargs,
    )


def sum(
    x: DimArr,
    /,
    *,
    dim: Dim | Dims | None = None,
    dtype: DType | None = None,
    **kwargs: Any,
) -> DimArr:
    return _reduce(
        x,
        dim=dim,
        values_op=x.array_namespace.sum,
        unit_op=_keep_unit,
        dtype=dtype,
        **kwargs,
    )


def mean(x: DimArr, /, *, dim: Dim | Dims | None = None, **kwargs: Any) -> DimArr:
    return _reduce(
        x,
        dim=dim,
        values_op=x.array_namespace.mean,
        unit_op=_keep_unit,
        **kwargs,
    )


def prod(
    x: DimArr,
    /,
    *,
    dim: Dim | Dims | None = None,
    dtype: DType | None = None,
    **kwargs: Any,
) -> DimArr:
    return _reduce(
        x,
        dim=dim,
        values_op=x.array_namespace.prod,
        unit_op=_unit_must_be_idempotent,
        dtype=dtype,
        **kwargs,
    )


def std(
    x: DimArr,
    /,
    *,
    dim: Dim | Dims | None = None,
    correction: float = 0,
    **kwargs: Any,
) -> DimArr:
    return _reduce(
        x,
        dim=dim,
        values_op=x.array_namespace.std,
        unit_op=_keep_unit,
        correction=correction,
        **kwargs,
    )


def var(
    x: DimArr,
    /,
    *,
    dim: Dim | Dims | None = None,
    correction: float = 0,
    **kwargs: Any,
) -> DimArr:
    return _reduce(
        x,
        dim=dim,
        values_op=x.array_namespace.var,
        unit_op=lambda unit: None if unit is None else unit * unit,
        correction=correction,
        **kwargs,
    )


all.__doc__ = _reduce_docstring(
    short="Test whether all elements are true", op="a logical AND reduction"
)
any.__doc__ = _reduce_docstring(
    short="Test whether any elements are true", op="a logical OR reduction"
)
max.__doc__ = _reduce_docstring(short="Maximum value", op="a maximum reduction")
mean.__doc__ = _reduce_docstring(short="Mean", op="a mean reduction")
min.__doc__ = _reduce_docstring(short="Minimum value", op="a minimum reduction")
prod.__doc__ = _reduce_docstring(short="Product", op="a product reduction")
std.__doc__ = _reduce_docstring(
    short="Standard deviation", op="a standard deviation reduction"
)
sum.__doc__ = _reduce_docstring(short="Sum", op="a sum reduction")
var.__doc__ = _reduce_docstring(short="Variance", op="a variance reduction")

__all__ = ['all', 'any', 'max', 'min', 'sum', 'mean', 'prod', 'std', 'var']
