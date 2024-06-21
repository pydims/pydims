# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
"""
Functions from the "Utility Functions" section of the Python Array API.

"""

from .dimensioned_array import Dim, DimensionedArray, Dims


def all(x: DimensionedArray, /, *, dim: Dim | Dims | None = None) -> DimensionedArray:
    """
    Tests whether all input array elements evaluate to `True` along a given dimension.

    Parameters
    ----------
    x:
        Input array
    dim:
        Dimension or dimensions along which to perform a logical AND reduction.

    Returns
    -------
    :
        Result array
    """
    if x.unit is not None:
        # TODO Is this what we want? Is there any harm in allowing a unit?
        raise ValueError("Unit is not supported for logical operation")
    axis, dims = _axis_dims_for_reduce(x, dim)
    # boolean output has unit=None, input unit is discarded
    return DimensionedArray(
        values=x.array_api.all(x.values, axis=axis), dims=dims, unit=None
    )


def any(x: DimensionedArray, /, *, dim: Dim | Dims | None = None) -> DimensionedArray:
    """
    Tests whether any input array elements evaluate to `True` along a given dimension.

    Parameters
    ----------
    x:
        Input array
    dim:
        Dimension or dimensions along which to perform a logical OR reduction.

    Returns
    -------
    :
        Result array
    """
    if x.unit is not None:
        # TODO Is this what we want? Is there any harm in allowing a unit?
        raise ValueError("Unit is not supported for logical operation")
    axis, dims = _axis_dims_for_reduce(x, dim)
    # boolean output has unit=None, input unit is discarded
    return DimensionedArray(
        values=x.array_api.any(x.values, axis=axis), dims=dims, unit=None
    )


def _axis_dims_for_reduce(x, dim):
    dim = dim or x.dims
    if not isinstance(dim, tuple):
        dim = (dim,)

    axis = tuple(x.dims.index(d) for d in dim)
    dims = tuple(d for d in x.dims if d not in dim)
    return axis, dims


__all__ = ['all', 'any']
