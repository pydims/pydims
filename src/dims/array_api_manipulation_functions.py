# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
"""
Functions from the "Manipulation Functions" section of the Python Array API.

Will not support:

- broadcast_arrays
- broadcast_to
- reshape -> use fold and flatten instead
"""

import array_api_compat

from .dimensioned_array import Dim, DimensionedArray


def concat(
    arrays: tuple[DimensionedArray, ...], /, *, dim: Dim | None = None
) -> DimensionedArray:
    """
    Concatenate arrays along a given dimension.

    Parameters
    ----------
    arrays:
        Arrays to concatenate.
    dim:
        Dimension along which to concatenate. If None, arrays must be 1-D.

    Returns
    -------
    :
        Concatenated array.
    """
    first = arrays[0]
    dim = dim or first.dim
    if not all(arr.dims == first.dims for arr in arrays):
        raise ValueError("All arrays must have the same dims")
    if not all(arr.unit == first.unit for arr in arrays):
        raise ValueError("All arrays must have the same unit")
    axis = first.dims.index(dim)
    values = [arr.values for arr in arrays]
    xp = array_api_compat.array_namespace(*values)
    return DimensionedArray(
        values=xp.concat(values, axis=axis), dims=first.dims, unit=first.unit
    )


def reshape(*args, **kwargs):
    raise NotImplementedError(
        "`reshape` is deliberately not supported `fold` and `flatten` instead"
    )


def stack(
    arrays: tuple[DimensionedArray, ...] | list[DimensionedArray],
    /,
    *,
    dim: Dim,
    axis: int = 0,
) -> DimensionedArray:
    """
    Stack arrays along a new dimension.

    Parameters
    ----------
    arrays:
        Arrays to stack.
    dim:
        Dimension along which to stack.
    axis:
        Location of the new dimension.

    Returns
    -------
    :
        Stacked array.
    """
    first = arrays[0]
    if dim in first.dims:
        raise ValueError("Dimension already exists, did you mean to use `concat`?")
    if not all(arr.dims == first.dims for arr in arrays):
        raise ValueError("All arrays must have the same dims")
    if not all(arr.unit == first.unit for arr in arrays):
        raise ValueError("All arrays must have the same unit")
    dims = list(first.dims)
    dims.insert(axis if axis >= 0 else first.ndim + 1 + axis, dim)
    values = [arr.values for arr in arrays]
    xp = array_api_compat.array_namespace(*values)
    return DimensionedArray(
        values=xp.stack(values, axis=axis), dims=dims, unit=first.unit
    )


__all__ = ['concat', 'reshape', 'stack']
