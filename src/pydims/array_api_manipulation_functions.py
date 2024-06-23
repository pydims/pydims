# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
"""
Functions from the "Manipulation Functions" section of the Python Array API.

Will not support:

- broadcast_arrays
- broadcast_to
- reshape -> use fold and flatten instead
"""

from collections.abc import Mapping
from math import prod

import array_api_compat

from .dimensioned_array import Dim, DimArr, Dims


def concat(arrays: tuple[DimArr, ...], /, *, dim: Dim | None = None) -> DimArr:
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
    return first.__class__(
        values=xp.concat(values, axis=axis), dims=first.dims, unit=first.unit
    )


def flatten(
    array: DimArr,
    /,
    *,
    dims: tuple[Dim, ...] | None = None,
    dim: Dim | None = None,
) -> DimArr:
    """
    Flatten a set of dimensions into a single dimension.

    Parameters
    ----------
    array:
        Array to flatten.
    dims:
        Dimensions to flatten.
    dim:
        Name of the new dimension.

    Returns
    -------
    :
        Flattened array.
    """
    dims = dims or array.dims
    if not all(d in array.dims for d in dims):
        raise ValueError("All dims must be in the array")
    dim = dim or "_".join(dims)
    axes = [array.dims.index(dim) for dim in dims]
    if axes != list(range(min(axes), max(axes) + 1)):
        raise ValueError("Dimensions must be contiguous and ordered")
    shape = list(array.shape)
    shape[min(axes) : max(axes) + 1] = [prod(shape[min(axes) : max(axes) + 1])]
    new_dims = list(array.dims)
    new_dims[min(axes) : max(axes) + 1] = []
    if dim in new_dims:
        raise ValueError("Output dim must not be in preserved dims")
    new_dims[min(axes) : min(axes)] = [dim]
    values = array.array_api.reshape(array.values, shape)
    return array.__class__(values=values, dims=new_dims, unit=array.unit)


def fold(array: DimArr, /, dim: Dim, *, sizes: Mapping[Dim, int]) -> DimArr:
    """
    Fold a dimension of an array into a new set of dimensions.

    Parameters
    ----------
    array:
        Array to fold.
    dim:
        Dimension to fold.
    sizes:
        Sizes of the dimensions after folding.

    Returns
    -------
    :
        Folded array.
    """
    if dim not in array.dims:
        raise ValueError("Dimension not found")
    shape = list(array.shape)
    axis = array.dims.index(dim)
    shape[axis : axis + 1] = sizes.values()
    dims = list(array.dims)
    dims[axis : axis + 1] = sizes.keys()
    if len(dims) != len(set(dims)):
        raise ValueError("Duplicate dimensions")
    values = array.array_api.reshape(array.values, shape)
    return array.__class__(values=values, dims=dims, unit=array.unit)


def permute_dims(array: DimArr, /, dims: Dims) -> DimArr:
    """
    Permute the dimensions of an array.

    Parameters
    ----------
    array:
        Array to permute.
    dims:
        New order of dimensions.

    Returns
    -------
    :
        Permuted array.
    """
    if set(dims) != set(array.dims):
        raise ValueError("New dims must contain all old dims")
    axes = [array.dims.index(dim) for dim in dims]
    values = array.array_api.permute_dims(array.values, axes=axes)
    return array.__class__(values=values, dims=dims, unit=array.unit)


def reshape(*args, **kwargs):
    raise NotImplementedError(
        "`reshape` is not supported, use `fold` and `flatten` instead."
        "Reason: `reshape` is error-prone since it relies on a particular axis order."
    )


def stack(
    arrays: tuple[DimArr, ...] | list[DimArr],
    /,
    *,
    dim: Dim,
    axis: int = 0,
) -> DimArr:
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
    return first.__class__(
        values=xp.stack(values, axis=axis), dims=dims, unit=first.unit
    )


__all__ = ['concat', 'reshape', 'stack']
