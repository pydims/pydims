# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
"""
Indexing functions.

Includes functions from the "Indexing Functions" section of the Python Array API
standard.
"""

from .dimensioned_array import DimArr, DimensionedArray, DimensionError


def take(x: DimArr, /, indices: DimensionedArray) -> DimArr:
    """
    Returns elements of an array along an axis.

    The indices must be 1-D and their single dimension defines the axis along which
    to take elements.

    Parameters
    ----------
    x:
        Input array.
    indices:
        Array of indices to extract from the input array. Must be 1-D.

    Returns
    -------
    :
        Array containing the elements of the input array at the specified indices.
    """
    try:
        axis = x.dims.index(indices.dim)
    except ValueError:
        raise DimensionError(
            f"Indices dimension '{indices.dim}' not in data dimensions '{x.dims}'"
        ) from None
    return x.__class__(
        values=x.values.take(indices.values, axis=axis),
        dims=x.dims,
        unit=x.unit,
    )


__all__ = ['take']
