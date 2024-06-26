# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
"""
Indexing functions.

Includes functions from the "Indexing Functions" section of the Python Array API
standard.
"""

from .dimensioned_array import DimArr, DimensionedArray, DimensionError


def take(x: DimArr, /, indices: DimensionedArray) -> DimArr:
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
