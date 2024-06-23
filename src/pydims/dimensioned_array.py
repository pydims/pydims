# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
from __future__ import annotations

import operator
from collections.abc import Hashable, Iterator, Mapping
from types import EllipsisType
from typing import Any, Protocol, TypeVar

import array_api_compat

from . import units_api_compat

DType = Any  # Is the array API standard defining a DType type?


class DimensionError(Exception):
    pass


class UnitsError(Exception):
    pass


class ArrayImplementation(Protocol):
    """Array of values following the Python array API standard."""


class UnitImplementation(Protocol):
    pass


# Tuple is hashable so this leads to some ambiguity, but if in doubt a tuple is
# interpreted as a tuple of dimensions, not a single dimension.
Dim = Hashable
Dims = tuple[Dim, ...]
Shape = tuple[int, ...]


class Sizes(Mapping):
    def __init__(self, dims: Dims, shape: Shape):
        self._data = dict(zip(dims, shape, strict=True))

    def __getitem__(self, key) -> int:
        return self._data[key]

    def __iter__(self) -> Iterator[Dim]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


DimArr = TypeVar('DimArr', bound='DimensionedArray')


class DimensionedArray:
    """
    Array with named dimensions and optional unit.
    """

    def __init__(
        self,
        *,
        dims: Dims,
        values: ArrayImplementation,
        unit: UnitImplementation | None,
    ):
        """
        Parameters
        ----------
        dims:
            Dimension names.
        values:
            Array of values.
        unit:
            Optional unit.
        """
        if len(dims) != values.ndim:
            raise ValueError(
                f"Number of dimensions ({values.ndim}) does "
                f"not match number of dims ({len(dims)})"
            )
        if len(set(dims)) != len(dims):
            raise ValueError(f"Dimensions must be unique, got {dims}")
        self._values = values
        self._dims = tuple(dims)
        self._unit = unit

    def __str__(self) -> str:
        return (
            f"dims={self.dims}\nshape={self.shape}\n"
            f"values={self.values}\nunit={self.unit}"
        )

    def _repr_html_(self) -> str:
        dims_str = ", ".join(f"{dim}: {size}" for dim, size in self.sizes.items())
        return (
            f"<table><tr>"
            f"<td>DimensionedArray ({dims_str}):</td>"
            f"<td>[{self.unit}]</td>"
            f"<td>{self.values}</td>"
            f"</tr></table>"
        )

    @property
    def array_namespace(self) -> Any:
        return array_api_compat.array_namespace(self.values)

    @property
    def units_namespace(self) -> Any:
        return units_api_compat.units_namespace(self.unit)

    @property
    def dtype(self) -> DType:
        return self._values.dtype

    @property
    def ndim(self) -> int:
        return self._values.ndim

    @property
    def size(self) -> int:
        return self._values.size

    @property
    def dims(self) -> Dims:
        return self._dims

    @property
    def dim(self) -> Dim:
        if len(self.dims) != 1:
            raise DimensionError("Number of dimensions must be 1")
        return self.dims[0]

    @property
    def unit(self) -> UnitImplementation | None:
        return self._unit

    @property
    def shape(self) -> Shape:
        return tuple(self._values.shape)

    @property
    def sizes(self) -> Sizes:
        return Sizes(dims=self.dims, shape=self.shape)

    @property
    def values(self) -> ArrayImplementation:
        return self._values

    def astype(self: DimArr, dtype: DType, copy: bool = True) -> DimArr:
        return self.__class__(
            values=self.array_namespace.astype(self.values, dtype, copy=copy),
            dims=self.dims,
            unit=self.unit,
        )

    def _to_unit(self: DimArr, unit: Any, copy: bool = True) -> DimArr:
        scale = self.units_namespace.get_scale(src=self.unit, dst=unit)
        if scale == 1 and not copy:
            return self
        return self.__class__(
            values=self.values * scale,
            dims=self.dims,
            unit=self.units_namespace.Unit(unit),
        )

    def to(
        self: DimArr,
        *,
        dtype: DType | None = None,
        unit: Any | None = None,
        copy: bool = True,
    ) -> DimArr:
        """
        Convert to a new dtype and/or unit.

        Parameters
        ----------
        dtype:
            New dtype, None if no conversion is needed.
        unit:
            New unit, None if no conversion is needed.
        copy:
            If True, a copy of the values is made, even if no conversion is needed.

        Returns
        -------
        :
            New array with the requested dtype and/or unit.
        """
        if unit is None and dtype is None:
            raise ValueError("Must provide dtype or unit or both")

        if dtype is None:
            return self._to_unit(unit, copy=copy)

        if unit is None:
            return self.astype(dtype, copy=copy)

        # TODO Logic copied from Scipp. Probably not complete with the Array API dtypes
        api = self.array_namespace
        if dtype == api.float64:
            convert_dtype_first = True
        elif self.dtype == api.float64:
            convert_dtype_first = False
        elif dtype == api.float32:
            convert_dtype_first = True
        elif self.dtype == api.float32:
            convert_dtype_first = False
        elif self.dtype == api.int64 and dtype == api.int32:
            convert_dtype_first = False
        elif self.dtype == api.int32 and dtype == api.int64:
            convert_dtype_first = True
        else:
            convert_dtype_first = True

        if convert_dtype_first:
            return self.to(dtype=dtype, copy=copy).to(unit=unit, copy=False)
        else:
            return self.to(unit=unit, copy=copy).to(dtype=dtype, copy=False)

    def _parse_key(
        self, key: int | slice | dict[Dim, int | slice] | EllipsisType
    ) -> tuple[Dims, Shape]:
        if key is Ellipsis:
            return self.dims, key
        if not isinstance(key, Mapping):
            if self.ndim != 1:
                raise DimensionError("Only 1-D arrays can be indexed without dims")
            key = {self.dim: key}

        dims = tuple(dim for dim in self.dims if not isinstance(key.get(dim), int))
        values_key = tuple(key.pop(dim, slice(None)) for dim in self.dims)
        if key:
            raise DimensionError(f"Unknown dimensions: {tuple(key.keys())}")
        return dims, values_key

    def __getitem__(
        self: DimArr, key: int | slice | dict[Dim, int | slice] | EllipsisType
    ) -> DimArr:
        """
        Get a sub-array identified by key.

        Parameters
        ----------
        key:
            Index or slice or dictionary of dimension names and indices.

        Returns
        -------
        :
            Sub-array.
        """
        dims, values_key = self._parse_key(key)
        return self.__class__(values=self.values[values_key], dims=dims, unit=self.unit)

    def __setitem__(
        self: DimArr,
        key: int | slice | dict[Dim, int | slice] | EllipsisType,
        array: DimArr,
    ):
        """
        Set a sub-array identified by key to the values of array.

        The array to set will be automatically transposed and/or broadcast to match the
        dimensions of the sub-array, using the named dimensions.

        Parameters
        ----------
        key:
            Index or slice or dictionary of dimension names and indices.
        array:
            Array to set.
        """
        from .common import (
            broadcast_and_transpose_values,
            check_compatible_dims_and_shape,
        )

        check_compatible_dims_and_shape(self, array)

        dims, values_key = self._parse_key(key)
        if any(dim not in dims for dim in array.dims):
            raise DimensionError("Value has extra dimensions")
        if array.unit != self.unit:
            raise UnitsError("Units must be identical")
        self.values[values_key] = broadcast_and_transpose_values(array=array, dims=dims)

    def __neg__(self: DimArr) -> DimArr:
        from .common import unary

        return unary(
            self, values_op=self.values.__class__.__neg__, unit_op=_unchanged_unit
        )

    def __add__(self: DimArr, other: DimArr) -> DimArr:
        from .common import elemwise_binary

        return elemwise_binary(
            self,
            other,
            values_op=self.values.__class__.__add__,
            unit_op=_same_unit,
        )

    def __mul__(self: DimArr, other: DimArr) -> DimArr:
        from .common import elemwise_binary

        return elemwise_binary(
            self,
            other,
            values_op=self.values.__class__.__mul__,
            unit_op=operator.mul,
        )


def _unchanged_unit(unit: UnitImplementation) -> UnitImplementation:
    return unit


def _same_unit(a: UnitImplementation, b: UnitImplementation) -> UnitImplementation:
    if a != b:
        raise ValueError("Units must be identical")
    return a


def _unit_must_be_dimensionless(unit: UnitImplementation) -> UnitImplementation:
    if unit * unit != unit:
        raise ValueError("Unit must be dimensionless")
    return unit


def exp(x: DimArr, /) -> DimArr:
    from .common import unary

    return unary(
        x, values_op=x.array_namespace.exp, unit_op=_unit_must_be_dimensionless
    )
