# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Dims contributors (https://github.com/pydims)
from __future__ import annotations

import operator
from collections.abc import Hashable, Iterator, Mapping
from typing import Any, Protocol

import array_api_compat

DType = Any  # Is the array API standard defining a DType type?


class DimensionError(Exception):
    pass


class ArrayImplementation(Protocol):
    """Array of values following the Python array API standard."""


class UnitImplementation(Protocol):
    pass


# Tuple is hashable so this leads to some ambiguity, but if in doubt a tuple is
# interpreted as a tuple of dimensions, not a single dimension.
Dim = Hashable
Dims = tuple[Dim, ...]


class Sizes(Mapping):
    def __init__(self, dims: Dims, shape: tuple[int, ...]):
        self._data = dict(zip(dims, shape, strict=True))

    def __getitem__(self, key) -> int:
        return self._data[key]

    def __iter__(self) -> Iterator[Dim]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


class DimensionedArray:
    """
    Array with named dimensions and optional unit.
    """

    def __init__(
        self,
        *,
        values: ArrayImplementation,
        dims: Dims,
        unit: UnitImplementation | None,
        unit_api: Any | None = None,
    ):
        if len(dims) != values.ndim:
            raise ValueError(
                f"Number of dimensions ({values.ndim}) does "
                f"not match number of dims ({len(dims)})"
            )
        self._values = values
        self._dims = tuple(dims)
        self._unit = unit
        self.unit_api = unit_api

    def __str__(self) -> str:
        return (
            f"dims={self.dims}\nshape={self.shape}\n"
            f"values={self.values}\nunit={self.unit}"
        )

    @property
    def array_api(self) -> Any:
        return array_api_compat.array_namespace(self.values)

    # @property
    # def unit_api(self) -> Any:
    #    # Hack for now, consider need to define a standard for unit APIs
    #    return self.unit.__class__

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
    def shape(self) -> tuple[int, ...]:
        return self._values.shape

    @property
    def sizes(self) -> Sizes:
        return Sizes(dims=self.dims, shape=self.shape)

    @property
    def values(self) -> ArrayImplementation:
        return self._values

    def to(
        self, *, dtype: DType | None = None, unit: Any | None = None, copy: bool = True
    ) -> DimensionedArray:
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
        # TODO Use correct conversion order like Scipp
        if dtype is not None:
            values = self.array_api.astype(self.values, dtype, copy=copy)
        else:
            values = self.array_api.asarray(self.values, copy=copy)
        if unit is not None and (scale := self.unit.to(unit)) != 1:
            values = values * scale
        return DimensionedArray(
            values=values,
            dims=self.dims,
            unit=self.unit if unit is None else self.unit_api(unit),
        )

    def __getitem__(
        self, key: int | slice | dict[Dim, int | slice]
    ) -> DimensionedArray:
        if isinstance(key, int | slice):
            if self.ndim != 1:
                raise DimensionError("Only 1-D arrays can be indexed without dims")
            key = {self.dim: key}

        dims = tuple(dim for dim in self.dims if not isinstance(key.get(dim), int))
        values_key = tuple(key.pop(dim, slice(None)) for dim in self.dims)
        if key:
            raise ValueError(f"Unknown dimensions: {key.keys()}")
        return DimensionedArray(
            values=self.values[values_key], dims=dims, unit=self.unit
        )

    def __neg__(self) -> DimensionedArray:
        from .common import elemwise_unary

        return elemwise_unary(
            self, values_op=self.values.__class__.__neg__, unit_op=_unchanged_unit
        )

    def __add__(self, other: DimensionedArray) -> DimensionedArray:
        from .common import elemwise_binary

        return elemwise_binary(
            self,
            other,
            values_op=self.values.__class__.__add__,
            unit_op=_same_unit,
        )

    def __mul__(self, other: DimensionedArray) -> DimensionedArray:
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
    if unit != unit.dimensionless:
        raise ValueError("Unit must be dimensionless")
    return unit


def exp(x: DimensionedArray, /) -> DimensionedArray:
    from .common import elemwise_unary

    return elemwise_unary(
        x, values_op=x.array_api.exp, unit_op=_unit_must_be_dimensionless
    )
