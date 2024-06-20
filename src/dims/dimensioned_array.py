# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Dims contributors (https://github.com/pydims)
from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from functools import cached_property
from typing import Any, Protocol

import array_api_compat

DType = Any  # Is the array API standard defining a DType type?


class DimensionError(Exception):
    pass


class ArrayImplementation(Protocol):
    """Array of values following the Python array API standard."""


class UnitImplementation(Protocol):
    pass


class Sizes(Mapping):
    def __init__(self, dims: tuple[Hashable, ...], shape: tuple[int, ...]):
        self._data = dict(zip(dims, shape, strict=True))

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


Dims = tuple[Hashable, ...]


class DimensionedArray:
    """
    General idea:
    - __getitem__ accepts dict with dims labels. Only 1-D allows for omitting index.
    - Probably we need to support duplicate dims
    - dims should be readonly?
    - unit must avoid assigning from slices? Do we need readonly flags?
    """

    def __init__(
        self,
        values: ArrayImplementation,
        dims: Dims,
        unit: UnitImplementation | None,
    ):
        if len(dims) != values.ndim:
            raise ValueError(
                f"Number of dimensions ({values.ndim}) does "
                f"not match number of dims ({len(dims)})"
            )
        self._values = values
        self._dims = tuple(dims)
        self._unit = unit

    def __str__(self) -> str:
        return (
            f"dims={self.dims}\nshape={self.shape}\n"
            f"values={self.values}\nunit={self.unit}"
        )

    @cached_property
    def array_api(self) -> Any:
        return array_api_compat.array_namespace(self.values)

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
    def dim(self) -> Hashable:
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

    def elemwise_unary(
        self,
        values_op: Callable[[ArrayImplementation], ArrayImplementation],
        unit_op: Callable[[UnitImplementation], UnitImplementation],
    ) -> DimensionedArray:
        return DimensionedArray(
            values=values_op(self.values),
            dims=self.dims,
            unit=None if self.unit is None else unit_op(self.unit),
        )

    def elemwise_binary(
        self,
        /,
        other: DimensionedArray,
        *,
        values_op: Callable[
            [ArrayImplementation, ArrayImplementation], ArrayImplementation
        ],
        unit_op: Callable[[UnitImplementation, UnitImplementation], UnitImplementation],
    ) -> DimensionedArray:
        dims = _merge_dims(self.dims, other.dims)
        a = self.values
        b = other.array_api.moveaxis(
            other.values,
            source=[other.dims.index(dim) for dim in dims if dim in other.dims],
            destination=range(other.ndim),
        )
        for dim in dims:
            if dim not in self.dims:
                a = self.array_api.expand_dims(a, axis=dims.index(dim))
            if dim not in other.dims:
                b = other.array_api.expand_dims(b, axis=dims.index(dim))
        return DimensionedArray(
            values=values_op(a, b),
            dims=dims,
            unit=(
                None
                # TODO do not mix unit with None
                if self.unit is None and other.unit is None
                else unit_op(self.unit, other.unit)
            ),
        )

    def __getitem__(
        self, key: int | slice | dict[Hashable, int | slice]
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
        return self.elemwise_unary(
            values_op=self.values.__class__.__neg__, unit_op=_unchanged_unit
        )

    def __add__(self, other: DimensionedArray) -> DimensionedArray:
        return self.elemwise_binary(
            other,
            values_op=self.values.__class__.__add__,
            unit_op=_same_unit,
        )


def _merge_dims(a: Dims, b: Dims) -> Dims:
    """Favor order in a."""
    # TODO Avoid transpose of b if possible
    return a + tuple(dim for dim in b if dim not in a)


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
    return x.elemwise_unary(
        values_op=x.array_api.exp, unit_op=_unit_must_be_dimensionless
    )
