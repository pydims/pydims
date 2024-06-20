# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Dims contributors (https://github.com/pydray)
from __future__ import annotations

import importlib
from collections.abc import Callable, Hashable, Mapping
from types import ModuleType
from typing import Any, Protocol

DType = Any  # Is the array API standard defining a DType type?


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
        dims: tuple[Hashable, ...],
        unit: UnitImplementation | None,
    ):
        if len(dims) != values.ndim:
            raise ValueError(
                f"Number of dimensions ({values.ndim}) does "
                f"not match number of dims ({len(dims)})"
            )
        self._values = values
        self._dims = dims
        self._unit = unit

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
    def dims(self) -> tuple[Hashable, ...]:
        return self._dims

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

    def _unary_op(
        self,
        values_op: Callable[[ArrayImplementation], ArrayImplementation],
        unit_op: Callable[[UnitImplementation], UnitImplementation] | None = None,
    ) -> DimensionedArray:
        return DimensionedArray(
            values=values_op(self.values),
            dims=self.dims,
            unit=None if self.unit is None else unit_op(self.unit),
        )

    def __neg__(self) -> DimensionedArray:
        return self._unary_op(
            values_op=self.values.__class__.__neg__, unit_op=_unchanged_unit
        )


def _unchanged_unit(unit: UnitImplementation) -> UnitImplementation:
    return unit


def _unit_must_be_dimensionless(unit: UnitImplementation) -> UnitImplementation:
    if unit != unit.dimensionless:
        raise ValueError("Unit must be dimensionless")
    return unit


def _array_namespace(x: DimensionedArray) -> ModuleType:
    # I thought __array_namespace__ should give this? NumPy does not have it.
    module_name = x.values.__class__.__module__
    return importlib.import_module(module_name)


def exp(x: DimensionedArray, /) -> DimensionedArray:
    return x._unary_op(
        values_op=_array_namespace(x).exp, unit_op=_unit_must_be_dimensionless
    )
