# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Dims contributors (https://github.com/pydray)
from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
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
            values=values_op(self.values), dims=self.dims, unit=unit_op(self._unit)
        )

    def __neg__(self) -> DimensionedArray:
        return self._unary_op(
            values_op=self.values.__class__.__neg__, unit_op=_unchanged_unit
        )


def _unchanged_unit(unit: UnitImplementation) -> UnitImplementation:
    return unit


def _must_have_unit(unit: UnitImplementation) -> UnitImplementation:
    if unit is None:
        raise ValueError("Unit must be provided")


def _unit_must_be_dimensionless(unit: UnitImplementation) -> UnitImplementation:
    _must_have_unit(unit)
    if unit != unit.dimensionless:
        raise ValueError("Unit must be dimensionless")
    return unit
