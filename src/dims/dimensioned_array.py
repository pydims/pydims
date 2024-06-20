# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Dims contributors (https://github.com/pydray)
from collections.abc import Hashable, Mapping
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
    def shape(self) -> tuple[int, ...]:
        return self._values.shape

    @property
    def sizes(self) -> Sizes:
        return Sizes(dims=self.dims, shape=self.shape)

    @property
    def values(self) -> ArrayImplementation:
        return self._values
