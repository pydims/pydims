# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)
from __future__ import annotations

from typing import Any

from array_api_compat import array_namespace

from .dimensioned_array import (
    ArrayImplementation,
    Dim,
    DimensionedArray,
    Dims,
    Shape,
    UnitImplementation,
)
from .units_api import units_namespace

_default_unit = object()


class CreationFunctions:
    def __init__(self, array_api: Any, unit_api: Any):
        # TODO What should we pass here? Modules? A Unit class?
        self._array_api = array_namespace(array_api.zeros(0))
        self._unit_api = None if unit_api is None else units_namespace(unit_api(''))

    def _maybe_unit(
        self, unit: Any | UnitImplementation | None, values: ArrayImplementation
    ) -> UnitImplementation | None:
        if unit is _default_unit:
            if self._array_api.isdtype(values.dtype, 'numeric'):
                return self._unit_api.dimensionless
            else:
                return None
        return None if unit is None else self._unit_api.Unit(unit)

    def arange(
        self,
        dim: Dim,
        start: float,
        /,
        stop: float,
        step: float = 1,
        *,
        unit: Any | UnitImplementation | None = _default_unit,
        **kwargs: Any,  # dtype, device
    ) -> DimensionedArray:
        values = self._array_api.arange(start, stop, step, **kwargs)
        return DimensionedArray(
            values=values, dims=(dim,), unit=self._maybe_unit(unit, values)
        )

    def asarray(
        self,
        dims: Dims,
        values: Any,
        *,
        unit: Any | UnitImplementation | None = _default_unit,
        **kwargs: Any,  # dtype, device, copy
    ) -> DimensionedArray:
        values = self._array_api.asarray(values, **kwargs)
        return DimensionedArray(
            values=values, dims=dims, unit=self._maybe_unit(unit, values)
        )

    def linspace(
        self,
        dim: Dim,
        start: complex,
        stop: complex,
        /,
        num: int,
        *,
        unit: Any | UnitImplementation | None = _default_unit,
        **kwargs: Any,  # dtype, device, endpoint
    ) -> DimensionedArray:
        values = self._array_api.linspace(start, stop, num, **kwargs)
        return DimensionedArray(
            values=values, dims=(dim,), unit=self._maybe_unit(unit, values)
        )

    def empty(
        self,
        dims: Dims,
        shape: Shape,
        *,
        unit: Any | UnitImplementation | None = _default_unit,
        **kwargs: Any,  # dtype, device
    ) -> DimensionedArray:
        values = (self._array_api.empty(shape, **kwargs),)
        return DimensionedArray(
            values=values, dims=dims, unit=self._maybe_unit(unit, values)
        )

    def ones(
        self,
        dims: Dims,
        shape: Shape,
        *,
        unit: Any | UnitImplementation | None = _default_unit,
        **kwargs: Any,  # dtype, device
    ) -> DimensionedArray:
        values = self._array_api.ones(shape, **kwargs)
        return DimensionedArray(
            values=values, dims=dims, unit=self._maybe_unit(unit, values)
        )

    def zeros(
        self,
        dims: Dims,
        shape: Shape,
        *,
        unit: Any | UnitImplementation | None = _default_unit,
        **kwargs: Any,  # dtype, device
    ) -> DimensionedArray:
        values = self._array_api.zeros(shape, **kwargs)
        return DimensionedArray(
            values=values, dims=dims, unit=self._maybe_unit(unit, values)
        )
