# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Dims contributors (https://github.com/pydims)
from __future__ import annotations

from typing import Any

from .dimensioned_array import Dim, DimensionedArray, UnitImplementation

# TODO
# For creation functions, add a helper that can be initialized with an array namespace,
# to forward the value creation to the array namespace. This way, the array namespace
# can be set once and then all creation functions can be called without specifying the
# array namespace.


class CreationFunctions:
    def __init__(self, array_api: Any, unit_api: Any):
        self._array_api = array_api
        self._unit_api = unit_api

    def linspace(
        self,
        dim: Dim,
        start: complex,
        stop: complex,
        /,
        num: int,
        *,
        unit: (
            Any | UnitImplementation
        ),  # TODO Default to different unit based on type if not given?
        **kwargs: Any,  # dtype, device, endpoint
    ) -> DimensionedArray:
        # TODO always call _unit_api, or check if isinstance already?
        return DimensionedArray(
            values=self._array_api.linspace(start, stop, num, **kwargs),
            dims=(dim,),
            unit=self._unit_api(unit),
        )
