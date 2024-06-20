# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Pydims contributors (https://github.com/pydims)
# ruff: noqa: E402, F401

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

from .creation_functions import CreationFunctions
from .dimensioned_array import DimensionedArray, DimensionError, exp
from .array_api_manipulation_functions import concat, reshape, stack
from .array_api_utility_functions import all, any

__all__ = [
    'all',
    'any',
    'CreationFunctions',
    'DimensionedArray',
    'DimensionError',
    'exp',
    'concat',
    'reshape',
    'stack',
]
