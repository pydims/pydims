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
from .dimensioned_array import DimensionedArray, DimensionError, exp, UnitsError
from .array_api_manipulation_functions import (
    broadcast_to,
    concat,
    expand_dims,
    flatten,
    fold,
    moveaxis,
    permute_dims,
    reshape,
    squeeze,
    stack,
)
from .reduction_functions import all, any, max, min, sum, mean, prod, std, var

DimensionedArray.expand_dims = expand_dims
DimensionedArray.flatten = flatten
DimensionedArray.fold = fold
DimensionedArray.permute_dims = permute_dims
DimensionedArray.reshape = reshape
DimensionedArray.squeeze = squeeze

__all__ = [
    'all',
    'any',
    'broadcast_to',
    'CreationFunctions',
    'DimensionedArray',
    'DimensionError',
    'expand_dims',
    'exp',
    'flatten',
    'fold',
    'concat',
    'moveaxis',
    'permute_dims',
    'reshape',
    'squeeze',
    'stack',
    'UnitsError',
    'max',
    'mean',
    'min',
    'prod',
    'std',
    'sum',
    'var',
]
