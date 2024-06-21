# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 PyDims contributors (https://github.com/pydims)

import pydims as dms


def assert_identical(a: dms.DimensionedArray, b: dms.DimensionedArray):
    # TODO This is a quick hack for early testing
    assert a.dims == b.dims  # noqa: S101
    assert a.shape == b.shape  # noqa: S101
    assert a.dtype == b.dtype  # noqa: S101
    assert a.unit == b.unit  # noqa: S101
    assert a.array_api.all(a.values == b.values)  # noqa: S101
