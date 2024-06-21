# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Pydims contributors (https://github.com/pydims)
import pydims as pkg


def test_has_version():
    assert hasattr(pkg, '__version__')
