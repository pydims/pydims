[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![PyPI badge](http://img.shields.io/pypi/v/pydims.svg)](https://pypi.python.org/pypi/pydims)
[![Anaconda-Server Badge](https://anaconda.org/pydims/pydims/badges/version.svg)](https://anaconda.org/pydims/pydims)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

# PyDims

## About

Python arrays with named dimension and physical units.

Our goal is to provide a library wrapping any array library that supports the [Python array API standard](https://data-apis.org/array-api/latest/), adding named dimensions as well as physical units.
This is an alternative to Xarray's `NamedArray`.
The two main differences are
1. NamedArray supports `attrs`, which we think add unnecessary complexity while suffering from [conceptual problems](https://scipp.github.io/development/adr/0016-do-not-support-attrs.html).
2. PyDims supports physical units via a `unit` attribute, which we consider central and tightly linked to the concept of an array.

## Installation

```sh
python -m pip install pydims
```
