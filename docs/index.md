# PyDims

<span style="font-size:1.2em;font-style:italic;color:#5a5a5a">
  Python arrays with named dimension and physical units
  </br></br>
</span>

## Why PyDims?

By now everyone has concluded that named dimensions are great and essential.
Xarray provides this, and is currently working on factoring out `xarray.Variable` into a separate package, providing `NamedArray`.
We think that `NamedArray` is a great idea, but we also think that it is missing a crucial feature, physical units, as well as introducing some unnecessary complexity and conceptual problems, by supporting `attrs`.
We are not convinced that handling units in the underlying array (via the `dtype`, via subclassing the array, or via a separate array) is ideal when this array is wrapped to add dimension names, since usability suffers.

The idea of PyDims is thus:

1. Define a new array class with an array of values, dimension names, and an optional physical unit.
   The interface of the new class will *not*  implement the [Python array API standard](https://data-apis.org/array-api/latest/).
   Instead it will try to *follow the gist of the standard*, but with modifications enabled by or forced by named dimensions and units.
2. The `values` can be anything that implements the [Python array API standard](https://data-apis.org/array-api/latest/).
3. The `unit` can be anything that implements a to-be-defined units API.
   Just like the Array API avoids forcing the user into a specific array library, the goal of a units API will avoid forcing the user into a specific units library.


## At a glance

### Install

```sh
pip install pydims
```

### Use

Note that the implementation is currently very incomplete.
This is mainly a proof of concept.
We give a couple of example combining some common Array implementations and units libraries:

With NumPy and Pint:

```python
import numpy as np
from pint import UnitRegistry
import pydims as dims

ureg = UnitRegistry()

make = dims.CreationFunctions(np, ureg.Unit)
a = make.ones(dims=('x', 'y'), shape=(10,10), unit='1/s')
b = make.linspace('x', 0, 9000, 10, unit='m')
c = a * b
c = c.to(unit='km/s')
result = c[{'x': slice(2, 7), 'y': slice(2, 4)}]
result
```

With Dask Array and AstroPy units:

```python
from dask import array
from astropy.units import Unit
import pydims as dms

make = dms.CreationFunctions(array, Unit)

a = make.ones(dims=('x', 'y'), shape=(10,10), unit='1/s', chunks=(5,5))
b = make.linspace('x', 0, 9000, 10, unit='m', chunks=(5,))
c = a * b
c = c.to(unit='km/s')
c = c[{'x': slice(2, 7), 'y': slice(2, 4)}]
result = dms.common.unary(c, values_op=lambda x: x.compute(), unit_op=lambda x: x)
result
```

```{toctree}
---
hidden:
---

user-guide/index
api-reference/index
developer/index
about/index
```
