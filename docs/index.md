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

With NumPy:

```python
import pydims as dims
import numpy as np

a = dims.DimensionedArray(dims=('x',), values=np.arange(10), unit=None)
b = a[{'x': slice(2, 5)}]
```

With Dask Array and AstroPy units:

```python
from dask import array
from astropy import units
import pydims as dms

values = array.ones((10, 10), chunks=(5, 5))
a = dms.DimensionedArray(dims=('x', 'y'), values=values, unit=units.m)
b = a * a
c = b[{'x': slice(2, 7), 'y': slice(2, 4)}]
result = dms.common.unary(c, values_op=lambda x: x.compute(), unit_op=lambda x: x)
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
