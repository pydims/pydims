# Array API Implementation Overview

## Array object

### Operators

| Operation | Unit | Group |
|-----------|------|-------|
|`__pos__`|keep||
|`__neg__`|keep||
|`__add__`|compare + keep||
|`__sub__`|compare + keep||
|`__mul__`|`*`||
|`__truediv__`|`/`||
|`__floordiv__`|`/`||
|`__mod__`|TODO||
|`__pow__`|`**`||
|`__invert__`|must be `None`|
|`__and__`|must be `None`|
|`__or__`|must be `None`|
|`__xor__`|must be `None`|
|`__lshift__`|must be `None`|
|`__rshift__`|must be `None`|
|`__lt__`|compare, then `None`|
|`__le__`|compare, then `None`|
|`__gt__`|compare, then `None`|
|`__ge__`|compare, then `None`|
|`__eq__`|compare, then `None`|
|`__ne__`|compare, then `None`|
|`__abs__`|keep||
|`__bool__`|must be `None`||
|`__complex__`|must be dimensionless or `None`||
|`__float__`|must be dimensionless or `None`||
|`__index__`|must be `None`||
|`__int__`|must be dimensionless or `None`||


- Inplace operators `__iadd__`, `__isub__`, `__imul__`, `__itruediv__`, `__ifloordiv__`, `__imod__`, `__ipow__`, `__iand__`, `__ior__`, `__ixor__`, `__ilshift__`, `__irshift__` compare unit if a view, otherwise update as above.
- Reflected operators `__radd__`, `__rsub__`, `__rmul__`, `__rtruediv__`, `__rfloordiv__`, `__rmod__`, `__rpow__`, `__iand__`, `__ior__`, `__ixor__`, `__ilshift__`, `__irshift__` operate as above.
- If one operand is a plain value (Python scalar), convert it to a `DimensionedArray` with either dimensionless unit or `unit=None`, depending on the other operand.
  No other unit inference is done.
