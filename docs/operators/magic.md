# `jaxdf.operators.magic`

🛠️ **Documentation is work in progress**

The functions in this module override the corresponding magic methods of the `Field` class and derived classes.

## `__add__`

Implements the `+` operator for fields.

{{ implementations('jaxdf.operators.magic', '__add__') }}

## `__bool__`

Implements the `bool()` operator for fields.

## `__divmod__`

Implements the `divmod()` operator for fields, which for two fields $`u`$ and $`v`$ returns a pair of fields `(u // v, u % v)`.

{{ implementations('jaxdf.operators.magic', '__divmod__') }}


## `__mul__`

Implements the `*` operator for fields.

{{ implementations('jaxdf.operators.magic', '__mul__') }}

## `__neg__`

Given a field $`u`$, returns the field $`-u`$, using the syntax `-u`.

{{ implementations('jaxdf.operators.magic', '__neg__') }}

## `__pow__`

Given a field $`u`$ and a generic $`c`$, returns the field $`u^c`$, using the syntax `u**c`.

{{ implementations('jaxdf.operators.magic', '__pow__') }}

## `__radd__`

Implements the operation `x + u`, when `x` is **not** a field.

{{ implementations('jaxdf.operators.magic', '__radd__') }}

## `__rmul__`

Implements the operation `x * u`, when `x` is **not** a field.

{{ implementations('jaxdf.operators.magic', '__rmul__') }}

## `__rsub__`

Implements the operation `x - u`, when `x` is **not** a field.

{{ implementations('jaxdf.operators.magic', '__rsub__') }}

## `__rtruediv__`

Implements the operation `x / u`, when `x` is **not** a field.

{{ implementations('jaxdf.operators.magic', '__rtruediv__') }}

## `__sub__`

Implements the `-` operator for fields.

{{ implementations('jaxdf.operators.magic', '__sub__') }}

## `__truediv__`

Implements the `/` operator for fields.

{{ implementations('jaxdf.operators.magic', '__truediv__') }}
