# `jaxdf.operators.functions`

## `compose`

Implements a decorator that allows to compose `jax` functions with fields. Given a function $`f`$ and a `Field` $`x`$, the result is a new field representing

```math
y = f(x)
```

The usage of the decorator is as follows:
```python
y = compose(x)(f)
```

{{ implementations('jaxdf.operators.functions', function='compose') }}

## `shift_operator`

Implements the shift operator $`S(\Delta x)`$ which is used to shift (spatially) a field $`u`$ by a constant $`\Delta x`$:

```math
v = S(\Delta x) u = u(x - \Delta x)
```

{{ implementations('jaxdf.operators.functions', function='shift_operator') }}


## `sum_over_dims`

Reduces a vector field $`u = (u_x, u_y, \dots)`$ to a scalar field by summing over the dimensions:

```math
v = \sum_{i \in \{x,y,\dots\}} u_i
```
  
{{ implementations('jaxdf.operators.functions', function='sum_over_dims') }}