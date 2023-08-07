# `jaxdf.operators.functions`

## Operators

## `compose`

Implements a decorator that allows to compose `jax` functions with fields. Given a function $f$ and a `Field` $x$, the result is a new field representing

$$
y = f(x)
$$

The usage of the decorator is as follows:
```python
y = compose(x)(f)
```

{{ implementations('jaxdf.operators.functions', 'compose') }}

## `functional`

It works exactly like `compose`, but the function `f` maps one or more fields to a scalar value.

The usage of the decorator is as follows:
```python
y = functional(x)(f)
```

It is useful to improve the readibility of the code.

{{ implementations('jaxdf.operators.functions', 'functional') }}

## `get_component`

This operator $A(u, \text{dim})$ which has the signature `(u: Field, dim: int) -> Field`. It returns the component of the field $u$ at the dimension $dim$.

$$
u(x) = (u_0(x), u_1(x), \ldots, u_N(x)) \to u_{\text{dim}}(x)
$$

{{ implementations('jaxdf.operators.functions', 'get_component') }}

## `shift_operator`

Implements the shift operator $S(\Delta x)$ which is used to shift (spatially) a field $u$ by a constant $\Delta x$:

$$
v = S(\Delta x) u = u(x - \Delta x)
$$

{{ implementations('jaxdf.operators.functions', 'shift_operator') }}


## `sum_over_dims`

Reduces a vector field $u = (u_x, u_y, \dots)$ to a scalar field by summing over the dimensions:

$$
v = \sum_{i \in \{x,y,\dots\}} u_i
$$

{{ implementations('jaxdf.operators.functions', 'sum_over_dims') }}

## Utilities

::: jaxdf.operators.functions
    handler: python
    options:
        members:
            - fd_shift_kernels
        show_root_heading: false
        show_root_toc_entry: false
        show_source: false
        show_object_full_path: false
