# `jaxdf.operators.differential`
## Operators

## `derivative`

Given a field $`u`$, it returns the field

```math
\frac{\partial}{\partial \epsilon} u, \qquad \epsilon \in \{x, y, \dots \}
```

{{ implementations('jaxdf.operators.differential', 'derivative') }}


## `diag_jacobian`

Given a vector field $`u = (u_x,u_y,\dots)`$ with the same dimensions as the dimensions of the domain, it returns the diagonal of the Jacobian matrix

```math
\left( \frac{\partial u_x}{\partial x}, \frac{\partial u_y}{\partial y}, \dots \right)
```

{{ implementations('jaxdf.operators.differential', 'diag_jacobian') }}
## `gradient`

Given a field $`u`$, it returns the vector field

```math
\nabla u = \left(\frac{\partial u}{\partial x}, \frac{\partial u}{\partial y}, \dots\right)
```

{{ implementations('jaxdf.operators.differential', 'gradient') }}

## `heterog_laplacian`

Given a field $`u`$ and a cofficient field $`c`$, it returns the field

```math
\nabla_c^2 u = \nabla \cdot (c \nabla u)
```

{{ implementations('jaxdf.operators.differential', 'heterog_laplacian') }}


## `laplacian`

Given a scalar field $`u`$, it returns the scalar field

```math
\nabla^2 u = \nabla \cdot \nabla u = \sum_{\epsilon \in \{x,y,\dots\}} \frac{\partial^2 u}{\partial \epsilon^2}
```

{{ implementations('jaxdf.operators.differential', 'laplacian') }}


## Utilities

::: jaxdf.operators.differential
    handler: python
    options:
        members:
            - get_fd_coefficients
            - fd_derivative_init
        show_root_heading: false
        show_root_toc_entry: false
        show_source: false
        show_object_full_path: false
