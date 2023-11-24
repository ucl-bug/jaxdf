from typing import List

import numpy as np
from jax import numpy as jnp

from jaxdf.conv import reflection_conv
from jaxdf.core import operator
from jaxdf.discretization import (Continuous, FiniteDifferences, FourierSeries,
                                  OnGrid)
from jaxdf.operators.differential import get_fd_coefficients


# compose
@operator.abstract
def compose(x):    # pragma: no cover
  raise NotImplementedError


@operator    # type: ignore
def compose(x: Continuous, *, params=None):
  r"""Applies function composition on the `get_fun` of the Continuous object."""
  get_x = x.get_fun

  def decorator(fun):

    def new_fun(p, coord):
      return fun(get_x(p, coord))

    return Continuous(x.params, x.domain, new_fun)

  return decorator


@operator    # type: ignore
def compose(x: OnGrid, *, params=None):
  r"""Maps the given function over the pytree of parameters
    of the `Field`.

    """

  def decorator(fun):
    return x.replace_params(fun(x.params))

  return decorator


@operator    # type: ignore
def compose(x: object, *, params=None):
  r"""For non-field objects, the composition is simply the
    application of the `jax` function to the input.

    ```python
    compose(x)(fun) == fun(x)
    ```
    """

  def decorator(fun):
    return fun(x)

  return decorator


@operator    # type: ignore
def functional(x: object, *, params=None):
  r"""For non-field objects, a functional is simply the
    application of the `jax` function to the input.

    ```python
    functional(x)(fun) == fun(x)
    ```
    """

  def decorator(fun):
    return fun(x)

  return decorator


@operator    # type: ignore
def functional(x: OnGrid, *, params=None):
  r"""Maps the given function over the parameters of the field

    !!! example
        ```python
        x = OnGrid(params=-1.0, ...)
        y = functional(x)(jnp.sum)
        y.params # This is -1.0
        ```
    """

  def decorator(fun):
    return fun(x.params)

  return decorator


# get_component
@operator.abstract
def get_component(x):    # pragma: no cover
  raise NotImplementedError


@operator
def get_component(x: OnGrid, *, dim: int, params=None) -> OnGrid:
  r"""Slices the parameters of the field along the last dimensions,
    at the index specified by `dim`.

    Args:
      x: The field to slice
      dim: The index to slice at

    Returns:
      A new 1D field corresponding to the `dim`-th component of the input field.
    """
  new_params = jnp.expand_dims(x.params[..., dim], axis=-1)
  return x.replace_params(new_params)


# shift_operator
@operator.abstract
def shift_operator(x):    # pragma: no cover
  raise NotImplementedError


@operator    # type: ignore
def shift_operator(x: Continuous, *, dx: object, params=None) -> Continuous:
  r"""Shifts the field by `dx` using function composition.

    Args:
      x: The field to shift
      dx: The shift to apply

    Returns:
      A new field corresponding to the shifted input field.
    """
  get_x = x.get_fun

  def fun(p, coord):
    return get_x(p, coord + dx)

  return Continuous(x.params, x.domain, fun)


def fd_shift_kernels(x: FiniteDifferences, dx: List[float], *args, **kwargs):
  r"""Computes the shift kernels for FiniteDifferences fields.

    Args:
      x: The field to shift
      dx: The shift to apply

    Returns:
      The kernel to apply to the field coefficients in order to
      shift the field.
    """

  def single_kernel(axis, stagger):
    kernel = get_fd_coefficients(x, order=0, stagger=stagger)
    if x.domain.ndim > 1:
      for _ in range(x.domain.ndim - 1):
        kernel = np.expand_dims(kernel, axis=0)
      # Move kernel to the correct axis
      kernel = np.moveaxis(kernel, -1, axis)
    # kernel = kernel / x.domain.dx[axis]
    return kernel

  stagger = dx[0] / x.domain.dx[0]
  params = []
  for i in range(x.domain.ndim):
    params.append(single_kernel(axis=i, stagger=stagger))

  return params


@operator(init_params=fd_shift_kernels)    # type: ignore
def shift_operator(x: FiniteDifferences,
                   *,
                   dx=[0.0],
                   params=None) -> FiniteDifferences:
  r"""Shifts the field by `dx` using finite differences.

    Args:
      x: The field to shift
      dx: The shift to apply. It is ignored if the `params` argument is not `None`.

    Returns:
      A new field corresponding to the shifted input field.
    """
  # Apply convolution
  kernels = params
  array = x.on_grid

  # Apply the corresponding kernel to each dimension
  outs = [
      reflection_conv(kernels[i], array[..., i], i)
      for i in range(x.domain.ndim)
  ]
  new_params = jnp.stack(outs, axis=-1)

  return x.replace_params(new_params)


@operator(
    init_params=lambda x, *, dx: {"k_vec": x._freq_axis})    # type: ignore
def shift_operator(x: FourierSeries, *, dx=[0], params=None) -> FourierSeries:
  r"""Shifts the field by `dx` using the shift theorem in Fourier space.

    Args:
      x: The field to shift
      dx: The shift to apply

    Returns:
      A new field corresponding to the shifted input field.
    """
  if x.is_real:
    ffts = [jnp.fft.rfft, jnp.fft.irfft]
  else:
    ffts = [jnp.fft.fft, jnp.fft.ifft]
  k_vec = params["k_vec"]

  # Adding staggering
  if len(dx) == 1 and len(x.domain.N) != 1:
    dx = dx * len(x.domain.N)

  k_vec = [jnp.exp(1j * k * delta) for k, delta in zip(k_vec, dx)]

  new_params = jnp.zeros_like(x.params)

  def single_grad(axis, u):
    u = jnp.moveaxis(u, axis, -1)
    Fx = ffts[0](u, axis=-1)
    iku = Fx * k_vec[axis]
    du = ffts[1](iku, axis=-1, n=u.shape[-1])
    return jnp.moveaxis(du, -1, axis)

  for ax in range(x.domain.ndim):
    new_params = new_params.at[..., ax].set(single_grad(ax, x.params[..., ax]))

  return FourierSeries(new_params, x.domain), params


# sum_over_dims
@operator    # type: ignore
def sum_over_dims(x: Continuous, *, params=None):
  get_x = x.get_fun

  def fun(p, coords):
    return jnp.sum(get_x(p, coords), axis=-1, keepdims=True)

  return x.update_fun_and_params(x.params, fun)


@operator    # type: ignore
def sum_over_dims(x: OnGrid, *, params=None):
  new_params = jnp.sum(x.params, axis=-1, keepdims=True)
  return x.replace_params(new_params)


if __name__ == "__main__":
  from jaxdf.util import _get_implemented

  print(compose.__name__)
  print(compose.__doc__)
  funcs = [compose, sum_over_dims]

  print("functions.py:")
  print("----------------")
  for f in funcs:
    _get_implemented(f)
  print("\n")
