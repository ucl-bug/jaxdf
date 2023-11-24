from typing import Union

import jax
import numpy as np
from jax import numpy as jnp

from jaxdf.conv import fd_coefficients_fornberg, reflection_conv
from jaxdf.core import operator
from jaxdf.discretization import Continuous, FiniteDifferences, FourierSeries


# derivative
@operator.abstract
def derivative(x):    # pragma: no cover
  # Implements the derivative operator
  raise NotImplementedError


@operator
def derivative(x: Continuous, *, axis=0, params=None) -> Continuous:
  r"""Derivative operator for continuous fields.

    Args:
      x: Continuous field
      axis: Axis along which to take the derivative

    Returns:
      Continuous field
    """
  get_x = x.get_fun

  def grad_fun(p, coords):
    f_jac = jax.jacfwd(get_x, argnums=(1, ))
    return jnp.expand_dims(f_jac(p, coords)[0][0][axis], -1)

  return Continuous(x.params, x.domain, grad_fun)


def get_fd_coefficients(x: FiniteDifferences,
                        order: int = 1,
                        stagger: Union[float, int] = 0):
  r"""Returnst the stencil coefficients for a 1D Finite Differences derivative
    operator.

    Args:
      x: FiniteDifferences field
      order: Order of the derivative
      stagger: Stagger of the derivative

    Returns:
      Stencil coefficients
    """

  # Check that all the values of stagger are in [0, 0.5, -0.5]
  # assert stagger in [0, -0.5, 0.5], f'Staggering must be in [0, 0.5, -0.5] for finite differences, got {stagger}'
  accuracy = x.accuracy
  points = np.arange(-accuracy // 2, accuracy // 2 + 1)
  if stagger > 0:
    points = (points + stagger)[:-1]
  elif stagger < 0:
    points = (points + stagger)[1:]

  # get coefficients
  coeffs = fd_coefficients_fornberg(order, points, x0=0)[0].tolist()

  # Append zero if a coefficient has been removed
  if stagger > 0:
    coeffs = coeffs + [0.0]
  elif stagger < 0:
    coeffs = [0.0] + coeffs

  return np.asarray(coeffs)


def fd_derivative_init(x: FiniteDifferences,
                       axis=0,
                       stagger=0,
                       *args,
                       **kwargs):
  r"""Initializes the stencils for FiniteDifferences derivatives. Accepts
    an arbitrary number of positional and keyword arguments after the
    mandatory arguments, which are ignored.

    Args:
      x: FiniteDifferences field
      axis: Axis along which to take the derivative
      stagger: Stagger of the derivative

    Returns:
      Stencil coefficients
    """
  kernel = get_fd_coefficients(x, order=1, stagger=stagger)

  if x.domain.ndim > 1:
    for _ in range(x.domain.ndim - 1):
      kernel = np.expand_dims(kernel, axis=0)
    # Move kernel to the correct axis
    kernel = np.moveaxis(kernel, -1, axis)

  # Add dx
  kernel = kernel / x.domain.dx[axis]

  return kernel


def fd_diag_jacobian_init(x: FiniteDifferences, stagger, *args, **kwargs):
  r"""Initializes the parameters for the diagonal Jacobian of a FiniteDifferences field. Accepts
    an arbitrary number of positional and keyword arguments after the
    mandatory arguments, which are ignored.

    Args:
      x: FiniteDifferences field
      stagger: Stagger of the derivative

    Returns:
      Stencil coefficients

    """
  if len(stagger) != x.domain.ndim:
    stagger = [stagger[0] for _ in range(x.domain.ndim)]

  kernels = []
  for i in range(x.domain.ndim):
    kernels.append(fd_derivative_init(x, axis=i, stagger=stagger[i]))

  return kernels


# gradient
@operator.abstract
def gradient(x):    # pragma: no cover
  # Implements the gradient operator
  raise NotImplementedError


@operator    # type: ignore
def gradient(x: Continuous, *, params=None) -> Continuous:
  r"""Gradient operator for continuous fields.

    Args:
      x: Continuous field

    Returns:
      The gradient of the field
    """
  get_x = x.get_fun

  def grad_fun(p, coords):
    f_jac = jax.jacfwd(get_x, argnums=(1, ))
    v = f_jac(p, coords)[0][0]
    return v

  return x.update_fun_and_params(x.params, grad_fun)


@operator(init_params=fd_diag_jacobian_init)    # type: ignore
def gradient(x: FiniteDifferences,
             *,
             stagger=[0],
             params=None) -> FiniteDifferences:
  r"""Gradient operator for finite differences fields.

    Args:
      x: FiniteDifferences field
      stagger: Stagger of the derivative

    Returns:
      The gradient of the field
    """
  return diag_jacobian(x, stagger=stagger, params=params)


@operator(init_params=lambda x, *args, **kwargs: {"k_vec": x._freq_axis}
          )    # type: ignore
def gradient(x: FourierSeries,
             *,
             stagger=[0],
             correct_nyquist=True,
             params=None) -> FourierSeries:
  r"""Gradient operator for Fourier series fields.

    Args:
        x (FourierSeries): Input field
        stagger (list, optional): Staggering value for the returned fields.
          The fields are staggered in the direction of their derivative.
          Defaults to [0].
        correct_nyquist (bool, optional): If `True`, uses a correction of the
          derivative filter for the Nyquist frequency, which preserves Hermitian
          symmetric and null space. See [those notes](https://math.mit.edu/~stevenj/fft-deriv.pdf)
          for more details. Defaults to True.

    Returns:
        FourierSeries: The gradient of the input field.
    """
  assert x.dims == 1    # Gradient only defined for scalar fields

  k_vec = params["k_vec"]

  # Adding staggering
  if len(stagger) == 1 and len(x.domain.N) != 1:
    stagger = stagger * len(x.domain.N)

  dx = x.domain.dx
  k_vec = [
      1j * k * jnp.exp(1j * k * s * delta)
      for k, delta, s in zip(k_vec, dx, stagger)
  ]

  # Set to zero the filter at the Nyquist frequency
  # if the dimension is even
  # see https://math.mit.edu/~stevenj/fft-deriv.pdf
  if correct_nyquist:
    for f in range(len(k_vec)):
      if x.domain.N[f] % 2 == 0:
        f_nyq = x.domain.N[f] // 2
        k_vec[f] = k_vec[f].at[f_nyq].set(0.0)

  u = x.params[..., 0]

  def single_grad(axis, u):
    u = jnp.moveaxis(u, axis, -1)
    Fx = x._ffts[0](u, axis=-1)
    iku = Fx * k_vec[axis]
    du = x._ffts[1](iku, axis=-1, n=u.shape[-1])
    return jnp.moveaxis(du, -1, axis)

  new_params = jnp.stack([single_grad(i, u) for i in range(x.domain.ndim)],
                         axis=-1)
  return FourierSeries(new_params, x.domain)


# diag_jacobian
@operator.abstract
def diag_jacobian(x):    # pragma: no cover
  # Extracts the diagonal of the Jacobian operator
  raise NotImplementedError


@operator    # type: ignore
def diag_jacobian(x: Continuous, *, params=None) -> Continuous:
  r"""Diagonal Jacobian operator for continuous fields.

    Args:
      x: Continuous field

    Returns:
      The diagonal Jacobian of the field
    """
  get_x = x.get_fun

  def diag_fun(p, coords):
    f_jac = jax.jacfwd(get_x, argnums=(1, ))
    return jnp.diag(f_jac(p, coords)[0])

  return x.update_fun_and_params(x.params, diag_fun)


@operator(init_params=fd_diag_jacobian_init)    # type: ignore
def diag_jacobian(x: FiniteDifferences,
                  *,
                  stagger=[0],
                  params=None) -> FiniteDifferences:
  r"""Diagonal Jacobian operator for finite differences fields.

    Args:
      x: FiniteDifferences field
      stagger: Stagger of the derivative

    Returns:
      The diagonal Jacobian of the field
    """
  kernels = params
  array = x.on_grid

  # Apply the corresponding kernel to each dimension
  outs = [
      reflection_conv(kernels[i], array[..., i]) for i in range(x.domain.ndim)
  ]
  new_params = jnp.stack(outs, axis=-1)

  return x.replace_params(new_params)


@operator(init_params=lambda x, *args, **kwargs: {"k_vec": x._freq_axis}
          )    # type: ignore
def diag_jacobian(x: FourierSeries,
                  *,
                  stagger=[0],
                  correct_nyquist=True,
                  params=None) -> FourierSeries:
  r"""Diagonal Jacobian operator for Fourier series fields.

    Args:
        x (FourierSeries): Input field
        stagger (list, optional): Staggering value for the returned fields.
          The fields are staggered in the direction of their derivative.
          Defaults to [0].
        correct_nyquist (bool, optional): If `True`, uses a correction of the
          derivative filter for the Nyquist frequency, which preserves Hermitian
          symmetric and null space. See [those notes](https://math.mit.edu/~stevenj/fft-deriv.pdf)
          for more details. Defaults to True.

    Returns:
        The vector field whose components are the diagonal entries
          of the Jacobian of the input field.
    """
  k_vec = params["k_vec"]

  # Adding staggering
  if len(stagger) == 1 and len(x.domain.N) != 1:
    stagger = stagger * len(x.domain.N)

  dx = x.domain.dx
  k_vec = [
      1j * k * jnp.exp(1j * k * s * delta)
      for k, delta, s in zip(k_vec, dx, stagger)
  ]

  # Set to zero the filter at the Nyquist frequency
  # if the dimension is even
  # see https://math.mit.edu/~stevenj/fft-deriv.pdf
  if correct_nyquist:
    for f in range(len(k_vec)):
      if x.domain.N[f] % 2 == 0:
        f_nyq = x.domain.N[f] // 2
        k_vec[f] = k_vec[f].at[f_nyq].set(0.0)

  new_params = jnp.zeros_like(x.params)

  def single_grad(axis, u):
    u = jnp.moveaxis(u, axis, -1)
    Fx = x._ffts[0](u, axis=-1)
    iku = Fx * k_vec[axis]
    du = x._ffts[1](iku, axis=-1, n=u.shape[-1])
    return jnp.moveaxis(du, -1, axis)

  for ax in range(x.domain.ndim):
    new_params = new_params.at[..., ax].set(single_grad(ax, x.params[..., ax]))

  return FourierSeries.from_grid(new_params, x.domain)


# laplacian
@operator.abstract
def laplacian(x):    # pragma: no cover
  # Implements the Laplacian operator
  raise NotImplementedError


@operator    # type: ignore
def laplacian(x: Continuous, *, params=None) -> Continuous:
  r"""Laplacian operator for continuous fields.

    Args:
      x: Continuous field

    Returns:
      The Laplacian of the field
    """
  get_x = x.get_fun

  def grad_fun(p, coords):
    hessian = jax.hessian(get_x, argnums=(1, ))(p, coords)[0][0][0]
    return jnp.diag(hessian)

  return x.update_fun_and_params(x.params, grad_fun)


@operator(init_params=lambda x, *args, **kwargs: {"k_vec": x._freq_axis}
          )    # type: ignore
def laplacian(x: FourierSeries, *, params=None):
  r"""Laplacian operator for Fourier series fields.

    Args:
      x (FourierSeries): Input field

    Returns:
      The Laplacian of the field
    """
  assert x.dims == 1    # Laplacian only defined for scalar fields

  k_vec = params["k_vec"]
  u = x.params[..., 0]

  def single_grad(axis, u):
    u = jnp.moveaxis(u, axis, -1)
    Fx = x._ffts[0](u, axis=-1)
    iku = -Fx * (k_vec[axis]**2)
    du = x._ffts[1](iku, axis=-1, n=u.shape[-1])
    return jnp.moveaxis(du, -1, axis)

  new_params = jnp.sum(
      jnp.stack([single_grad(i, u) for i in range(x.domain.ndim)], axis=-1),
      axis=-1,
      keepdims=True,
  )
  return FourierSeries(new_params, x.domain)


def fd_laplacian_init(x: FiniteDifferences, *args, **kwargs):
  r"""Initializes the parameters for the Laplacian of a FiniteDifferences field. Accepts
    an arbitrary number of positional and keyword arguments after the
    mandatory arguments, which are ignored.

    Args:
      x: FiniteDifferences field

    Returns:
      Stencil coefficients

    """

  # Get kernel for 1D
  kernel = get_fd_coefficients(x, order=2)

  # Make it into an ND kernel by repeating it
  if x.domain.ndim == 1:
    return kernel

  full_kernel_shape = [kernel.shape[0]] * x.domain.ndim
  full_kernel = np.zeros(full_kernel_shape, dtype=kernel.dtype)
  center = kernel.shape[0] // 2

  # Put the kernel in the middle of the array for each dimension, by adding it to the full kernel
  for i in range(x.domain.ndim):
    full_kernel[tuple([
        center if j == i else slice(None) for j in range(x.domain.ndim)
    ])] += kernel

  return full_kernel


@operator(init_params=fd_laplacian_init)    # type: ignore
def laplacian(x: FiniteDifferences, *, params=None) -> FiniteDifferences:
  r"""Gradient operator for finite differences fields.

    Args:
      x: FiniteDifferences field

    Returns:
      The gradient of the field
    """
  assert x.dims == 1, "Laplacian only defined for scalar fields"
  new_params = reflection_conv(params, x.on_grid[..., 0], reverse=False)
  return x.replace_params(new_params)


# heterog_laplacian
@operator.abstract
def heterog_laplacian(x):    # pragma: no cover
  # Implements the Heterogeneous Laplacian operator
  raise NotImplementedError


@operator(init_params=lambda x, *args, **kwargs: {"k_vec": x._freq_axis}
          )    # type: ignore
def heterog_laplacian(x: FourierSeries,
                      c: FourierSeries,
                      *,
                      params=None) -> FourierSeries:
  r"""Computes the position-varying laplacian using algorithm 4 of
    [[Johnson, 2011]](https://math.mit.edu/~stevenj/fft-deriv.pdf).

    Args:
      x (FourierSeries): Input field
      c (FourierSeries): Coefficient field

    Returns:
      The Laplacian of the field
    """
  assert x.dims == 1    # Laplacian only defined for scalar fields

  k_vec = params["k_vec"]
  u = x.params[..., 0]
  v = c.params[..., 0]

  def single_coordinate(axis, u):
    u = jnp.moveaxis(u, axis, -1)
    U = x._ffts[0](u, axis=-1)
    U_prime = 1j * k_vec[axis] * U

    # Handle Nyquist frequency
    N_on_L = 1 / u.domain.dx[axis]
    N = u.domain.N[axis]
    if N % 2 == 0:
      U_prime = U_prime.at[...,
                           N // 2].set(U[..., N // 2] * N_on_L * jnp.pi * 1j)

    # Multiply with heterogeneous field
    u_prime = x._ffts[1](U_prime, axis=-1, n=u.shape[-1])
    c_uprime = v * u_prime

    # Get the second derivative
    V = x._ffts[0](c_uprime, axis=-1)
    V_prime = 1j * k_vec[axis] * V

    # Handle Nyquist frequency
    if N % 2 == 0:
      V_prime = V_prime.at[...,
                           N // 2].set(V[..., N // 2] * N_on_L * jnp.pi * 1j)

    # Return to space
    ddu = x._ffts[1](V_prime, axis=-1, n=u.shape[-1])

    return jnp.moveaxis(ddu, -1, axis)

  new_params = jnp.sum(
      jnp.stack([single_coordinate(i, u) for i in range(x.domain.ndim)],
                axis=-1),
      axis=-1,
      keepdims=True,
  )
  return FourierSeries(new_params, x.domain)


if __name__ == "__main__":
    """# Gets implemented functions
    from jaxdf.util import _get_implemented

    funcs = [
      derivative, diag_jacobian, gradient, laplacian,
    ]

    print('differential.py:')
    print('----------------')
    for f in funcs:
      _get_implemented(f)
    print('\n')
    """
