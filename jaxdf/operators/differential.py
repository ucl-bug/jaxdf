import jax
import numpy as np
from jax import numpy as jnp
from jax import scipy as jsp

from jaxdf.core import operator
from jaxdf.discretization import *


def _convolve_with_pad(
  kernel: jnp.ndarray,
  array: jnp.ndarray,
  axis: int
) -> jnp.ndarray:
  r'''Convolves an array with a kernel, using reflection padding.
  The kernel is supposed to be with the same number of indices as the array,
  but the only dimension different than 1 corresponds to the axis. Padding
  is only applied to such axis.
  Parameters:
    kernel (jnp.ndarray): The kernel to convolve with.
    array (jnp.ndarray): The array to convolve.
  Returns:
    jnp.ndarray: The convolved array.
  '''
  # Reflection padding the array where appropriate
  pad_size = max(kernel.shape)//2
  extra_pad = (pad_size,pad_size)
  pad = [(0, 0)] * array.ndim
  pad[axis] = extra_pad
  f = jnp.pad(array, pad, mode="wrap")

  # Apply kernel
  out = jsp.signal.convolve(f, kernel, mode="valid")

  return out

def _get_ffts(x):
  if x.real:
    ffts = [jnp.fft.rfft, jnp.fft.irfft]
  else:
    ffts = [jnp.fft.fft, jnp.fft.ifft]
  return ffts

## derivative
@operator
def derivative(x: Continuous, *, axis=0, params=None):
  get_x = x.aux['get_field']
  def grad_fun(p, coords):
    f_jac = jax.jacfwd(get_x, argnums=(1,))
    return jnp.expand_dims(f_jac(p, coords)[0][0][axis], -1)
  return Continuous(x.params, x.domain, grad_fun), None

def _bubble_sort_gridpoints(grid_points):
    # Sorts by distance from 0
    # [-3, -2, -1, 0, 1, 2, 3] -> [0, 1, -1, 2, -2, 3, -3]
    # [0.5, 1.5, -0.5, 2.5, -1.5, -2.5] -> [0.5, -0.5, 1.5, -1.5, 2.5, -2.5]
    for i in range(len(grid_points)):
        for j in range(0, len(grid_points) - i - 1):
            magnitude_condition = abs(grid_points[j]) > abs(grid_points[j + 1])
            same_mag_condition = abs(grid_points[j]) == abs(grid_points[j + 1])
            sign_condition = np.sign(grid_points[j]) < np.sign(grid_points[j + 1])
            if  magnitude_condition or (same_mag_condition and sign_condition):
                temp = grid_points[j]
                grid_points[j] = grid_points[j+1]
                grid_points[j+1] = temp

    return grid_points

def _fd_coefficients_fornberg(order, grid_points, x0 = 0):
  # from Generation of Finite Difference Formulas on Arbitrarily Spaced Grids
  # Bengt Fornberg, 1998
  # https://web.njit.edu/~jiang/math712/fornberg.pdf
  M = order
  N = len(grid_points) - 1

  # Sort the grid points
  alpha = _bubble_sort_gridpoints(grid_points)
  delta = dict() # key: (m,n,v)
  delta[(0,0,0)] = 1.
  c1 = 1.

  for n in range(1, N+1):
    c2 = 1.
    for v in range(n):
      c3 = alpha[n] - alpha[v]
      c2 = c2 * c3
      if n < M:
        delta[(n,n-1,v)] = 0.
      for m in range(min([n, M])+1):
        d1 = delta[(m,n-1,v)] if (m,n-1,v) in delta.keys() else 0.
        d2 = delta[(m-1, n-1, v)] if (m-1,n-1,v) in delta.keys() else 0.
        delta[(m,n,v)] = ((alpha[n] - x0)*d1 - m*d2)/c3

    for m in range(min([n,M])+1):
      d1 = delta[(m-1, n-1, n-1)] if (m-1,n-1,n-1) in delta.keys() else 0.
      d2 = delta[(m,n-1,n-1)] if (m,n-1,n-1) in delta.keys() else 0.
      delta[(m,n,n)] = (c1/c2)*(m*d1 - (alpha[n-1] - x0)*d2)
    c1 = c2

  # Extract the delta with m = M and n = N
  coeffs = [None]*(N+1)
  for key in delta:
    if key[0] == M and key[1] == N:
      coeffs[key[2]] = delta[key]

  # sort coefficeient and alpha by alpha
  idx = np.argsort(alpha)
  alpha = np.take_along_axis(np.asarray(alpha),idx, axis=-1)
  coeffs = np.take_along_axis(np.asarray(coeffs),idx, axis=-1)

  return coeffs, alpha

def _get_fd_coefficients(x: FiniteDifferences, order=1, stagger = 0):
  # Check that all the values of stagger are in [0, 0.5, -0.5]
  assert stagger in [0, -0.5, 0.5], 'Staggering must be in [0, 0.5, -0.5] for finite differences'
  dx = np.asarray(x.domain.dx)
  accuracy = x.accuracy
  points = np.arange(-accuracy//2, accuracy//2+1)
  if stagger > 0:
    points = (points + stagger)[:-1]
  elif stagger < 0:
    points = (points + stagger)[1:]

  # get coefficients
  coeffs = _fd_coefficients_fornberg(order, points, x0 = 0)[0].tolist()

  # Append zero if a coefficient has been removed
  if stagger > 0:
    coeffs = coeffs + [0.]
  elif stagger < 0:
    coeffs = [0.] + coeffs

  return np.asarray(coeffs)

def fd_derivative_init(
  x: FiniteDifferences,
  *,
  axis=0,
  stagger = 0
):
  kernel = _get_fd_coefficients(x, order = 1, stagger=stagger)

  if x.domain.ndim > 1:
    for _ in range(x.domain.ndim - 1):
      kernel = np.expand_dims(kernel, axis=0)
    # Move kernel to the correct axis
    kernel = np.moveaxis(kernel, -1, axis)

  # Add dx
  kernel = kernel / x.domain.dx[axis]

  return kernel

def ft_diag_jacobian_init(
  x: FiniteDifferences,
  *,
  stagger = [0]
):
  if len(stagger) != x.domain.ndim:
    stagger = [stagger[0] for _ in range(x.domain.ndim)]

  kernels = []
  for i in range(x.domain.ndim):
    kernels.append(fd_derivative_init(x, axis=i, stagger=stagger[i]))

  return kernels

## gradient
@operator
def gradient(x: Continuous, *, params=None):
  get_x = x.aux['get_field']
  def grad_fun(p, coords):
    f_jac = jax.jacfwd(get_x, argnums=(1,))
    v = f_jac(p, coords)[0]
    return v
  return x.update_fun_and_params(x.params, grad_fun), None


@operator(init_params=ft_diag_jacobian_init)
def gradient(
  x: FiniteDifferences,
  *,
  stagger = [0],
  params = None
) -> FiniteDifferences:
  if params is None:
    params = ft_diag_jacobian_init(x, stagger=stagger)
  return diag_jacobian(x, stagger, params=params), params

@operator
def gradient(
  x: FourierSeries,
  *,
  stagger = [0],
  correct_nyquist = True,
  params=None
) -> FourierSeries:
  r"""
  Args:
      x (FourierSeries): Input field
      stagger (list, optional): Staggering value for the returned fields.
        The fields are staggered in the direction of their derivative.
        Defaults to [0].
      correct_nyquist (bool, optional): If `True`, uses a correction of the
        derivative filter for the Nyquist frequency, which preserves Hermitian
        symmetric and null space. See [those notes](https://math.mit.edu/~stevenj/fft-deriv.pdf)
        for more details. Defaults to True.
      params (Any, optional): Operator parameters. Defaults to None.

  Returns:
      FourierSeries: The gradient of the input field.
  """
  if params == None:
    params = {'k_vec': x._freq_axis}
  assert x.dims == 1 # Gradient only defined for scalar fields

  ffts = _get_ffts(x)
  k_vec = params['k_vec']

  # Adding staggering
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
        k_vec[f] = k_vec[f].at[f_nyq].set(0.)

  u = x.params[...,0]

  def single_grad(axis, u):
    u = jnp.moveaxis(u, axis, -1)
    Fx = ffts[0](u, axis=-1)
    iku = Fx * k_vec[axis]
    du = ffts[1](iku, axis=-1, n=u.shape[-1])
    return jnp.moveaxis(du, -1, axis)

  new_params = jnp.stack([single_grad(i, u) for i in range(x.ndim)], axis=-1)
  return FourierSeries(new_params, x.domain), params

# diag_jacobian
@operator
def diag_jacobian(x: Continuous, *, params=None):
  get_x = x.aux['get_field']
  def diag_fun(p, coords):
    f_jac = jax.jacfwd(get_x, argnums=(1,))
    return jnp.diag(f_jac(p, coords)[0])
  return x.update_fun_and_params(x.params, diag_fun), None

@operator(init_params=ft_diag_jacobian_init)
def diag_jacobian(
  x: FiniteDifferences,
  *,
  stagger = [0],
  params = None
) -> FiniteDifferences:
  if params == None:
    params = ft_diag_jacobian_init(x, stagger)

  kernels = params
  array = x.on_grid

  # Apply the corresponding kernel to each dimension
  outs = [_convolve_with_pad(kernels[i], array[...,i], i) for i in range(x.ndim)]
  new_params = jnp.stack(outs, axis=-1)

  return x.replace_params(new_params)


@operator
def diag_jacobian(
  x: FourierSeries,
  *,
  stagger = [0],
  correct_nyquist = True,
  params=None
) -> FourierSeries:
  r"""
  Args:
      x (FourierSeries): Input field
      stagger (list, optional): Staggering value for the returned fields.
        The fields are staggered in the direction of their derivative.
        Defaults to [0].
      correct_nyquist (bool, optional): If `True`, uses a correction of the
        derivative filter for the Nyquist frequency, which preserves Hermitian
        symmetric and null space. See [those notes](https://math.mit.edu/~stevenj/fft-deriv.pdf)
        for more details. Defaults to True.
      params (Any, optional): Operator parameters. Defaults to None.

  Returns:
      FourierSeries: The vector field whose components are the diagonal entries
        of the Jacobian of the input field.
  """
  if params == None:
    params = {'k_vec': x._freq_axis}

  ffts = _get_ffts(x)
  k_vec = params["k_vec"]

  # Adding staggering
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
        k_vec[f] = k_vec[f].at[f_nyq].set(0.)

  new_params = jnp.zeros_like(x.params)

  def single_grad(axis, u):
    u = jnp.moveaxis(u, axis, -1)
    Fx = ffts[0](u, axis=-1)
    iku = Fx * k_vec[axis]
    du = ffts[1](iku, axis=-1, n=u.shape[-1])
    return jnp.moveaxis(du, -1, axis)

  for ax in range(x.ndim):
    new_params = new_params.at[..., ax].set(single_grad(ax, x.params[..., ax]))

  return FourierSeries(new_params, x.domain), params

# laplacian
@operator
def laplacian(x: Continuous, *, params=None):
  get_x = x.aux['get_field']
  def grad_fun(p, coords):
    hessian = jax.hessian(get_x, argnums=(1,))(p,coords)[0][0][0]
    return jnp.diag(hessian)
  return x.update_fun_and_params(x.params, grad_fun), None


@operator
def laplacian(x: FourierSeries, *, params=None):
  if params == None:
    params = {'k_vec': x._freq_axis}
  assert x.dims == 1 # Laplacian only defined for scalar fields

  ffts = _get_ffts(x)
  k_vec = params["k_vec"]
  u = x.params[...,0]

  def single_grad(axis, u):
    u = jnp.moveaxis(u, axis, -1)
    Fx = ffts[0](u, axis=-1)
    iku = -Fx * (k_vec[axis] ** 2)
    du = ffts[1](iku, axis=-1, n=u.shape[-1])
    return jnp.moveaxis(du, -1, axis)

  new_params = jnp.sum(
        jnp.stack([single_grad(i, u) for i in range(x.ndim)], axis=-1),
        axis=-1,
        keepdims=True,
    )
  return FourierSeries(new_params, x.domain), params

@operator
def heterog_laplacian(x: FourierSeries, c: FourierSeries, *, params=None):
  '''Computes the position-varying laplacian using algorithm 4 of
  https://math.mit.edu/~stevenj/fft-deriv.pdf.
  '''
  if params == None:
    params = {'k_vec': x._freq_axis}
  assert x.dims == 1 # Laplacian only defined for scalar fields

  ffts = _get_ffts(x)
  k_vec = params["k_vec"]
  u = x.params[...,0]
  v = c.params[...,0]

  def single_coordinate(axis, u):
    u = jnp.moveaxis(u, axis, -1)
    U = ffts[0](u, axis=-1)
    U_prime = 1j * k_vec[axis] * U

    # Handle Nyquist frequency
    N_on_L = 1/u.domain.dx[axis]
    N = u.domain.N[axis]
    if N % 2 == 0:
      U_prime = U_prime.at[..., N//2].set(
        U[..., N//2]*N_on_L*jnp.pi*1j
      )

    # Multiply with heterogeneous field
    u_prime = ffts[1](U_prime, axis=-1, n=u.shape[-1])
    c_uprime = v*u_prime

    # Get the second derivative
    V = ffts[0](c_uprime, axis=-1)
    V_prime = 1j * k_vec[axis] * V

    # Handle Nyquist frequency
    if N % 2 == 0:
      V_prime = V_prime.at[..., N//2].set(
        V[..., N//2]*N_on_L*jnp.pi*1j
      )

    # Return to space
    ddu = ffts[1](V_prime, axis=-1, n=u.shape[-1])

    return jnp.moveaxis(ddu, -1, axis)

  new_params = jnp.sum(
        jnp.stack([single_coordinate(i, u) for i in range(x.ndim)], axis=-1),
        axis=-1,
        keepdims=True,
    )
  return FourierSeries(new_params, x.domain), params


if __name__ == '__main__':
  from jaxdf.util import _get_implemented

  funcs = [
    derivative, diag_jacobian, gradient, laplacian,
  ]

  print('differential.py:')
  print('----------------')
  for f in funcs:
    _get_implemented(f)
  print('\n')
