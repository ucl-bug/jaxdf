import jax
from findiff import coefficients as findif_coeff
from jax import numpy as jnp
from jax import scipy as jsp

from jaxdf.core import operator
from jaxdf.discretization import *


def _get_ffts(x):
  if x.real:
    ffts = [jnp.fft.rfft, jnp.fft.irfft]
  else:
    ffts = [jnp.fft.fft, jnp.fft.ifft]
  return ffts

## derivative
@operator
def derivative(x: Continuous, axis=0, params=None):
  get_x = x.aux['get_field']
  def grad_fun(p, coords):
    f_jac = jax.jacfwd(get_x, argnums=(1,))
    return jnp.expand_dims(f_jac(p, coords)[0][0][axis], -1)
  return Continuous(x.params, x.domain, grad_fun), None

## gradient
@operator
def gradient(x: Continuous, params=None):
  get_x = x.aux['get_field']
  def grad_fun(p, coords):
    f_jac = jax.jacfwd(get_x, argnums=(1,))
    v = f_jac(p, coords)[0]
    return v
  return x.update_fun_and_params(x.params, grad_fun), None


def _convolve_kernel(x, kernel):
  # Make kernel the right size
  extra_pad = (len(kernel) // 2, len(kernel) // 2)
  for ax in range(x.ndim-1):
    kernel = jnp.expand_dims(kernel, axis=0)  # Kernel on the last axis

  # Convolve in each dimension
  outs = []
  img = x.params[...,0]
  for i in range(x.ndim):
    k = jnp.moveaxis(kernel, -1, i)

    pad = [(0, 0)] * x.ndim
    pad[i] = extra_pad
    f = jnp.pad(img, pad, mode="constant")

    out = jsp.signal.convolve(f, k, mode="valid")/x.domain.dx[i]
    outs.append(out)

  new_params = jnp.stack(outs, -1)
  return new_params

def _fd_coefficients(
  order: int = 1,
  accuracy: int = 2,
  staggered: str = 'center'
):
  fd_kernel = findif_coeff(order, accuracy)[staggered]
  coeffs = fd_kernel['coefficients'].tolist()
  offsets = fd_kernel['offsets']

  # Add zeros if needed, to make it work with padding
  if staggered == 'forward':
    coeffs = [0.,]*offsets[-1] + coeffs
  elif staggered == 'backward':
    coeffs = coeffs + [0.,]*(-offsets[0])

  return jnp.asarray(coeffs)

@operator
def gradient(x: FiniteDifferences, params=None, accuracy=2, staggered='center'):
  if params is None:
    params = _fd_coefficients(1, accuracy, staggered)

  kernel = params
  new_params = _convolve_kernel(x, kernel)
  return FiniteDifferences(new_params, x.domain), params

@operator
def gradient(x: FourierSeries, stagger = [0], params=None):
  if params == None:
    params = {'k_vec': x._freq_axis}
  assert x.dims == 1 # Gradient only defined for scalar fields

  if x.real:
    ffts = [jnp.fft.rfft, jnp.fft.irfft]
  else:
    ffts = [jnp.fft.fft, jnp.fft.ifft]
  k_vec = params['k_vec']

  # Adding staggering
  if len(stagger) == 1 and len(x.domain.N) != 1:
    stagger = stagger * len(x.domain.N)

  dx = x.domain.dx
  k_vec = [
    1j * k * jnp.exp(1j * k * s * delta)
    for k, delta, s in zip(k_vec, dx, stagger)
  ]

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
def diag_jacobian(x: Continuous, params=None):
  get_x = x.aux['get_field']
  def diag_fun(p, coords):
    f_jac = jax.jacfwd(get_x, argnums=(1,))
    return jnp.diag(f_jac(p, coords)[0])
  return x.update_fun_and_params(x.params, diag_fun), None

@operator
def diag_jacobian(x: FiniteDifferences, params=None, accuracy=2, staggered='center'):
  if params is None:
    params = _fd_coefficients(1, accuracy, staggered)

  outs = []
  img = x.params
  kernel = params

  # Make kernel the right size
  extra_pad = (len(kernel) // 2, len(kernel) // 2)
  for ax in range(x.ndim-1):
    kernel = jnp.expand_dims(kernel, axis=0)  # Kernel on the last axis

  for ax in range(x.ndim):
    img_shifted = jnp.moveaxis(img[...,ax], ax, -1)
    pad = [(0, 0)] * x.ndim
    pad[-1] = extra_pad
    f = jnp.pad(img_shifted, pad, mode="constant")
    out = jsp.signal.convolve(f, kernel, mode="valid")/x.domain.dx[ax]
    out = jnp.moveaxis(out, -1, ax)
    outs.append(out)

  outs = jnp.stack(outs, axis=-1)

  return FiniteDifferences(outs, x.domain), params


@operator
def diag_jacobian(x: FourierSeries, stagger = [0], params=None):
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
def laplacian(x: Continuous, params=None):
  get_x = x.aux['get_field']
  def grad_fun(p, coords):
    hessian = jax.hessian(get_x, argnums=(1,))(p,coords)[0][0][0]
    return jnp.diag(hessian)
  return x.update_fun_and_params(x.params, grad_fun), None


@operator
def laplacian(x: FourierSeries, params=None):
  if params == None:
    params = {'k_vec': x._freq_axis}
  assert x.dims == 1 # Laplacian only defined for scalar fields

  ffts = _get_ffts(x)
  k_vec = params["k_vec"]
  u = x.params[...,0]

  def single_grad(axis, u):
    u = jnp.moveaxis(u, axis, -1)
    Fx = ffts[0](u, axis=-1)
    iku = -Fx * k_vec[axis] ** 2
    du = ffts[1](iku, axis=-1, n=u.shape[-1])
    return jnp.moveaxis(du, -1, axis)

  new_params = jnp.sum(
        jnp.stack([single_grad(i, u) for i in range(x.ndim)], axis=-1),
        axis=-1,
        keepdims=True,
    )
  return FourierSeries(new_params, x.domain), params

@operator
def laplacian(x: FiniteDifferences, params=None, accuracy=4):
  if params == None:
    coeffs = {
      2: [1, -2, 1],
      4: [-1 / 12, 4 / 3, -5/2, 4 / 3, -1 / 12],
      6: [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90],
    }
    params = {"laplacian_kernel": jnp.asarray(coeffs[accuracy])}

  kernel = params["laplacian_kernel"]
  new_params = _convolve_kernel(x, kernel)
  return FiniteDifferences(new_params, x.domain), params


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
