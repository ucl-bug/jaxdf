from jaxdf.core import operator, Params, params_map
from jaxdf.discretization import *
from .functions import sum_over_dims
from jax import numpy as jnp
import jax
from jax import scipy as jsp


def _get_ffts(x):
  if x.real:
    ffts = [jnp.fft.rfft, jnp.fft.irfft]
  else:
    ffts = [jnp.fft.fft, jnp.fft.ifft]
  return ffts
  

## gradient
@operator
def gradient(x: Continuous, params=Params):
  get_x = x.aux['get_field']
  def grad_fun(p, coords):
    f_jac = jax.jacfwd(get_x, argnums=(1,))
    return f_jac(p, coords)[0][0]
  return x.update_fun_and_params(x.params, grad_fun), None

@operator
def gradient(x: FourierSeries, params=None):
  if params == None:
    params = {'k_vec': x._freq_axis}
  assert x.dims == 1 # Gradient only defined for scalar fields

  if x.real:
    ffts = [jnp.fft.rfft, jnp.fft.irfft]
  else:
    ffts = [jnp.fft.fft, jnp.fft.ifft]
  k_vec = params['k_vec']
  u = x.params[...,0]
  
  def single_grad(axis, u):
    u = jnp.moveaxis(u, axis, -1)
    Fx = ffts[0](u, axis=-1)
    iku = 1j * Fx * k_vec[axis]
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

    out = jsp.signal.convolve(f, k, mode="valid")*x.domain.dx[i]
    outs.append(out)
  
  new_params = jnp.expand_dims(sum(outs), -1)
  return FiniteDifferences(new_params, x.domain), params