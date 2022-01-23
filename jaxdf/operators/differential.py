from jaxdf.core import operator, Params, params_map
from jaxdf.discretization import *
from jax import numpy as jnp
import jax

## gradient
@operator
def gradient(x: Continuous, params=Params):
  get_x = x.aux['get_field']
  def grad_fun(p, coords):
    f_jac = jax.jacfwd(get_x, argnums=(1,))
    return f_jac(p, coords)[0]
  return x.update_fun_and_params(x.params, grad_fun), None

def _setup_k_vec(x: FourierSeries, params=Params):
  return {'k_vec': x._freq_axis}

@operator
def gradient(x: FourierSeries, params=Params):
  if params == Params:
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
