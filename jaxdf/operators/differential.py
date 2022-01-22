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
  return x.update_fun_and_params(x.params, grad_fun)