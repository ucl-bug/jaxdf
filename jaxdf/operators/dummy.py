from jaxdf.core import operator, Params
from jaxdf.discretization import *
from jax import jit


@operator
def dummy(x: OnGrid, params=Params):
  r'''A dummy operator that is useful for debugging.'''
  if params == Params:
    params = {"k": 3}
  return params["k"]*x, params

@operator
def dummy(x: Continuous, params=Params):
  if params == Params:
    params = {"k": 3}
  get_x = x.aux['get_field']
  def get_fun(p__par, coords):
    p, params = p__par
    return get_x(p, coords) + params['k']
  return x.update_fun_and_params([x.params, params], get_fun), params