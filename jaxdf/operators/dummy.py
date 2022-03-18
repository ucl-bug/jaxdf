from jaxdf.core import operator
from jaxdf.discretization import *


@operator
def dummy(x: OnGrid, params=None):
  r'''A dummy operator that is useful for debugging.'''
  if params is None:
    params = {"k": 3}
  return params["k"]*x, params

@operator
def dummy(x: Continuous, params=None):
  if params is None:
    params = {"k": 3}
  get_x = x.aux['get_field']
  def get_fun(p__par, coords):
    p, params = p__par
    return get_x(p, coords) + params['k']
  return x.update_fun_and_params([x.params, params], get_fun), params
