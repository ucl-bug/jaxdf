from jaxdf.core import operator, Params, params_map
from jaxdf.discretization import *
from jax import tree_util
from jaxdf.discretization import OnGrid
from jax import eval_shape
from typing import Callable

## compose
@operator
def compose(x: Continuous, params=Params):
  def decorator(fun):
    def new_fun(p, coord):
      return fun(x.get_field(p, coord))
    return Continuous.from_fun_and_params(x.params, x.domain, new_fun)
  return decorator

@operator
def compose(x: OnGrid, params=Params):
  def decorator(fun):
    return x.replace_params(fun(x.params))
  return decorator