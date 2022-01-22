from jaxdf.core import operator, Params
from jaxdf.discretization import *
from jax import jit


def setup_dummy(x: OnGrid):
  return {"k": 3}

@operator(setup_fun=setup_dummy)
def dummy(x: OnGrid, params=Params):
  return params["k"]*x


@operator(setup_fun=setup_dummy)
def dummy(x: Continuous, params=Params):
  get_x = x.aux['get_field']
  def get_fun(p__par, coords):
    p, params = p__par
    return get_x(p, coords) + params['k']
  return x.update_fun_and_params([x.params, params], get_fun)