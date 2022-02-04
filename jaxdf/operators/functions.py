from jaxdf.core import operator, Params, params_map
from jaxdf.discretization import *
from jaxdf.discretization import OnGrid
from jax import numpy as jnp


## compose
@operator
def compose(x: Continuous, params=None):
  r'''Applies function composition on the `get_fun` of the Continuous object.
  '''
  get_x = x.aux['get_field']
  def decorator(fun):
    def new_fun(p, coord):
      return fun(get_x(p, coord))
    return Continuous(x.params, x.domain, new_fun)
  return decorator, None

@operator
def compose(x: OnGrid, params=Params):
  r'''Maps the given function over the pytree of parameters
  of the `Field`.
  
  !!! example
      ```python
      x = OnGrid(params=-1.0, ...)
      
      # Applies the absolute value function to the parameters
      y = compose(x)(jnp.abs)
      
      y.params # This is 1.0
      ```
  '''
  def decorator(fun):
    return x.replace_params(fun(x.params))
  return decorator, None

## shift_operator
@operator
def shift_operator(x: Continuous, dx: object, params=None):
  get_x = x.aux['get_field']
  def fun(p, coord):
    return get_x(p, coord + dx)
  return Continuous(x.params, x.domain, fun), None

## sum_over_dims
@operator
def sum_over_dims(x: Continuous, params = None):
  get_x = x.aux['get_field']
  def fun(p, coords):
    return jnp.sum(get_x(p, coords), axis=-1, keepdims=True)
  return x.update_fun_and_params(x.params, fun), None

@operator
def sum_over_dims(x: OnGrid, params = None):
  new_params = jnp.sum(x.params, axis=-1, keepdims=True)
  return x.replace_params(new_params), None


if __name__ == '__main__':
  from jaxdf.util import _get_implemented
  
  print(compose.__name__)
  print(compose.__doc__)
  funcs = [
    compose, sum_over_dims
  ]

  print('functions.py:')
  print('----------------')
  for f in funcs:
    _get_implemented(f)
  print('\n')