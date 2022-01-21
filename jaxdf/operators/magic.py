from jaxdf.core import operator, Params, params_map
from jaxdf.discretization import *
from jax import tree_util
from jaxdf.discretization import OnGrid
from jax import ShapedArray

"""
This file contains the operators that are
bind with the magic functions of fields
"""

## __add__ 
@operator
def __add__(x: Linear, y: Linear, params=Params):
  return params_map(lambda x, y: x+y, x, y)

@operator(precedence=-1)
def __add__(x: OnGrid, y, params=Params):
  new_params = params_map(lambda x: x+y, x.params)
  return x.replace_params(new_params)


## __bool__
@operator
def __bool__(x: OnGrid, params=Params):
  return params_map(lambda x: bool(x), x)


## __divmod__
@operator
def __divmod__(x: OnGrid, y: OnGrid, params=Params):
  return params_map(lambda x, y: divmod(x, y), x, y)

@operator
def __divmod__(x: Linear, y, params=Params):
  new_params = params_map(lambda x: divmod(x, y), x.params)
  return x.replace_params(new_params)


## __float__
@operator
def __float__(x: OnGrid, params=Params):
  return params_map(lambda x: float(x), x)


## __mul__
@operator
def __mul__(x: OnGrid, y: OnGrid, params=Params):
  return params_map(lambda x, y: x*y, x, y)

@operator(precedence=1)
def __mul__(x: Linear, y, params=Params):
  new_params = params_map(lambda x: x*y, x.params)
  return x.replace_params(new_params)


## __neg__
@operator
def __neg__(x: Linear, params=Params):
  return params_map(lambda x: -x, x)
  

## __pow__
@operator
def __pow__(x: OnGrid, y: OnGrid, params=Params):
  return params_map(lambda x, y: x**y, x, y)

@operator(precedence=1)
def __pow__(x: OnGrid, y, params=Params):
  new_params = params_map(lambda x: x**y, x.params)
  return x.replace_params(new_params)


## __rpow__
@operator
def __rpow__(x: OnGrid, y, params=Params):
  return params_map(lambda x: y**x, x)


## __rsub__
@operator
def __rsub__(x: Linear, y, params=Params):
  return (-x) + y


## __sub__
@operator
def __sub__(x: Linear, y: Linear, params=Params):
  return params_map(lambda x, y: x-y, x, y)

@operator(precedence=-1)
def __sub__(x: OnGrid, y: object, params=Params):
  return params_map(lambda x: x-y, x)