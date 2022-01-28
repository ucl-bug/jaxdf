from jaxdf.core import operator, Params, params_map
from jaxdf.discretization import *
from jax import numpy as jnp

"""
This file contains the operators that are
bind with the magic functions of fields
"""


## __add__ 
@operator
def __add__(x: Linear, y: Linear, params=Params):
  new_params = params_map(lambda x,y: x+y, x.params, y.params)
  return x.replace_params(new_params), None

@operator(precedence=-1)
def __add__(x: OnGrid, y: object, params=Params):
  new_params = params_map(lambda x: x+y, x.params)
  return x.replace_params(new_params), None

@operator
def __add__(x: Continuous, y: Continuous, params=Params):
  get_x = x.aux['get_field']
  get_y = y.aux['get_field']
  def get_fun(p, coords):
    return get_x(p[0], coords) + get_y(p[1], coords)
  return Continuous([x.params, y.params], x.domain, get_fun), None

@operator
def __add__(x: Continuous, y: object, params=Params):
  get_x = x.aux['get_field']
  def get_fun(p, coords):
    return get_x(p, coords) + y
  return Continuous(x.params, x.domain, get_fun), None

## __bool__
@operator
def __bool__(x: OnGrid, params=Params):
  new_params = params_map(lambda x: bool(x), x.params)
  return x.replace_params(new_params), None


## __divmod__
@operator
def __divmod__(x: OnGrid, y: OnGrid, params=Params):
  new_params = params_map(lambda x,y: divmod(x, y), x.params,y.params)
  return x.replace_params(new_params), None

@operator
def __divmod__(x: Linear, y, params=Params):
  new_params = params_map(lambda x: divmod(x, y), x.params)
  return x.replace_params(new_params), None


## __float__
@operator
def __float__(x: OnGrid, params=Params):
  new_params = params_map(lambda x: float(x), x.params)
  return x.replace_params(new_params), None


## __mul__
@operator
def __mul__(x: OnGrid, y: OnGrid, params=Params):
  new_params = params_map(lambda x,y: x*y, x.params,y.params)
  return x.replace_params(new_params), None

@operator(precedence=-1)
def __mul__(x: Linear, y, params=Params):
  new_params = params_map(lambda x: x*y, x.params)
  return x.replace_params(new_params), None

@operator
def __mul__(x: Continuous, y: Continuous, params=Params):
  get_x = x.aux['get_field']
  get_y = y.aux['get_field']
  def get_fun(p, coords):
    return get_x(p[0], coords) * get_y(p[1], coords)
  return x.update_fun_and_params([x.params, y.params], get_fun), None

@operator
def __mul__(x: Continuous, y, params=Params):
  get_x = x.aux['get_field']
  def get_fun(p, coords):
    return get_x(p, coords) * y
  return x.update_fun_and_params(x.params, get_fun), None

## __neg__
@operator
def __neg__(x: Linear, params=Params):
  new_params = params_map(lambda x: -x, x.params)
  return x.replace_params(new_params), None

@operator
def __neg__(x: Continuous, params=Params):
  get_x = x.aux['get_field']
  def get_fun(p, coords):
    return -get_x(p, coords)
  return Continuous(x.params, x.domain, get_fun), None
  

## __pow__
@operator
def __pow__(x: OnGrid, y: OnGrid, params=Params):
  new_params = params_map(lambda x,y: x**y, x.params,y.params)
  return x.replace_params(new_params), None

@operator(precedence=-1)
def __pow__(x: OnGrid, y: object, params=Params):
  new_params = params_map(lambda x: x**y, x.params)
  return x.replace_params(new_params), None

@operator
def __pow__(x: Continuous, y: Continuous, params=Params):
  get_x = x.aux['get_field']
  get_y = y.aux['get_field']
  def get_fun(p, coords):
    return get_x(p[0], coords) ** get_y(p[1], coords)
  return x.update_fun_and_params([x.params, y.params], get_fun), None

@operator
def __pow__(x: Continuous, y: object, params=Params):
  get_x = x.aux['get_field']
  def get_fun(p, coords):
    return get_x(p, coords) ** y
  return Continuous(x.params, x.domain, get_fun), None


## __radd__
@operator(precedence=-1)
def __radd__(x: OnGrid, y: object, params=Params):
  return x + y, None

## __rmul__
@operator(precedence=-1)
def __rmul__(x: Field, y: object, params=Params):
  return x * y, None


@operator(precedence=-1)
def __radd__(x: Continuous, y: object, params=Params):
  return x + y, None


## __rpow__
@operator(precedence=-1)
def __rpow__(x: OnGrid, y: object, params=Params):
  new_params = params_map(lambda x: x**y, x.params)
  return x.replace_params(new_params), None


## __rsub__
@operator
def __rsub__(x: Linear, y: object, params=Params):
  return (-x) + y, None


## __rtruediv__
@operator
def __rtruediv__(x: OnGrid, y: object, params=Params):
  new_params = params_map(lambda x: y/x, x.params)
  return x.replace_params(new_params), None

@operator(precedence=-1)
def __rtruediv__(x: Continuous, y: object, params=Params):
  get_x = x.aux['get_field']
  def get_fun(p, coords):
    return y / get_x(p, coords)
  return Continuous(x.params, x.domain, get_fun), None

## __sub__
@operator
def __sub__(x: Linear, y: Linear, params=Params):
  new_params = params_map(lambda x,y: x-y, x.params,y.params)
  return x.replace_params(new_params), None

@operator(precedence=-1)
def __sub__(x: OnGrid, y: object, params=Params):
  new_params = params_map(lambda x: x-y, x.params)
  return x.replace_params(new_params), None

## __truediv__
@operator
def __truediv__(x: Continuous, y: Continuous, params=Params):
  get_x = x.aux['get_field']
  get_y = y.aux['get_field']
  def get_fun(p, coords):
    return get_x(p[0], coords) / get_y(p[1], coords)
  return Continuous([x.params, y.params], x.domain, get_fun), None

@operator
def __truediv__(x: Continuous, y: object, params=Params):
  get_x = x.aux['get_field']
  def get_fun(p, coords):
    return get_x(p, coords) / y
  return Continuous(x.params, x.domain, get_fun), None

@operator
def __truediv__(x: OnGrid, y: OnGrid, params=Params):
  new_params = params_map(lambda x,y: x/y, x.params,y.params)
  return x.replace_params(new_params), None

@operator(precedence=-1)
def __truediv__(x: Linear, y, params=Params):
  new_params = params_map(lambda x: x/y, x.params)
  return x.replace_params(new_params), None

## inverse
@operator
def inverse(x: OnGrid, params=Params):
  new_params = params_map(lambda x: 1/x, x.params)
  return x.replace_params(new_params), None