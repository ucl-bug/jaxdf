from jaxdf.core import operator, parametrized_operator, Params
from jaxdf.discretization import *
from jax import tree_util
from jaxdf.discretization import OnGrid
from jax import ShapedArray

## __add__ 
@operator
def __add__(x: OnGrid, y: OnGrid, params=Params):
  assert type(x) == type(y)
  return tree_util.tree_multimap(lambda x, y: x+y, x, y)

@operator
def __add__(x: OnGrid, y, params=Params):
  new_params = tree_util.tree_multimap(lambda x: x+y, x.params)
  return x.replace_params(new_params)


## __mul__
@operator
def __mul__(x: OnGrid, y: OnGrid, params=Params):
  assert type(x) == type(y)
  return tree_util.tree_multimap(lambda x, y: x*y, x, y)

@operator
def __mul__(x: OnGrid, y, params=Params):
  new_params = tree_util.tree_multimap(lambda x: x*y, x.params)
  return x.replace_params(new_params)