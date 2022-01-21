from jaxdf.core import operator, Params
from jaxdf.discretization import FiniteDifferences, OnGrid
from jax import jit
import jax

def setup_fun(x):
    return {"k": 3}

@jit
@parametrized_operator(setup_fun)
def f(x, params=Params):
    return params["k"]*x

def setup(x, y):
    return {"k": 3}

@jit
@parametrized_operator(setup)
def addp(x: OnGrid, y = 3.2, params=Params):
  return jax.tree_util.tree_multimap(lambda x: x+params["k"] + y, x)

@jit
@operator
def foo(x: OnGrid, params=Params):
  new_params = x.params * 2 + 100
  y = x.replace_params(new_params)
  return y

field = FiniteDifferences(3.0, 'dom')

print('f(1) =', f(1))
print('f(1.0) =', f(1.0))
print(addp(field, 1.3))
print(addp(field, 4.5))
print(addp(field, y=2.11))
print(foo(field))