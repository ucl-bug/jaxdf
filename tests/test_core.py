from jaxdf import *
from jax import jit, make_jaxpr
import inspect
import numpy as np
from jax import numpy as jnp

ATOL=1e-6

domain = geometry.Domain()

# Fields on grid
x = OnGrid(1.0, domain)
y = OnGrid(2.0, domain)

# Continuous fields
def f(p, x):
  return p + x
a = Continuous.from_fun_and_params(5.0, domain, f)
b = Continuous.from_fun_and_params(6.0, domain, f)

def test_override_operator():
  z = operators.compose(x)(jnp.exp)
  assert z.params == jnp.exp(1.0)
  
  @operator
  def compose(x: OnGrid, params=Params):
    def decorator(fun):
      return x.replace_params(fun(x.params) + 100)
    return decorator
  
  z = operators.compose(x)(jnp.exp)
  assert z.params == jnp.exp(1.0) + 100
  
    
if __name__ == '__main__':
  test_override_operator()